"""Encoder-decoder Transformer over jagged NestedTensors.

Inputs are batched as `torch.nested` jagged tensors of shape ``(B, j1)`` where
``B`` is batch (dim 0) and ``j1`` is the per-sequence length (dim 1). The jagged
layout removes padding masks: every kernel sees only real tokens. Blocks compose
with ``torch.compile``.

Attention dispatch:
  * Non-causal paths (encoder self-attn, decoder cross-attn) use
    ``F.scaled_dot_product_attention``, which selects FlashAttention on CUDA.
  * Causal jagged self-attn uses ``flex_attention`` with a nested block mask;
    SDPA's ``is_causal=True`` path is not implemented for jagged inputs.

Notes for CPU:
  * Forward-only inference works on CPU (SDPA dispatches to MATH; FlexAttention
    runs eager). Backward through FlexAttention on jagged is currently broken
    on CPU (a PyTorch issue) — train on CUDA.

Conventions:
  * Pre-norm residuals (LayerNorm before each sublayer)
  * RoPE on self-attention only; cross-attention reads positional context from
    the encoder's already-contextualized memory
  * Packed projections: QKV for self-attention, KV for cross-attention
  * No padding tokens, no attention masks, no per-sequence length tensors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_nested_block_mask


def _causal_mask_mod(_b, _h, q_idx, kv_idx):
    return q_idx >= kv_idx


@torch._dynamo.disable  # flex_attention's jagged path is not traceable by AOT autograd (PyTorch 2.9)
def _causal_jagged_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    block_mask = create_nested_block_mask(_causal_mask_mod, B=None, H=None, q_nt=q)
    return flex_attention(q, k, v, block_mask=block_mask)


def from_token_ids(seqs: list[torch.Tensor]) -> torch.Tensor:
    """Pack variable-length 1-D LongTensors into a jagged ``(B, j1)`` NestedTensor."""
    return torch.nested.nested_tensor(list(seqs), layout=torch.jagged)


class RotaryEmbedding(nn.Module):
    """RoPE that operates on flat values + offsets (jagged) or dense ``(B, L, H, D)``."""

    def __init__(self, d_head: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        assert d_head % 2 == 0, "d_head must be even"
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head))
        freqs = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), inv_freq)
        self.register_buffer("cos_cache", freqs.cos(), persistent=False)
        self.register_buffer("sin_cache", freqs.sin(), persistent=False)

    @staticmethod
    def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)

    def apply_jagged(self, values: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """Rotate flat values ``(T, H, D)`` given jagged ``offsets``."""
        # Operating on flat values (rather than on a nested view) avoids creating an
        # intermediate nested wrapper that confuses FlexAttention's shape tracking.
        lengths = offsets.diff()
        base = torch.repeat_interleave(offsets[:-1], lengths)
        positions = torch.arange(values.size(0), device=values.device) - base
        cos = self.cos_cache[positions].unsqueeze(1).to(values.dtype)          # (T, 1, D/2)
        sin = self.sin_cache[positions].unsqueeze(1).to(values.dtype)
        return self._rotate(values, cos, sin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dense path: x is (B, L, H, D)
        L = x.size(1)
        cos = self.cos_cache[:L].view(1, L, 1, -1).to(x.dtype)
        sin = self.sin_cache[:L].view(1, L, 1, -1).to(x.dtype)
        return self._rotate(x, cos, sin)


class SwiGLU(nn.Module):
    """SwiGLU FFN: ``w2(silu(w1 x) * w3 x)`` with a packed ``w13`` projection."""

    def __init__(self, d_model: int, d_ff: int, multiple_of: int = 64):
        super().__init__()
        d_ff = -(-d_ff // multiple_of) * multiple_of
        self.w13 = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w13(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class MultiHeadAttention(nn.Module):
    """Multi-head attention.

    Self-attention uses a packed QKV projection and applies RoPE to ``q`` / ``k``.
    Cross-attention uses a separate Q projection and a packed KV projection,
    which is more efficient when the same memory feeds many decoder layers.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 8192,
        cross_attn: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.cross_attn = cross_attn

        if cross_attn:
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        else:
            self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
            self.rope = RotaryEmbedding(self.d_head, max_seq_len)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        # `.contiguous()` after `chunk` collapses the view's inherited stride from the
        # packed projection — jagged SDPA rejects values with non-standard strides.
        if self.cross_attn:
            q = self.q_proj(x)
            k, v = (t.contiguous() for t in self.kv_proj(memory).chunk(2, dim=-1))
        else:
            q, k, v = (t.contiguous() for t in self.qkv_proj(x).chunk(3, dim=-1))

        rope = not self.cross_attn
        q = self._head_split(q, rope=rope)
        k = self._head_split(k, rope=rope)
        v = self._head_split(v, rope=False)

        if is_causal and q.is_nested:
            out = _causal_jagged_attention(q, k, v)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        return self.out_proj(out.transpose(1, 2).flatten(-2))

    def _head_split(self, t: torch.Tensor, rope: bool) -> torch.Tensor:
        """``(B, *, d_model)`` -> ``(B, n_heads, *, d_head)``, applying RoPE if requested.

        For jagged inputs, RoPE runs on the flat values buffer and the nested wrapper is
        built in a single ``from_jagged`` call — applying RoPE to a nested view and
        rewrapping breaks FlexAttention's symbolic shape inference.
        """
        if t.is_nested:
            vals = t.values().view(-1, self.n_heads, self.d_head)
            if rope:
                vals = self.rope.apply_jagged(vals, t.offsets())
            return torch.nested.nested_tensor_from_jagged(vals, t.offsets()).transpose(1, 2)
        out = t.unflatten(-1, (self.n_heads, self.d_head))
        if rope:
            out = self.rope(out)
        return out.transpose(1, 2)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int = 8192):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, max_seq_len)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int = 8192):
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, max_seq_len)
        self.cross_q_norm = nn.LayerNorm(d_model)
        self.cross_kv_norm = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, max_seq_len, cross_attn=True)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.self_attn_norm(x), is_causal=True)
        x = x + self.cross_attn(self.cross_q_norm(x), self.cross_kv_norm(memory))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Seq2SeqTransformer(nn.Module):
    """Encoder-decoder Transformer over jagged NestedTensors of token IDs.

    Inputs ``src`` and ``tgt`` are jagged ``(B, j1)`` LongTensors. The output is a
    jagged ``(B, j1, n_tokens)`` tensor of next-token logits aligned with ``tgt``.
    """

    def __init__(
        self,
        n_tokens: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int | None = None,
        max_seq_len: int = 8192,
        compile_layers: bool = False,
    ):
        super().__init__()
        d_ff = d_ff if d_ff is not None else int(8 * d_model / 3)
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.embed = nn.Embedding(n_tokens, d_model)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, max_seq_len) for _ in range(n_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, max_seq_len) for _ in range(n_layers)]
        )
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)
        self.project = nn.Linear(d_model, n_tokens, bias=False)

        if compile_layers:
            # Decoder layers excluded: flex_attention + nested + AOT autograd is broken in PyTorch 2.9.
            for i in range(n_layers):
                self.encoder_layers[i] = torch.compile(self.encoder_layers[i])

    def _embed(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.is_nested:
            return torch.nested.nested_tensor_from_jagged(self.embed(tokens.values()), tokens.offsets())
        return self.embed(tokens)

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        h = self._embed(src)
        for layer in self.encoder_layers:
            h = layer(h)
        return self.encoder_norm(h)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        h = self._embed(tgt)
        for layer in self.decoder_layers:
            h = layer(h, memory)
        return self.decoder_norm(h)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        return self.project(self.decode(tgt, self.encode(src)))


if __name__ == "__main__":
    torch.manual_seed(0)
    model = Seq2SeqTransformer(n_tokens=256, d_model=128, n_heads=4, n_layers=2)

    src = from_token_ids([torch.randint(0, 256, (n,)) for n in (7, 12, 3)])
    tgt = from_token_ids([torch.randint(0, 256, (n,)) for n in (5, 9, 4)])
    logits = model(src, tgt)
    print(f"src lengths: {src.offsets().diff().tolist()}")
    print(f"tgt lengths: {tgt.offsets().diff().tolist()}")
    print(f"logits values: {logits.values().shape}  (sum_tgt_len, n_tokens)")

    # encoder-only backward to exercise the SDPA jagged path on CPU
    model.zero_grad()
    model.encode(src).values().sum().backward()
    grads = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"encoder backward grad-norm-sum: {grads:.4f}")
