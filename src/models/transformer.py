import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules.position_embeddings import RotaryPositionalEmbeddings



class AttentionBlock(nn.Module):
    def __init__(self, d_embed: int, d_model: int, n_heads: int, dropout: float = 0.0, max_seq_len: int = 8192):
        super().__init__()
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_heads = n_heads

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_head = d_model // n_heads

        self.q = nn.Linear(d_embed, d_model)
        self.k = nn.Linear(d_embed, d_model)
        self.v = nn.Linear(d_embed, d_model)
        self.o = nn.Linear(d_model, d_embed)

        self.pos_emb = RotaryPositionalEmbeddings(self.d_head, max_seq_len=max_seq_len)

        self.norm = nn.LayerNorm(d_embed)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x: torch.Tensor):
        B, L, _ = x.shape
        normed_x = self.norm(x)                                         # (B, L, d_embed)
        q = self.q(normed_x)                                            # (B, L, d_model)
        k = self.k(normed_x)                                            # (B, L, d_model)
        v = self.v(normed_x)                                            # (B, L, d_model)

        q = q.view(B, L, self.n_heads, self.d_head)                     # (B, L, n_heads, d_head)
        k = k.view(B, L, self.n_heads, self.d_head)                     # (B, L, n_heads, d_head)
        v = v.view(B, L, self.n_heads, self.d_head)                     # (B, L, n_heads, d_head)

        # RoPE requires (B, L, n_heads, d_head)
        q = self.pos_emb(q, input_pos=None)                             # (B, L, n_heads, d_head)
        k = self.pos_emb(k, input_pos=None)                             # (B, L, n_heads, d_head)

        # Reshape to (B, n_heads, L, d_head) for attention computation
        # .contiguous() needed for CUDA strided batched matmul operations
        q = q.transpose(1, 2).contiguous()                              # (B, n_heads, L, d_head)
        k = k.transpose(1, 2).contiguous()                              # (B, n_heads, L, d_head)
        v = v.transpose(1, 2).contiguous()                              # (B, n_heads, L, d_head)

        # Use Flash Attention via scaled_dot_product_attention (O(L) memory instead of O(L²))
        # is_causal=True applies causal mask efficiently without materializing the full L×L matrix
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # (B, n_heads, L, d_head)
        stream = out.transpose(1, 2).contiguous().view(B, L, self.d_model) # (B, L, d_model); contig needed for view
        return self.dropout(self.o(stream))                             # (B, L, d_embed)
    

class FeedForwardBlock(nn.Module):
    def __init__(self, d_embed: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.d_embed = d_embed
        self.d_ff = d_ff

        self.fc1 = nn.Linear(d_embed, d_ff)
        self.fc2 = nn.Linear(d_ff, d_embed)
        self.norm = nn.LayerNorm(d_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        normed_x = self.norm(x) # (B, L, d_embed)
        x1 = F.relu(self.fc1(normed_x)) # (B, L, d_ff)
        return self.dropout(self.fc2(x1)) # (B, L, d_embed)


class DecoderBlock(nn.Module):
    def __init__(self, d_embed: int, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0, max_seq_len: int = 8192):
        super().__init__()
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout

        self.attention = AttentionBlock(d_embed, d_model, n_heads, dropout, max_seq_len=max_seq_len)
        self.feed_forward = FeedForwardBlock(d_embed, d_ff, dropout)
    
    def forward(self, x: torch.Tensor):
        # x has shape (B, L, d_embed)
        # Causal masking is handled by is_causal=True in scaled_dot_product_attention
        x = x + self.attention(x)
        x = x + self.feed_forward(x)
        return x


class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        n_tokens: int,
        d_embed: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.0,
        max_seq_len: int = 8192,
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = 4 * d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.dropout = dropout

        self.embed = nn.Embedding(self.n_tokens, self.d_embed)
        self.layers = nn.ModuleList([
            DecoderBlock(self.d_embed, self.d_model, self.n_heads, self.d_ff, self.dropout, max_seq_len=self.max_seq_len)
            for _ in range(self.n_layers)
        ])
        self.norm = nn.LayerNorm(self.d_embed)
        self.project = nn.Linear(self.d_embed, self.n_tokens, bias=False)
    
    def forward(self, x: torch.Tensor, return_all_logits: bool = False):
        # x has shape (B, L)
        x = self.embed(x) # (B, L, d_embed)
        for layer in self.layers:
            x = layer(x) # (B, L, d_embed)
        x = self.norm(x)
        if return_all_logits:
            return self.project(x) # (B, L, n_tokens)
        last = x[:, -1, :] # (B, d_embed)
        return self.project(last) # (B, n_tokens)
    
    def generate(self, x: torch.Tensor, max_tokens: int):
        out = torch.empty(x.size(0), 0, device=x.device) # empty tensor (B, 0)
        for _ in range(max_tokens):
            logits = self(x) # (B, n_tokens)
            next_token = torch.argmax(logits, dim=-1).unsqueeze(-1) # (B, 1)
            out = torch.cat([out, next_token], dim=1) # (B, L+1)
            x = torch.cat([x, next_token], dim=1) # (B, L+1)
        return out


if __name__ == '__main__':
    model = DecoderOnlyTransformer(
        n_tokens=100,
        d_embed=32,
        d_model=32,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
    )
    print(model)

    B, L = 2, 10
    x = torch.randint(0, model.n_tokens, (B, L))
    print(f"B: {B}, L: {L}")
    print(f"x: {x}")
    print(f"model(x): {model(x)}")
    print(f"model.generate(x): {model.generate(x, max_tokens=10)}")
