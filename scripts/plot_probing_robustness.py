"""Post-hoc robustness plot for the probing sweep.

Reads ``auroc.parquet`` produced by the ``probing_robustness`` run and emits
``robustness_<primitive>.pdf`` and a combined ``robustness_grid.pdf`` showing
AUROC vs encoder/decoder layer for each checkpoint.

Method names produced by ``build_probing_robustness_config.py`` look like
``tx_<ckpt_id>_<src>_<layer_tag>`` where ``<src>`` ∈ {enc, dec} and
``<layer_tag>`` ∈ {L0..LN, post}. We parse those back to (ckpt, source, layer)
to draw the lines. Non-matching method names (surface_features, mpl_best,
custom one-offs) are ignored.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

# Match e.g. ``tx_enum_iw_enc_L2`` / ``tx_enrl_es_dec_post``.
_PATTERN = re.compile(r"^tx_(?P<ckpt>[a-z0-9_]+?)_(?P<src>enc|dec)_(?P<tag>L\d+|post)$")


def _parse_method(name: str) -> tuple[str, str, int] | None:
    m = _PATTERN.match(name)
    if not m:
        return None
    tag = m.group("tag")
    layer = -1 if tag == "post" else int(tag[1:])
    return m.group("ckpt"), m.group("src"), layer


def _layer_order(layer: int, n_layers: int) -> int:
    # Order for x-axis: 0 (embed) → 1..N (blocks) → post-norm at the end.
    return n_layers + 1 if layer == -1 else layer


def _layer_label(layer: int) -> str:
    return "post-norm" if layer == -1 else f"L{layer}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="Path to the probing run dir, e.g. outputs/analysis/probe_robustness_v1/probing")
    args = ap.parse_args()

    auroc = pd.read_parquet(args.run_dir / "auroc.parquet")

    parsed_rows: list[dict] = []
    for _, r in auroc.iterrows():
        p = _parse_method(r["method"])
        if p is None:
            continue
        ckpt, src, layer = p
        parsed_rows.append({
            "ckpt": ckpt, "src": src, "layer": layer,
            "primitive": r["primitive"], "auroc": float(r["auroc"]),
        })
    df = pd.DataFrame(parsed_rows)
    if df.empty:
        raise SystemExit("No sweep methods (tx_<ckpt>_<src>_<tag>) in auroc.parquet")

    per_ml = df.groupby(["ckpt", "src", "layer", "primitive"])["auroc"].agg(["mean", "std", "count"]).reset_index()
    n_layers = max(l for l in per_ml["layer"] if l != -1)
    per_ml["xpos"] = per_ml["layer"].map(lambda l: _layer_order(l, n_layers))

    primitives = sorted(per_ml["primitive"].unique())
    ckpts = sorted(per_ml["ckpt"].unique())
    x_layers = sorted({(l, _layer_order(l, n_layers)) for l in per_ml["layer"]}, key=lambda t: t[1])
    x_pos = [p for _, p in x_layers]
    x_labels = [_layer_label(l) for l, _ in x_layers]

    src_colour = {"enc": "#1f77b4", "dec": "#d62728"}
    src_label = {"enc": "encoder", "dec": "decoder"}

    out_dir = args.run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Per-primitive figures: one panel per checkpoint.
    for prim in primitives:
        sub = per_ml[per_ml["primitive"] == prim]
        if sub.empty:
            continue
        fig, axes = plt.subplots(1, len(ckpts), figsize=(3.4 * len(ckpts), 3.2), sharey=True)
        if len(ckpts) == 1:
            axes = [axes]
        for ax, ck in zip(axes, ckpts):
            for src in ("enc", "dec"):
                s = sub[(sub["ckpt"] == ck) & (sub["src"] == src)].sort_values("xpos")
                if s.empty:
                    continue
                yerr = (s["std"].fillna(0).values / np.sqrt(np.maximum(s["count"].fillna(1).values, 1)))
                ax.errorbar(s["xpos"], s["mean"], yerr=yerr, marker="o",
                            color=src_colour[src], label=src_label[src], capsize=2)
            ax.axhline(0.5, color="gray", lw=0.6, alpha=0.6)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, rotation=30, ha="right")
            ax.set_title(ck, fontsize=10)
            ax.set_ylim(0.4, 1.02)
        axes[0].set_ylabel(f"AUROC ({prim})")
        axes[-1].legend(fontsize=8, frameon=False, loc="lower right")
        fig.tight_layout()
        fig.savefig(out_dir / f"robustness_{prim}.pdf")
        plt.close(fig)

    # 2) Grid: rows = primitives, cols = checkpoints. Compact overview.
    fig, axes = plt.subplots(
        len(primitives), len(ckpts),
        figsize=(2.8 * len(ckpts), 1.9 * len(primitives)),
        sharey=True, sharex=True, squeeze=False,
    )
    for i, prim in enumerate(primitives):
        for j, ck in enumerate(ckpts):
            ax = axes[i][j]
            sub = per_ml[(per_ml["primitive"] == prim) & (per_ml["ckpt"] == ck)]
            for src in ("enc", "dec"):
                s = sub[sub["src"] == src].sort_values("xpos")
                if s.empty:
                    continue
                ax.plot(s["xpos"], s["mean"], marker="o", color=src_colour[src],
                        label=src_label[src] if (i == 0 and j == 0) else None)
            ax.axhline(0.5, color="gray", lw=0.5, alpha=0.5)
            ax.set_ylim(0.4, 1.02)
            if i == 0:
                ax.set_title(ck, fontsize=9)
            if j == 0:
                ax.set_ylabel(prim, fontsize=9)
            ax.set_xticks(x_pos)
            if i == len(primitives) - 1:
                ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8)
            else:
                ax.set_xticklabels([])
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncols=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(out_dir / "robustness_grid.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote robustness_*.pdf and robustness_grid.pdf to {out_dir}")


if __name__ == "__main__":
    main()
