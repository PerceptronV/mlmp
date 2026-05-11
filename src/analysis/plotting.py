"""Shared matplotlib plumbing for the analysis subpackage.

Sets a consistent rcParams + a small palette and exposes ``save_fig``. Each
``AnalysisResult.plot`` writes its own figures using these helpers.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib  # type: ignore[import-untyped]
import matplotlib.pyplot as plt  # type: ignore[import-untyped]


_RCPARAMS = {
    "figure.figsize": (6.0, 4.0),
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "font.family": "sans-serif",
    "lines.linewidth": 1.6,
}


def apply_rc() -> None:
    matplotlib.rcParams.update(_RCPARAMS)


# Stable palette: keyed on method name so the same method is the same colour
# across figures. Falls back to a category10 cycle for unknown names.
_PALETTE_BASE = {
    "humans": "#222222",
    "mpl": "#d62728",
    "fleet": "#17becf",
    "tx_enum_in_weight": "#2ca02c",
    "tx_enum_easy_shuf": "#ff7f0e",
    "tx_enrl_in_weight": "#9467bd",
    "tx_enrl_easy_shuf": "#8c564b",
    "codex": "#1f77b4",
    "enumeration": "#7f7f7f",
    "metagol": "#e377c2",
    "robustfill": "#bcbd22",
    "surface_features": "#e377c2",
}


def colour_for(name: str) -> str:
    if name in _PALETTE_BASE:
        return _PALETTE_BASE[name]
    cycle = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    return cycle[hash(name) % len(cycle)]


# Display-label registry. Cache/data files are keyed on ``method.name`` (stable
# identifier), but plots should show ``method.label`` (human-facing). The CLI
# populates this once after building methods; ``label_for`` falls back to the
# name if the registry is empty or the key isn't registered.
_LABELS: dict[str, str] = {}


def register_labels(methods) -> None:
    """Register ``name -> label`` for plotting. Idempotent."""
    for m in methods:
        nm = getattr(m, "name", None)
        if not nm:
            continue
        lab = getattr(m, "label", "") or nm
        _LABELS[nm] = lab


def label_for(name: str) -> str:
    """Resolve a method ``name`` to its display label. Falls back to ``name``."""
    return _LABELS.get(name, name)


def labels_for(names) -> list[str]:
    return [label_for(n) for n in names]


def save_fig(fig, outdir: Path, name: str) -> Path:
    """Save ``fig`` as <name> (PDF, archival) and a sibling .png (for quick
    visual inspection / for tools that don't render vector PDFs). Returns the
    PDF path.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    # Automatically save both PDF and PNG for analysis and archival purposes.
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    # User-passed script defaults
    fig.savefig(path)
    plt.close(fig)
    return path
