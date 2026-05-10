"""Method registry: 10-line ``kind`` → class lookup, plus ``build_method``."""
from __future__ import annotations

from typing import Any

from .base import Method, Prediction  # noqa: F401  (re-exported)
from .codex import CodexMethod
from .csv_method import EMPTY, NO_RESPONSE, _parse_response  # noqa: F401
from .enumeration import EnumerationMethod
from .fleet import FleetBestMethod, FleetMethod
from .human import HumanMethod
from .metagol import MetagolMethod
from .mpl import MPLBestMethod, MPLMethod
from .robustfill import RobustFillMethod
from .transformer import TransformerMethod

KINDS: dict[str, type[Method]] = {
    "transformer": TransformerMethod,
    "human": HumanMethod,
    "mpl": MPLMethod,
    "mpl_best": MPLBestMethod,
    "fleet": FleetMethod,
    "fleet_best": FleetBestMethod,
    "codex": CodexMethod,
    "enumeration": EnumerationMethod,
    "metagol": MetagolMethod,
    "robustfill": RobustFillMethod,
}


def build_method(spec: dict[str, Any], rule_data_root: str | None = None) -> Method:
    """Instantiate a method from a YAML-derived dict.

    Required keys: ``kind``, ``name``. Other keys are forwarded as kwargs.
    ``rule_data_root`` is injected as ``root`` for CSV-backed methods if absent.
    """
    spec = dict(spec)
    kind = spec.pop("kind")
    if kind not in KINDS:
        raise ValueError(f"Unknown method kind {kind!r}; known: {sorted(KINDS)}")
    klass = KINDS[kind]
    if klass is not TransformerMethod and rule_data_root is not None and "root" not in spec:
        spec["root"] = rule_data_root
    return klass(**spec)
