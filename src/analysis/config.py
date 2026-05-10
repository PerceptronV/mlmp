"""YAML config loader.

PyYAML + ``@dataclass`` tree, no pydantic. Unknown keys fail loudly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


@dataclass
class AnalysisConfig:
    kind: str
    methods: list[str] | None = None  # subset of method names; None = all
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    run_name: str
    output_dir: Path
    rule_data_root: Path
    device: str = "cuda"
    methods: list[dict[str, Any]] = field(default_factory=list)
    analyses: list[AnalysisConfig] = field(default_factory=list)


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    required = {"run_name", "output_dir", "rule_data_root", "methods", "analyses"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"Config {path} missing keys: {sorted(missing)}")

    methods = list(raw["methods"])
    analyses_raw = raw["analyses"]
    analyses: list[AnalysisConfig] = []
    for a in analyses_raw:
        a = dict(a)
        kind = a.pop("kind")
        method_subset = a.pop("methods", None)
        analyses.append(AnalysisConfig(kind=kind, methods=method_subset, extra=a))

    return Config(
        run_name=raw["run_name"],
        output_dir=Path(raw["output_dir"]),
        rule_data_root=Path(raw["rule_data_root"]),
        device=raw.get("device", "cuda"),
        methods=methods,
        analyses=analyses,
    )
