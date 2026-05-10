"""CLI entry point: ``python -m src.analysis path/to/config.yaml``.

No subcommands. No multiprocessing in v1; transformer eval is gated by the
per-method disk cache so re-runs are cheap.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .metrics import build_analysis
from .cache import Cache
from .config import load_config
from .methods import build_method
from .task import TaskBundle

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the MLMP analysis pipeline")
    p.add_argument("config", type=Path, help="Path to a YAML config file")
    p.add_argument("--log-level", default="INFO", help="Python logging level (default INFO)")
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s [%(name)s] %(message)s")

    cfg = load_config(args.config)
    bundle = TaskBundle.load(cfg.rule_data_root)

    methods = []
    for spec in cfg.methods:
        spec = dict(spec)
        # Default device for transformer methods.
        if spec.get("kind") == "transformer" and "device" not in spec:
            spec["device"] = cfg.device
        methods.append(build_method(spec, rule_data_root=str(cfg.rule_data_root)))
    by_name = {m.name: m for m in methods}

    outdir_root = cfg.output_dir / cfg.run_name
    cache = Cache(outdir_root / "cache")

    for acfg in cfg.analyses:
        analysis = build_analysis({"kind": acfg.kind, **acfg.extra})
        if acfg.methods is not None:
            sel = [by_name[n] for n in acfg.methods]
        else:
            sel = methods
        logger.info("Running analysis %s with %d methods", acfg.kind, len(sel))
        result = analysis.run(sel, bundle, cache)
        kind_dir = outdir_root / acfg.kind
        result.save(kind_dir)
        result.plot(kind_dir)
        logger.info("  → %s", kind_dir)

    cache.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
