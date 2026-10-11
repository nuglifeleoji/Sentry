"""Offline validation of the declared runtime settings, without model requests."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .config import SentryConfig


def describe_config(path: str | Path) -> dict:
    source = Path(path)
    source_bytes = source.read_bytes()
    config = SentryConfig.from_yaml_text(source_bytes.decode("utf-8"))
    return {
        "format": "sentry-config-manifest-v2",
        "validation": "Sentry runtime configuration",
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "declared_model_identity": config.model.model,
        "api_configured": config.api is not None,
        "config": asdict(config),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Full Sentry YAML configuration")
    parser.add_argument("--manifest", type=Path, help="Write settings and the configuration hash as JSON")
    args = parser.parse_args()
    try:
        report = describe_config(args.config)
        output = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.manifest:
            if args.manifest.exists():
                raise ValueError("Manifest path already exists; choose a new path")
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation also prevents replacing a source config or symlink.
            with args.manifest.open("x", encoding="utf-8") as handle:
                handle.write(output)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Configuration validation failed: {exc}\n")
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
