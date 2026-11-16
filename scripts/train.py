"""Generate a fresh, reproducible PixelProof training bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pixelproof.training import write_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new output directory for the model bundle")
    arguments = parser.parse_args()
    bundle = write_bundle(arguments.output)
    summary = {
        "output": str(arguments.output),
        "model_sha256": bundle["evaluation"]["model_sha256"],
        "train_count": bundle["evaluation"]["train_count"],
        "test_count": bundle["evaluation"]["test_count"],
        "metrics": bundle["evaluation"]["metrics"],
    }
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
