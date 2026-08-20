from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .dataset import build_dataset, load_templates
from .surface_generation import (
    DEFAULT_ENDPOINT,
    SurfaceGenerationError,
    generate_surface_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src")
    commands = parser.add_subparsers(dest="command", required=True)

    surfaces = commands.add_parser("generate-surfaces")
    surfaces.add_argument("--templates", type=Path, default=DATA_DIR / "templates.json")
    surfaces.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR / "surface_variants.jsonl",
    )
    surfaces.add_argument("--seed", type=int, default=20260820)
    surfaces.add_argument("--endpoint", default=DEFAULT_ENDPOINT)

    dataset = commands.add_parser("build-dataset")
    dataset.add_argument("--templates", type=Path, default=DATA_DIR / "templates.json")
    dataset.add_argument(
        "--surfaces",
        type=Path,
        default=DATA_DIR / "surface_variants.jsonl",
    )
    dataset.add_argument("--output-dir", type=Path, default=DATA_DIR)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        if arguments.command == "generate-surfaces":
            count = generate_surface_file(
                load_templates(arguments.templates),
                arguments.output,
                seed=arguments.seed,
                endpoint=arguments.endpoint,
            )
            print(f"wrote {count} surface variants to {arguments.output}")
            return 0
        audit_count, truth_count = build_dataset(
            arguments.templates,
            arguments.surfaces,
            arguments.output_dir,
        )
        print(
            f"wrote {audit_count} audit rows and {truth_count} truth rows "
            f"to {arguments.output_dir}"
        )
        return 0
    except (OSError, ValueError, SurfaceGenerationError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
