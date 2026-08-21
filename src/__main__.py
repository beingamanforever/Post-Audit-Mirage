from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .dataset import build_dataset, canonical_json_line, load_templates, publish_rows
from .environment_validation import (
    BATCH_TRIAGE_EVALUATOR,
    NODE_EVALUATOR,
    build_planning,
    validate_environments,
)
from .experiments import run_experiments
from .lifecycle import run_methods
from .surface_generation import (
    DEFAULT_ENDPOINT,
    SurfaceGenerationError,
    generate_surface_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


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

    planning = commands.add_parser("build-planning")
    planning.add_argument(
        "--templates",
        type=Path,
        default=DATA_DIR / "planning_templates.json",
    )
    planning.add_argument("--output-dir", type=Path, default=DATA_DIR)
    planning.add_argument("--seed", type=int, default=20260820)
    planning.add_argument("--instances-per-template", type=int, default=2)
    planning.add_argument("--node-evaluator", type=Path, default=NODE_EVALUATOR)
    planning.add_argument(
        "--batch-triage-evaluator",
        type=Path,
        default=BATCH_TRIAGE_EVALUATOR,
    )

    validation = commands.add_parser("validate-environments")
    validation.add_argument(
        "--templates",
        type=Path,
        default=DATA_DIR / "planning_templates.json",
    )
    validation.add_argument("--seed", type=int, default=20260820)
    validation.add_argument("--instances-per-template", type=int, default=2)
    validation.add_argument("--node-evaluator", type=Path, default=NODE_EVALUATOR)
    validation.add_argument(
        "--batch-triage-evaluator",
        type=Path,
        default=BATCH_TRIAGE_EVALUATOR,
    )

    methods = commands.add_parser("run-methods")
    methods.add_argument("--data-dir", type=Path, default=DATA_DIR)
    methods.add_argument("--world", choices=("safe", "harmful"), default="harmful")
    methods.add_argument("--alpha", type=float, default=0.05)
    methods.add_argument("--seed", type=int, default=0)
    methods.add_argument("--output", type=Path)

    experiments = commands.add_parser("run-experiments")
    experiments.add_argument("--data-dir", type=Path, default=DATA_DIR)
    experiments.add_argument("--output-dir", type=Path, default=DATA_DIR)
    experiments.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    experiments.add_argument("--replications", type=int, default=500)
    experiments.add_argument("--seed", type=int, default=20260821)
    experiments.add_argument("--alpha", type=float, default=0.05)
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
        if arguments.command == "build-dataset":
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
        if arguments.command == "build-planning":
            proposer_count, audit_count, truth_count = build_planning(
                arguments.templates,
                arguments.output_dir,
                seed=arguments.seed,
                instances_per_template=arguments.instances_per_template,
                node_path=arguments.node_evaluator,
                batch_triage_path=arguments.batch_triage_evaluator,
            )
            print(
                f"wrote {proposer_count} proposer rows, {audit_count} audit rows, "
                f"and {truth_count} truth rows to {arguments.output_dir}"
            )
            return 0
        if arguments.command == "run-methods":
            rows = run_methods(
                arguments.data_dir,
                world=arguments.world,
                alpha=arguments.alpha,
                seed=arguments.seed,
            )
            if arguments.output is None:
                sys.stdout.writelines(canonical_json_line(row) for row in rows)
            else:
                publish_rows(
                    arguments.output.parent,
                    {arguments.output.name: rows},
                )
                print(f"wrote {len(rows)} decisions to {arguments.output}")
            return 0
        if arguments.command == "run-experiments":
            summary = run_experiments(
                arguments.output_dir,
                artifacts_dir=arguments.artifacts_dir,
                data_dir=arguments.data_dir,
                replications=arguments.replications,
                seed=arguments.seed,
                alpha=arguments.alpha,
            )
            statuses = ", ".join(
                f"{name}={result['status']}"
                for name, result in summary["experiments"].items()
            )
            print(
                f"wrote Phase 4 data to {arguments.output_dir} and plots to "
                f"{arguments.artifacts_dir}: {statuses}"
            )
            return 0
        instances, _ = validate_environments(
            arguments.templates,
            seed=arguments.seed,
            instances_per_template=arguments.instances_per_template,
            node_path=arguments.node_evaluator,
            batch_triage_path=arguments.batch_triage_evaluator,
        )
        print(
            f"validated all three exact evaluators on {len(instances)} "
            "planning instances"
        )
        return 0
    except (OSError, ValueError, SurfaceGenerationError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
