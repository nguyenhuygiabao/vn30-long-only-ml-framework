from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PipelineStep:
    name: str
    command: list[str]
    description: str


PIPELINE_STEPS = [
    PipelineStep(
        name="data_quality",
        command=["-m", "src.data_quality"],
        description="Validate raw OHLCV data and refresh the data quality report.",
    ),
    PipelineStep(
        name="features",
        command=["-m", "src.feature_pipeline"],
        description="Build the combined feature dataset.",
    ),
    PipelineStep(
        name="labels",
        command=["-m", "src.labels"],
        description="Build forward relative-return labels.",
    ),
    PipelineStep(
        name="walk_forward_check",
        command=["-m", "src.walk_forward_split"],
        description="Run walk-forward split and purging checks.",
    ),
    PipelineStep(
        name="linear_models",
        command=["-m", "src.linear_models"],
        description="Run linear model experiments.",
    ),
    PipelineStep(
        name="tree_models",
        command=["-m", "src.tree_models"],
        description="Run tree-based model experiments.",
    ),
    PipelineStep(
        name="classification_models",
        command=["-m", "src.classification_models"],
        description="Run classification model experiments.",
    ),
    PipelineStep(
        name="baseline_portfolios",
        command=["-m", "src.baseline_portfolios"],
        description="Run baseline portfolio comparisons.",
    ),
    PipelineStep(
        name="risk_model",
        command=["-m", "src.risk_model"],
        description="Build the past-only rolling risk model.",
    ),
    PipelineStep(
        name="optimizer",
        command=["-m", "src.optimizer"],
        description="Build constrained long-only optimized weights.",
    ),
    PipelineStep(
        name="backtester",
        command=["-m", "src.backtester"],
        description="Run transaction-cost and price-limit-aware backtests.",
    ),
    PipelineStep(
        name="ablation_tests",
        command=["-m", "src.ablation_tests"],
        description="Run feature-ablation tests.",
    ),
    PipelineStep(
        name="horizon_tests",
        command=["-m", "src.horizon_tests"],
        description="Run forecast-horizon tests.",
    ),
    PipelineStep(
        name="model_report",
        command=["-m", "src.model_report"],
        description="Refresh the model report.",
    ),
    PipelineStep(
        name="visualize_results",
        command=["-m", "src.visualize_results"],
        description="Regenerate report figures.",
    ),
    PipelineStep(
        name="report_summary",
        command=[str(ROOT / "scripts" / "report_summary.py")],
        description="Print quick report, table, and figure summary.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the VN30 long-only ML framework pipeline.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print pipeline steps without executing them.",
    )
    parser.add_argument(
        "--start-at",
        choices=[step.name for step in PIPELINE_STEPS],
        help="Start from a specific pipeline step.",
    )
    parser.add_argument(
        "--stop-after",
        choices=[step.name for step in PIPELINE_STEPS],
        help="Stop after a specific pipeline step.",
    )
    parser.add_argument(
        "--clean-first",
        action="store_true",
        help="Run the generated-output cleaner before the pipeline.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Run only the report summary step.",
    )
    return parser.parse_args()


def selected_steps(start_at: str | None, stop_after: str | None) -> list[PipelineStep]:
    steps = PIPELINE_STEPS

    if start_at is not None:
        start_index = next(index for index, step in enumerate(steps) if step.name == start_at)
        steps = steps[start_index:]

    if stop_after is not None:
        stop_index = next(index for index, step in enumerate(steps) if step.name == stop_after)
        steps = steps[: stop_index + 1]

    return steps


def run_step(step_number: int, total_steps: int, step: PipelineStep) -> None:
    print()
    print("=" * 80)
    print(f"[{step_number}/{total_steps}] {step.name}")
    print(step.description)
    print(f"Command: {sys.executable} {' '.join(step.command)}")
    print("=" * 80)

    start_time = time.perf_counter()

    completed = subprocess.run(
        [sys.executable, *step.command],
        cwd=ROOT,
        check=False,
    )

    elapsed = time.perf_counter() - start_time

    if completed.returncode != 0:
        raise RuntimeError(
            f"Pipeline stopped at step {step_number}/{total_steps}: {step.name}. "
            f"Exit code: {completed.returncode}"
        )

    print(f"Finished: {step.name} ({elapsed:.1f} seconds)")


def run_cleaner(dry_run: bool) -> None:
    cleaner_path = ROOT / "scripts" / "clean_generated_outputs.py"

    if not cleaner_path.exists():
        raise FileNotFoundError(f"Missing cleaner script: {cleaner_path}")

    command = [sys.executable, str(cleaner_path)]

    if not dry_run:
        command.append("--confirm")

    print()
    print("=" * 80)
    print("Generated output cleaner")
    print("=" * 80)

    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            f"Generated output cleaner failed. Exit code: {completed.returncode}"
        )


def main() -> None:
    args = parse_args()
    steps = selected_steps(args.start_at, args.stop_after)
    if args.summary_only and (args.start_at is not None or args.stop_after is not None):
        raise SystemExit("--summary-only cannot be combined with --start-at or --stop-after.")

    if args.summary_only and args.clean_first and not args.dry_run:
        raise SystemExit("--clean-first --summary-only is unsafe without --dry-run.")

    if args.summary_only:
        steps = [step for step in PIPELINE_STEPS if step.name == "report_summary"]

    print()
    print("VN30 LONG-ONLY ML PIPELINE")
    print("=" * 80)
    print(f"Project root: {ROOT}")
    print(f"Python executable: {sys.executable}")
    print()
    print("This runner refreshes local outputs using existing raw data.")
    print("It does not download new data.")
    print("It deletes generated parquet outputs only when --clean-first is used.")
    print()

    if args.clean_first:
        run_cleaner(dry_run=args.dry_run)

    if args.dry_run:
        print("DRY RUN ONLY. No commands will be executed.")
        print()
        for index, step in enumerate(steps, start=1):
            print(f"{index}. {step.name}")
            print(f"   {step.description}")
            print(f"   Command: {sys.executable} {' '.join(step.command)}")
        print()
        return

    total_steps = len(steps)

    for step_number, step in enumerate(steps, start=1):
        run_step(step_number, total_steps, step)

    print()
    print("=" * 80)
    print("FULL PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print()
    print("Useful next commands:")
    print("  py .\\scripts\\report_summary.py")
    print("  code .\\reports\\final_results.md")
    print("  code .\\reports\\report_index.md")
    print("  explorer .\\reports\\figures")
    print()


if __name__ == "__main__":
    main()