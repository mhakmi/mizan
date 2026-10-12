import argparse
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import assert_never

from mizan.files import load_project, write_baseline, write_report
from mizan.gate import baseline_from, evaluate
from mizan.providers import connect, label
from mizan.report import render_text
from mizan.runner import run_cases
from mizan.types import (
    ConfigError,
    GateFailed,
    Incomplete,
    MizanError,
    Passed,
    Report,
    RunSummary,
    Verdict,
)

EXIT_PASSED = 0
EXIT_GATE_FAILED = 1
EXIT_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mizan",
        description=(
            "Run a dataset against a prompt, score the answers, and exit non-zero "
            "when quality drops or the prompt goes over its token budget."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run",
        help="run the dataset and check the gate",
        description="Exit 0 when the run passes, 1 when the gate fails, 2 when it cannot finish.",
    )
    run.add_argument(
        "--config",
        type=Path,
        default=Path("mizan.toml"),
        metavar="PATH",
        help="the config file (default: mizan.toml)",
    )
    run.add_argument(
        "--system-prompt",
        type=Path,
        metavar="PATH",
        help="use this file as the system prompt for this run, to try a draft",
    )
    run.add_argument(
        "--update-baseline",
        action="store_true",
        help="save this run as the new baseline, if it passes",
    )
    return parser


def run_command(config_path: Path, system_prompt: Path | None, update_baseline: bool) -> int:
    if system_prompt is not None and update_baseline:
        raise ConfigError(
            "--system-prompt cannot be used with --update-baseline. "
            "The baseline records the prompt named in the config."
        )

    project = load_project(config_path, system_prompt)
    config = project.config
    complete = connect(config.model, project.config_dir)
    ask_judge = None
    judge_label = None
    if config.judge is not None:
        ask_judge = connect(config.judge, project.config_dir)
        judge_label = label(config.judge)

    results = run_cases(project.cases, complete, ask_judge, config.concurrency)
    summary = RunSummary(model=label(config.model), judge=judge_label, results=results)

    baseline = None if update_baseline else project.baseline
    verdict = evaluate(summary, baseline, config.gate)

    if system_prompt is not None:
        print(f"System prompt for this run: {system_prompt}\n")
    print(render_text(config_path, summary, verdict, config.gate))
    write_report(project.report_path, Report(verdict=verdict, run=summary))
    print(f"\nReport written to {project.report_path}")

    if update_baseline:
        if isinstance(verdict, Passed):
            write_baseline(project.baseline_path, baseline_from(summary))
            print(f"Baseline written to {project.baseline_path}")
        else:
            print("The baseline was not written, because the run did not pass.")
    elif project.baseline is None:
        print(f"No baseline at {project.baseline_path}.")
        print("Create one by running the same command with --update-baseline.")

    return exit_code(verdict)


def exit_code(verdict: Verdict) -> int:
    match verdict:
        case Passed():
            return EXIT_PASSED
        case GateFailed():
            return EXIT_GATE_FAILED
        case Incomplete():
            return EXIT_ERROR
        case _:
            assert_never(verdict)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        return run_command(arguments.config, arguments.system_prompt, arguments.update_baseline)
    except MizanError as error:
        print(f"mizan: {error}", file=sys.stderr)
        return EXIT_ERROR
    except Exception:  # noqa: BLE001
        # Python exits 1 on an uncaught exception, and 1 must only mean a failed gate.
        traceback.print_exc()
        return EXIT_ERROR
