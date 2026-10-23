import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from mizan.cases import build_cases
from mizan.types import Baseline, Case, Config, ConfigError, LlmRubric, Report, explain


@dataclass(frozen=True)
class Project:
    config_path: Path
    config_dir: Path
    config: Config
    cases: tuple[Case, ...]
    baseline: Baseline | None
    baseline_path: Path
    report_path: Path


def load_project(config_path: Path, system_prompt_override: Path | None = None) -> Project:
    """Read the config, the prompt files, the dataset and the baseline.

    system_prompt_override is used in place of the config's system prompt. It
    is relative to the current directory, not to the config.

    Raises ConfigError that names the file for anything that is wrong.
    """
    config_dir = config_path.parent
    try:
        config = Config.model_validate(read_toml(config_path))
    except ValidationError as error:
        raise ConfigError(f"{config_path} is not a valid config.\n{explain(error)}") from error

    system_template = None
    user_template = None
    if config.prompt is not None:
        if system_prompt_override is not None:
            system_template = read_text(system_prompt_override)
        elif config.prompt.system is not None:
            system_template = read_text(config_dir / config.prompt.system)
        user_template = read_text(config_dir / config.prompt.user)

    dataset_path = config_dir / config.dataset
    rows = read_toml(dataset_path).get("case", [])
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ConfigError(f"{dataset_path}: the cases must be [[case]] tables.")
    try:
        cases = build_cases(rows, config.assertion_templates, system_template, user_template)
    except ConfigError as error:
        raise ConfigError(f"{dataset_path}: {error}") from error

    if config.judge is None:
        for case in cases:
            for assertion in case.assertions:
                if isinstance(assertion, LlmRubric):
                    raise ConfigError(
                        f'{config_path}: case "{case.id}" has an llm-rubric assertion, '
                        "but the config has no [judge] table."
                    )

    baseline_path = config_dir / config.baseline
    return Project(
        config_path=config_path,
        config_dir=config_dir,
        config=config,
        cases=cases,
        baseline=read_baseline(baseline_path),
        baseline_path=baseline_path,
        report_path=config_dir / config.report,
    )


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ConfigError(f"Could not read {path}. {error}") from error


def read_toml(path: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(read_text(path))
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{path} is not valid TOML. {error}") from error


def read_baseline(path: Path) -> Baseline | None:
    if not path.exists():
        return None
    try:
        return Baseline.model_validate_json(read_text(path))
    except ValidationError as error:
        raise ConfigError(f"{path} is not a valid baseline.\n{explain(error)}") from error


def write_baseline(path: Path, baseline: Baseline) -> None:
    write_whole_file(path, baseline.model_dump_json(indent=2) + "\n")


def write_report(path: Path, report: Report) -> None:
    write_whole_file(path, report.model_dump_json(indent=2) + "\n")


def write_whole_file(path: Path, text: str) -> None:
    """Write beside the target and rename over it, so a crash cannot leave half a file."""
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(text, encoding="utf-8")
    temporary_path.replace(path)
