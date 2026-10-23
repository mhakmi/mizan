import json
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, assert_never

from pydantic import ValidationError

from mizan.cases import build_cases
from mizan.fields import Kind
from mizan.types import (
    Baseline,
    Case,
    Config,
    ConfigError,
    FieldEquals,
    FolderDataset,
    LlmRubric,
    Report,
    explain,
)

EXPECTED_SUFFIX = ".expected.json"


@dataclass(frozen=True)
class Project:
    config_path: Path
    config_dir: Path
    config: Config
    cases: tuple[Case, ...]
    folder: Path | None
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

    if system_prompt_override is not None and config.prompt is None:
        raise ConfigError(
            "--system-prompt needs a [prompt] table, and this project runs a command."
        )

    system_template = None
    user_template = None
    if config.prompt is not None:
        if system_prompt_override is not None:
            system_template = read_text(system_prompt_override)
        elif config.prompt.system is not None:
            system_template = read_text(config_dir / config.prompt.system)
        user_template = read_text(config_dir / config.prompt.user)

    match config.dataset:
        case FolderDataset():
            dataset_path = config_dir / config.dataset.folder
            folder = dataset_path
            rows = read_folder_rows(dataset_path, config.dataset.kinds)
        case Path():
            dataset_path = config_dir / config.dataset
            folder = None
            rows = read_case_tables(dataset_path)
        case _:
            assert_never(config.dataset)
    try:
        cases = build_cases(rows, config.assertion_templates, system_template, user_template)
    except ConfigError as error:
        raise ConfigError(f"{dataset_path}: {error}") from error

    for case in cases:
        for assertion in case.assertions:
            if not isinstance(assertion, LlmRubric):
                continue
            if case.prompt is None:
                raise ConfigError(
                    f'{config_path}: case "{case.id}" has an llm-rubric assertion, '
                    "but a command project has no prompt to show the judge."
                )
            if config.judge is None:
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
        folder=folder,
        baseline=read_baseline(baseline_path),
        baseline_path=baseline_path,
        report_path=config_dir / config.report,
    )


def read_case_tables(path: Path) -> list[dict[str, Any]]:
    rows = read_toml(path).get("case", [])
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ConfigError(f"{path}: the cases must be [[case]] tables.")
    return rows


def read_folder_rows(folder: Path, kinds: Mapping[str, Kind]) -> list[dict[str, Any]]:
    if not folder.is_dir():
        raise ConfigError(f"The dataset folder {folder} does not exist.")

    rows = []
    for path in sorted(folder.glob("*" + EXPECTED_SUFFIX)):
        try:
            expected = json.loads(read_text(path))
        except json.JSONDecodeError as error:
            raise ConfigError(f"{path} is not valid JSON. {error}") from error
        if not isinstance(expected, dict):
            raise ConfigError(f"{path} must hold one JSON object.")

        assertions = []
        for key, value in expected.items():
            if "." in key:
                raise ConfigError(f'{path}: the key "{key}" has a dot, so no path can name it.')
            try:
                assertion = FieldEquals(
                    type="field-equals", path=key, expected=value, kind=kinds.get(key, "text")
                )
            except ValidationError as error:
                raise ConfigError(
                    f'{path}: the key "{key}" cannot be checked.\n{explain(error)}'
                ) from error
            assertions.append(assertion)
        rows.append({"id": path.name.removesuffix(EXPECTED_SUFFIX), "assert": assertions})
    return rows


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
