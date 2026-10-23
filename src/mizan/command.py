import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mizan.cases import render
from mizan.types import Answer, Case, CommandModel, ConfigError, ProviderError, Reply

SHOWN_CHARACTERS = 80


@dataclass(frozen=True)
class CommandRun:
    stdout: str
    exit_status: int


@dataclass(frozen=True)
class ParsedLines:
    by_id: Mapping[str, list[dict[str, Any]]]
    line_count: int
    unread: list[str]
    """Lines that are not a JSON object with a string id, cut short for a message."""
    first_keys: list[str] | None
    """The keys of the first line that is a JSON object, to show when no id matched."""


def answers_from_command(
    settings: CommandModel, config_dir: Path, folder: Path | None, cases: Sequence[Case]
) -> Answer:
    """Run the program once and return a lookup of its reply for each case.

    Every outcome is decided before the runner starts its threads, so they share
    nothing that changes. Raises ConfigError when the program cannot start.
    """
    run = run_process(command_line(settings.argv, folder), config_dir, settings.env)
    parsed = parse_lines(run.stdout, settings.id_key)
    any_case_matched = any(case.id in parsed.by_id for case in cases)

    outcomes = {}
    for case in cases:
        outcomes[case.id] = outcome_for(case.id, parsed, run, settings, any_case_matched)

    def answer(case: Case) -> Reply:
        outcome = outcomes[case.id]
        if isinstance(outcome, ProviderError):
            raise outcome
        return outcome

    return answer


def command_line(argv: Sequence[str], folder: Path | None) -> list[str]:
    variables = {}
    if folder is not None:
        variables["folder"] = str(folder.resolve())
    try:
        return [render(part, variables) for part in argv]
    except ConfigError as error:
        raise ConfigError(
            f"[model] argv: {error} Only a dataset folder defines {{{{folder}}}}."
        ) from error


def run_process(argv: Sequence[str], cwd: Path, env: Mapping[str, str]) -> CommandRun:
    # The program's stderr is left alone, so its progress and errors show as it runs.
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            env={**os.environ, **env},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            check=False,
        )
    except OSError as error:
        raise ConfigError(f'The command "{argv[0]}" could not start. {error}') from error
    return CommandRun(
        stdout=completed.stdout.decode("utf-8", errors="replace"),
        exit_status=completed.returncode,
    )


def parse_lines(stdout: str, id_key: str) -> ParsedLines:
    by_id: dict[str, list[dict[str, Any]]] = {}
    line_count = 0
    unread = []
    first_keys = None
    for line in stdout.splitlines():
        if line.strip() == "":
            continue
        line_count += 1
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            unread.append(shorten(line))
            continue

        if isinstance(value, dict) and first_keys is None:
            first_keys = list(value)
        if isinstance(value, dict) and isinstance(value.get(id_key), str):
            by_id.setdefault(value[id_key], []).append(value)
        else:
            unread.append(shorten(line))

    return ParsedLines(by_id=by_id, line_count=line_count, unread=unread, first_keys=first_keys)


def outcome_for(
    case_id: str,
    parsed: ParsedLines,
    run: CommandRun,
    settings: CommandModel,
    any_case_matched: bool,
) -> Reply | ProviderError:
    lines = parsed.by_id.get(case_id, [])
    if not lines:
        return ProviderError(no_line_reason(case_id, parsed, run, settings, any_case_matched))
    if len(lines) > 1:
        return ProviderError(f'The command wrote {len(lines)} lines for "{case_id}".')

    line = lines[0]
    error = line.get(settings.error_key)
    if error is not None:
        return ProviderError(
            f'The command reported an error for "{case_id}": {output_text(error)}.'
        )

    output = line.get(settings.output_key)
    if output is None:
        return ProviderError(f'The line for "{case_id}" has no "{settings.output_key}".')

    tokens = line.get(settings.tokens_key)
    # bool is a subclass of int, and true is not a count.
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
        return ProviderError(
            f'The line for "{case_id}" has no whole number "{settings.tokens_key}".'
        )

    return Reply(text=output_text(output), prompt_tokens=tokens)


def no_line_reason(
    case_id: str,
    parsed: ParsedLines,
    run: CommandRun,
    settings: CommandModel,
    any_case_matched: bool,
) -> str:
    # When no case matched, the id key is likely wrong. The message leaves out the
    # case id, so the report shows it once for all the cases.
    if not any_case_matched and parsed.first_keys is not None:
        reason = (
            f'No line names a case in its "{settings.id_key}" key. '
            f"The command wrote {count_lines(parsed.line_count)}. "
            f"The first object has the keys: {', '.join(parsed.first_keys)}."
        )
    else:
        reason = f'The command wrote no line for "{case_id}".'
        if parsed.unread:
            reason += (
                f" It also wrote {count_lines(len(parsed.unread))} that Mizan could not use."
                f' The first is "{parsed.unread[0]}".'
            )
    if run.exit_status != 0:
        reason += f" It exited with status {run.exit_status}."
    return reason


def output_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def shorten(line: str) -> str:
    return " ".join(line.split())[:SHOWN_CHARACTERS]


def count_lines(count: int) -> str:
    if count == 1:
        return "1 line"
    return f"{count} lines"
