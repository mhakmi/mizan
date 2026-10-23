import re
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import TypeAdapter, ValidationError

from mizan.types import Assertion, Case, ConfigError, FieldEquals, Prompt, explain

VARIABLE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")

ASSERTION = TypeAdapter(Assertion)


def render(template: str, variables: Mapping[str, str]) -> str:
    def value_of(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in variables:
            raise ConfigError(f'A template uses {{{{{name}}}}}, but "{name}" is not defined.')
        return variables[name]

    return VARIABLE.sub(value_of, template)


def build_cases(
    rows: Sequence[Mapping[str, Any]],
    assertion_templates: Sequence[Mapping[str, Any]],
    system_template: str | None,
    user_template: str | None,
) -> tuple[Case, ...]:
    if not rows:
        raise ConfigError("The dataset has no cases.")

    cases = []
    seen_ids = set()
    for position, row in enumerate(rows, start=1):
        case_id = row.get("id")
        if not isinstance(case_id, str) or case_id == "":
            raise ConfigError(f'Case number {position} has no "id".')
        if case_id in seen_ids:
            raise ConfigError(f'The case id "{case_id}" is used more than once.')
        seen_ids.add(case_id)

        try:
            case = build_case(case_id, row, assertion_templates, system_template, user_template)
        except ConfigError as error:
            raise ConfigError(f'Case "{case_id}": {error}') from error
        cases.append(case)

    return tuple(cases)


def build_case(
    case_id: str,
    row: Mapping[str, Any],
    assertion_templates: Sequence[Mapping[str, Any]],
    system_template: str | None,
    user_template: str | None,
) -> Case:
    variables = {}
    for name, value in row.items():
        if name in ("id", "assert"):
            continue
        if not isinstance(value, str):
            raise ConfigError(f'The value of "{name}" must be a string.')
        variables[name] = value

    own_assertions = row.get("assert", [])
    if not isinstance(own_assertions, list):
        raise ConfigError('"assert" must be a list of tables.')

    prompt = None
    if user_template is not None:
        system = None
        if system_template is not None:
            system = render(system_template, variables)
        prompt = Prompt(system=system, user=render(user_template, variables))

    assertions = []
    for raw in [*assertion_templates, *own_assertions]:
        if isinstance(raw, FieldEquals):
            assertions.append(raw)
        else:
            assertions.append(build_assertion(raw, variables))
    if not assertions:
        raise ConfigError(
            "There is nothing to check. Add an [[assert]] table to the config "
            'or an "assert" list to the case.'
        )

    return Case(id=case_id, prompt=prompt, assertions=tuple(assertions))


def build_assertion(raw: Any, variables: Mapping[str, str]) -> Assertion:
    if not isinstance(raw, Mapping):
        raise ConfigError("Each assertion must be a table.")

    filled = {}
    for key, value in raw.items():
        if isinstance(value, str) and key != "type":
            filled[key] = render(value, variables)
        else:
            filled[key] = value

    try:
        return ASSERTION.validate_python(filled)
    except ValidationError as error:
        raise ConfigError(f"An assertion is not valid.\n{explain(error)}") from error
