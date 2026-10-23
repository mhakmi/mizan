import json
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Literal, assert_never

from pydantic import JsonValue

ScalarKind = Literal["text", "decimal", "date", "exact"]
# A table of kinds compares an object such as money key by key. It is one level
# deep because Python 3.11 cannot write a recursive type alias.
Kind = ScalarKind | dict[str, ScalarKind]

# A numeric day-and-month form such as 03/10/2026 reads two ways, so none is here.
DATE_FORMATS = ("%Y-%m-%d", "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y")


class NoSuchField(Exception):
    """The path leads to nothing in the output."""


def split_path(path: str) -> tuple[str, ...]:
    keys = tuple(path.split("."))
    if "" in keys:
        raise ValueError(f'path "{path}" is not valid')
    return keys


def lookup(output: JsonValue, path: tuple[str, ...]) -> JsonValue:
    value = output
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise NoSuchField()
        value = value[key]
    return value


def parse_scalar(kind: ScalarKind, value: JsonValue) -> str | Decimal | date:
    """Return the form of a value that compares by meaning. Raises ValueError."""
    match kind:
        case "text":
            if isinstance(value, str):
                return " ".join(value.split()).casefold()
            raise ValueError("not text")
        case "decimal":
            number = decimal_from(value)
            if number is None:
                raise ValueError("not a decimal")
            return number
        case "date":
            day = date_from(value)
            if day is None:
                raise ValueError("not a date")
            return day
        case "exact":
            # In Python true equals 1. As JSON text they differ.
            return json.dumps(value, sort_keys=True)
        case _:
            assert_never(kind)


def decimal_from(value: JsonValue) -> Decimal | None:
    # bool is a subclass of int, and true is not a number.
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    return number


def date_from(value: JsonValue) -> date | None:
    if not isinstance(value, str):
        return None
    for date_format in DATE_FORMATS:
        try:
            return datetime.strptime(value, date_format).date()
        except ValueError:
            continue
    return None


def check_expected(kind: Kind, expected: JsonValue) -> None:
    """Raise ValueError when an expected value cannot be compared under its kind."""
    if expected is None:
        return
    if isinstance(kind, dict):
        if isinstance(expected, dict):
            for key, value in expected.items():
                if key not in kind:
                    raise ValueError(f'expected has the key "{key}", but its kinds do not')
                check_expected(kind[key], value)
            return
        raise ValueError(f"expected {show(expected)} is not an object")
    try:
        parse_scalar(kind, expected)
    except ValueError as error:
        raise ValueError(f"expected {show(expected)} is {error}") from None


def differences(kind: Kind, expected: JsonValue, read: JsonValue) -> str | None:
    """Return None when the read value equals the expected one, else the reason it does not.

    The expected value has passed check_expected.
    """
    if expected is None or read is None:
        if expected is None and read is None:
            return None
        return f"expected {show(expected)}, read {show(read)}"

    if isinstance(kind, dict):
        return object_differences(kind, expected, read)

    try:
        if same_scalar(kind, expected, read):
            return None
    except ValueError as error:
        return f"expected {show(expected)}, read {show(read)} ({error})"
    return f"expected {show(expected)}, read {show(read)}"


def same_scalar(kind: ScalarKind, expected: JsonValue, read: JsonValue) -> bool:
    return parse_scalar(kind, expected) == parse_scalar(kind, read)


def object_differences(
    kinds: dict[str, ScalarKind], expected: JsonValue, read: JsonValue
) -> str | None:
    if not isinstance(read, dict):
        return f"expected an object, read {show(read)}"
    assert isinstance(expected, dict), "check_expected requires an object"

    clauses = []
    for key, value in expected.items():
        if key not in read:
            clauses.append(f"{key}: missing")
            continue
        clause = differences(kinds[key], value, read[key])
        if clause is not None:
            clauses.append(f"{key}: {clause}")
    if not clauses:
        return None
    return "; ".join(clauses)


def show(value: JsonValue) -> str:
    return json.dumps(value, ensure_ascii=False)
