from collections import Counter
from pathlib import Path
from typing import assert_never

from mizan.gate import count_cases, shared_baseline_score, shared_run_score
from mizan.types import (
    Assertion,
    Comparison,
    Contains,
    Errored,
    Gate,
    GateFailed,
    Incomplete,
    IsJson,
    LlmRubric,
    NotContains,
    Passed,
    Regex,
    RunSummary,
    Scored,
    Verdict,
)

CASES_SHOWN_ON_FAILURE = 5


def render_text(config_path: Path, summary: RunSummary, verdict: Verdict, gate: Gate) -> str:
    lines = [f"Mizan   {config_path}", f"Model   {summary.model}"]
    if summary.judge is not None:
        lines.append(f"Judge   {summary.judge}")
    lines.append("")

    match verdict:
        case Passed():
            lines.extend(table(summary, verdict.score, verdict.comparison, gate))
            lines.append("")
            lines.append("PASS")
            lines.extend(passed_details(summary, verdict, gate))
        case GateFailed():
            lines.extend(table(summary, verdict.score, verdict.comparison, gate))
            lines.append("")
            lines.append("FAIL")
            lines.extend(failed_details(summary, verdict))
        case Incomplete():
            lines.extend(table(summary, None, None, gate))
            lines.append("")
            lines.append("INCOMPLETE")
            lines.extend(incomplete_details(summary))
        case _:
            assert_never(verdict)

    return "\n".join(lines)


def table(
    summary: RunSummary, score: float | None, comparison: Comparison | None, gate: Gate
) -> list[str]:
    width = max(len("Case"), *(len(result.case_id) for result in summary.results))
    lines = [f"{'Case':<{width}}  Score  Baseline  Change  Prompt tokens"]

    for result in summary.results:
        if isinstance(result, Errored):
            lines.append(f"{result.case_id:<{width}}  error")
            continue

        baseline = "-"
        change = "-"
        if comparison is not None and result.case_id in comparison.changes:
            case_change = comparison.changes[result.case_id]
            baseline = f"{case_change.baseline:.2f}"
            change = format_change(case_change.current - case_change.baseline)
        line = (
            f"{result.case_id:<{width}}  {result.score:>5.2f}  {baseline:>8}  {change:>6}"
            f"  {result.prompt_tokens:>13}"
        )
        if gate.max_prompt_tokens is not None and result.prompt_tokens > gate.max_prompt_tokens:
            line += "  over"
        lines.append(line)

    if score is not None:
        baseline = "-"
        change = "-"
        if comparison is not None:
            baseline = f"{shared_baseline_score(comparison):.2f}"
            change = format_change(shared_run_score(comparison) - shared_baseline_score(comparison))
        largest = f"largest {largest_prompt(summary.scored())}"
        lines.append(f"{'Run':<{width}}  {score:>5.2f}  {baseline:>8}  {change:>6}  {largest:>13}")

    return lines


def format_change(change: float) -> str:
    text = f"{change:+.2f}"
    if text in ("+0.00", "-0.00"):
        return "0.00"
    return text


def largest_prompt(scored: tuple[Scored, ...]) -> int:
    return max(result.prompt_tokens for result in scored)


def passed_details(summary: RunSummary, verdict: Passed, gate: Gate) -> list[str]:
    score_line = f"  Score {verdict.score:.2f}."
    if verdict.comparison is not None:
        score_line += f" Baseline {shared_baseline_score(verdict.comparison):.2f}."
        score_line += f" Allowed drop {gate.max_drop:.2f}."
    elif gate.min_score is not None:
        score_line += f" Minimum {gate.min_score:.2f}."

    budget_line = f"  Largest prompt {largest_prompt(summary.scored())} tokens."
    if gate.max_prompt_tokens is not None:
        budget_line += f" Budget {gate.max_prompt_tokens}."

    return [score_line, budget_line, *(f"  {note}" for note in verdict.notes)]


def failed_details(summary: RunSummary, verdict: GateFailed) -> list[str]:
    width = max(len(failure.check) for failure in verdict.failures)
    lines = [f"  {failure.check:<{width}}   {failure.message}" for failure in verdict.failures]
    lines.extend(f"  {note}" for note in verdict.notes)

    only_the_budget_failed = all(
        failure.check == "max_prompt_tokens" for failure in verdict.failures
    )
    if only_the_budget_failed:
        return lines

    changes = {}
    if verdict.comparison is not None:
        changes = verdict.comparison.changes
    dropped = [
        result
        for result in summary.scored()
        if result.case_id in changes and result.score < changes[result.case_id].baseline
    ]
    lowest = [result for result in summary.scored() if result.score < 1]

    if dropped:
        dropped.sort(key=lambda result: result.score - changes[result.case_id].baseline)
        lines.extend(["", "Cases that dropped the most"])
        for result in dropped[:CASES_SHOWN_ON_FAILURE]:
            baseline = changes[result.case_id].baseline
            lines.append(f"  {result.case_id}  {baseline:.2f} to {result.score:.2f}")
            lines.extend(failed_checks(result))
    elif lowest:
        lowest.sort(key=lambda result: result.score)
        lines.extend(["", "Cases with the lowest scores"])
        for result in lowest[:CASES_SHOWN_ON_FAILURE]:
            lines.append(f"  {result.case_id}  {result.score:.2f}")
            lines.extend(failed_checks(result))
    return lines


def failed_checks(result: Scored) -> list[str]:
    failed = [check for check in result.checks if check.score < 1]
    labels = [describe(check.assertion) for check in failed]
    width = max((len(label) for label in labels), default=0)
    return [
        f"    {check.score:.2f}  {label:<{width}}    {check.reason}"
        for check, label in zip(failed, labels)
    ]


def describe(assertion: Assertion) -> str:
    match assertion:
        case Contains() | NotContains():
            text = f'{assertion.type} "{assertion.value}"'
        case Regex():
            text = f'{assertion.type} "{assertion.pattern}"'
        case IsJson() | LlmRubric():
            text = assertion.type
        case _:
            assert_never(assertion)
    if assertion.weight != 1:
        text += f" (weight {assertion.weight:g})"
    return text


def incomplete_details(summary: RunSummary) -> list[str]:
    errored = summary.errored()
    lines = [f"  {count_cases(len(errored))} could not be scored, so the gate was not checked."]
    for reason, count in Counter(result.reason for result in errored).items():
        lines.append(f"  {count_cases(count)}  {reason}")
    return lines
