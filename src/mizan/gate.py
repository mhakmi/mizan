from collections.abc import Sequence
from statistics import fmean

from mizan.types import (
    SCORE_DECIMALS,
    Baseline,
    CaseChange,
    Comparison,
    Gate,
    GateFailed,
    GateFailure,
    Incomplete,
    Passed,
    RunSummary,
    Scored,
    Verdict,
)


def run_score(scored: Sequence[Scored]) -> float:
    return round(fmean(result.score for result in scored), SCORE_DECIMALS)


def compare(scored: Sequence[Scored], baseline: Baseline) -> Comparison | None:
    changes = {}
    new_cases = []
    for result in scored:
        if result.case_id in baseline.cases:
            changes[result.case_id] = CaseChange(
                baseline=baseline.cases[result.case_id], current=result.score
            )
        else:
            new_cases.append(result.case_id)
    if not changes:
        return None

    current_ids = {result.case_id for result in scored}
    removed_cases = [case_id for case_id in baseline.cases if case_id not in current_ids]
    return Comparison(
        changes=changes, new_cases=tuple(new_cases), removed_cases=tuple(removed_cases)
    )


def shared_baseline_score(comparison: Comparison) -> float:
    return fmean(change.baseline for change in comparison.changes.values())


def shared_run_score(comparison: Comparison) -> float:
    return fmean(change.current for change in comparison.changes.values())


def evaluate(summary: RunSummary, baseline: Baseline | None, gate: Gate) -> Verdict:
    """Decide the outcome of a run.

    min_score uses every case in the run. max_drop uses only the cases the
    baseline also has, so adding a hard case does not read as a regression.
    """
    if summary.errored():
        return Incomplete()

    scored = summary.scored()
    score = run_score(scored)
    comparison = None
    if baseline is not None:
        comparison = compare(scored, baseline)
    failures = []

    if gate.max_prompt_tokens is not None:
        over = [result for result in scored if result.prompt_tokens > gate.max_prompt_tokens]
        if over:
            largest = max(over, key=lambda result: result.prompt_tokens)
            failures.append(
                GateFailure(
                    check="max_prompt_tokens",
                    message=(
                        f"{count_cases(len(over))} sent more than {gate.max_prompt_tokens} "
                        f"prompt tokens. The largest is {largest.case_id} "
                        f"with {largest.prompt_tokens}."
                    ),
                )
            )

    if gate.min_score is not None and score < gate.min_score:
        failures.append(
            GateFailure(
                check="min_score",
                message=f"The run score is {score:.2f}. The minimum is {gate.min_score:.2f}.",
            )
        )

    if comparison is not None:
        before = shared_baseline_score(comparison)
        after = shared_run_score(comparison)
        drop = round(before - after, SCORE_DECIMALS)
        if drop > gate.max_drop:
            failures.append(
                GateFailure(
                    check="max_drop",
                    message=(
                        f"The run score fell {drop:.2f} below the baseline, "
                        f"from {before:.2f} to {after:.2f}. "
                        f"The allowed drop is {gate.max_drop:.2f}."
                    ),
                )
            )
        if gate.max_case_drop is not None:
            for case_id, change in comparison.changes.items():
                case_drop = round(change.baseline - change.current, SCORE_DECIMALS)
                if case_drop > gate.max_case_drop:
                    failures.append(
                        GateFailure(
                            check="max_case_drop",
                            message=(
                                f"{case_id} fell {case_drop:.2f} below its baseline, "
                                f"from {change.baseline:.2f} to {change.current:.2f}. "
                                f"The allowed drop for one case is {gate.max_case_drop:.2f}."
                            ),
                        )
                    )

    notes = tuple(build_notes(summary, baseline, comparison))
    if failures:
        return GateFailed(score=score, failures=tuple(failures), comparison=comparison, notes=notes)
    return Passed(score=score, comparison=comparison, notes=notes)


def build_notes(
    summary: RunSummary, baseline: Baseline | None, comparison: Comparison | None
) -> list[str]:
    if baseline is None:
        return ["Not compared with a baseline. Checked min_score and the token budget only."]

    notes = []
    if baseline.model != summary.model:
        notes.append(
            f"The baseline was recorded with the model {baseline.model}. "
            f"This run used {summary.model}."
        )
    if baseline.judge != summary.judge:
        notes.append(
            f"The baseline was recorded with the judge {baseline.judge or 'none'}. "
            f"This run used {summary.judge or 'none'}."
        )
    if comparison is None:
        notes.append("The baseline shares no case with the dataset, so nothing was compared.")
        return notes

    if comparison.new_cases:
        names = ", ".join(comparison.new_cases)
        notes.append(f"Not in the baseline, so left out of the comparison: {names}.")
    if comparison.removed_cases:
        names = ", ".join(comparison.removed_cases)
        notes.append(f"In the baseline but no longer in the dataset: {names}.")
    return notes


def count_cases(count: int) -> str:
    if count == 1:
        return "1 case"
    return f"{count} cases"


def baseline_from(summary: RunSummary) -> Baseline:
    cases = {result.case_id: result.score for result in summary.scored()}
    return Baseline(model=summary.model, judge=summary.judge, cases=cases)
