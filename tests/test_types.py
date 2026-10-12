import pytest
from pydantic import TypeAdapter, ValidationError

from mizan.types import (
    Assertion,
    AssertionScore,
    Contains,
    Errored,
    Incomplete,
    LlmRubric,
    Report,
    RunSummary,
    Scored,
)


def scored(case_id, *checks):
    return Scored(case_id=case_id, output="an answer", prompt_tokens=10, checks=checks)


def test_a_case_score_is_the_weighted_mean_of_its_assertion_scores():
    rubric = LlmRubric(type="llm-rubric", rubric="Agrees with the reference.", weight=2)
    contains = Contains(type="contains", value="30 days")

    result = scored(
        "late-return",
        AssertionScore(assertion=rubric, score=0.25, reason="Offers a refund."),
        AssertionScore(assertion=contains, score=1.0, reason="Found."),
    )

    # (2 * 0.25 + 1 * 1.0) / 3. A plain mean would give 0.625.
    assert result.score == 0.5


def test_a_case_score_is_rounded_to_four_places():
    rubric = LlmRubric(type="llm-rubric", rubric="Agrees with the reference.", weight=2)
    contains = Contains(type="contains", value="3 to 5")

    result = scored(
        "shipping-time",
        AssertionScore(assertion=rubric, score=0.75, reason="Leaves out business days."),
        AssertionScore(assertion=contains, score=1.0, reason="Found."),
    )

    assert result.score == 0.8333


def test_a_summary_separates_scored_cases_from_errored_ones():
    check = AssertionScore(
        assertion=Contains(type="contains", value="x"), score=1.0, reason="Found."
    )
    summary = RunSummary(
        model="fake:model.toml",
        judge=None,
        results=(scored("first", check), Errored(case_id="second", reason="The call failed.")),
    )

    assert [result.case_id for result in summary.scored()] == ["first"]
    assert [result.case_id for result in summary.errored()] == ["second"]


def test_the_report_carries_the_case_score():
    check = AssertionScore(
        assertion=Contains(type="contains", value="x"), score=1.0, reason="Found."
    )
    summary = RunSummary(model="fake:model.toml", judge=None, results=(scored("first", check),))

    report = Report(verdict=Incomplete(), run=summary).model_dump(mode="json")

    assert report["run"]["results"][0]["score"] == 1.0


@pytest.mark.parametrize(
    "raw",
    [
        {"type": "regex", "pattern": "(unclosed"},
        {"type": "is-json", "json_schema": {"type": "no-such-type"}},
        {"type": "contains", "value": "x", "weight": 0},
        {"type": "contains", "value": "x", "threshold": 0.5},
        {"type": "icontains", "value": "x"},
    ],
    ids=["bad regex", "bad schema", "zero weight", "unknown key", "unknown type"],
)
def test_an_assertion_that_cannot_work_is_rejected_when_it_is_loaded(raw):
    with pytest.raises(ValidationError):
        TypeAdapter(Assertion).validate_python(raw)
