from pathlib import Path

from mizan.gate import evaluate
from mizan.report import render_text
from mizan.types import (
    AssertionScore,
    Baseline,
    Contains,
    Errored,
    FieldEquals,
    Gate,
    LlmRubric,
    RunSummary,
    Scored,
)

CONFIG_PATH = Path("mizan.toml")
RUBRIC = LlmRubric(type="llm-rubric", rubric="Agrees with the reference.", weight=2)
CONTAINS = Contains(type="contains", value="30 days")


def scored(case_id, rubric_score, contains_score, prompt_tokens=100):
    return Scored(
        case_id=case_id,
        output="An answer.",
        prompt_tokens=prompt_tokens,
        checks=(
            AssertionScore(assertion=RUBRIC, score=rubric_score, reason=f"Judge on {case_id}."),
            AssertionScore(assertion=CONTAINS, score=contains_score, reason="Phrase check."),
        ),
    )


def render(results, baseline_cases, gate):
    summary = RunSummary(model="fake:model.toml", judge="fake:judge.toml", results=tuple(results))
    baseline = None
    if baseline_cases is not None:
        baseline = Baseline(model="fake:model.toml", judge="fake:judge.toml", cases=baseline_cases)
    return render_text(CONFIG_PATH, summary, evaluate(summary, baseline, gate), gate)


def test_a_pass_shows_each_case_beside_its_baseline_and_the_limits_that_held():
    text = render(
        [scored("late-return", 1.0, 1.0, 187), scored("shipping-time", 0.75, 1.0, 184)],
        {"late-return": 1.0, "shipping-time": 0.8333},
        Gate(min_score=0.8, max_prompt_tokens=400),
    )

    assert text == (
        "Mizan   mizan.toml\n"
        "Model   fake:model.toml\n"
        "Judge   fake:judge.toml\n"
        "\n"
        "Case           Score  Baseline  Change  Prompt tokens\n"
        "late-return     1.00      1.00    0.00            187\n"
        "shipping-time   0.83      0.83    0.00            184\n"
        "Run             0.92      0.92    0.00    largest 187\n"
        "\n"
        "PASS\n"
        "  Score 0.92. Baseline 0.92. Allowed drop 0.05.\n"
        "  Largest prompt 187 tokens. Budget 400."
    )


def test_a_quality_failure_explains_the_cases_that_dropped_worst_first():
    text = render(
        [
            scored("held", 1.0, 1.0),
            scored("slipped", 0.5, 1.0),
            scored("collapsed", 0.0, 0.0),
        ],
        {"held": 1.0, "slipped": 1.0, "collapsed": 1.0},
        Gate(max_drop=0.05),
    )

    details = text.split("Cases that dropped the most\n")[1]
    assert details == (
        "  collapsed  1.00 to 0.00\n"
        "    0.00  llm-rubric (weight 2)    Judge on collapsed.\n"
        '    0.00  contains "30 days"       Phrase check.\n'
        "  slipped  1.00 to 0.67\n"
        "    0.50  llm-rubric (weight 2)    Judge on slipped."
    )
    assert "slipped     0.67      1.00   -0.33            100\n" in text
    assert "max_drop   The run score fell 0.44 below the baseline" in text


def test_a_quality_failure_explains_at_most_five_cases():
    results = [scored(f"case-{number}", 0.0, 0.0) for number in range(7)]
    baseline = {result.case_id: 1.0 for result in results}

    text = render(results, baseline, Gate(max_drop=0.05))

    details = text.split("Cases that dropped the most\n")[1]
    assert details.count("  1.00 to 0.00\n") == 5


def test_a_failure_with_no_baseline_explains_the_lowest_scores():
    text = render([scored("weak", 0.25, 0.0), scored("fine", 1.0, 1.0)], None, Gate(min_score=0.8))

    assert "FAIL\n  min_score   The run score is 0.58. The minimum is 0.80." in text
    assert text.endswith(
        "Cases with the lowest scores\n"
        "  weak  0.17\n"
        "    0.25  llm-rubric (weight 2)    Judge on weak.\n"
        '    0.00  contains "30 days"       Phrase check.'
    )


def test_a_budget_overrun_marks_the_rows_and_does_not_blame_any_answer():
    text = render(
        [scored("short", 0.75, 1.0, 380), scored("long", 1.0, 1.0, 643)],
        {"short": 0.8333, "long": 1.0},
        Gate(max_prompt_tokens=400),
    )

    assert "short   0.83      0.83    0.00            380\n" in text
    assert "long    1.00      1.00    0.00            643  over\n" in text
    assert text.endswith(
        "FAIL\n"
        "  max_prompt_tokens   1 case sent more than 400 prompt tokens. "
        "The largest is long with 643."
    )


def test_an_incomplete_run_has_no_run_score_and_groups_the_errors_by_reason():
    overloaded = "The model call failed. HTTP 529: overloaded_error, after 3 attempts."
    text = render(
        [
            scored("late-return", 1.0, 1.0, 187),
            Errored(case_id="damaged-item", reason=overloaded),
            Errored(case_id="gift-card", reason=overloaded),
            Errored(case_id="cancel-order", reason="The judge reply could not be read."),
        ],
        {"late-return": 1.0},
        Gate(min_score=0.8),
    )

    assert "damaged-item  error\n" in text
    assert "\nRun " not in text
    assert text.endswith(
        "INCOMPLETE\n"
        "  3 cases could not be scored, so the gate was not checked.\n"
        f"  2 cases  {overloaded}\n"
        "  1 case  The judge reply could not be read."
    )


def test_notes_about_the_baseline_are_shown_under_the_verdict():
    text = render(
        [scored("old", 1.0, 1.0), scored("new", 0.0, 0.0)], {"old": 1.0, "gone": 1.0}, Gate()
    )

    assert "new    0.00         -       -            100\n" in text
    assert text.endswith(
        "PASS\n"
        "  Score 0.50. Baseline 1.00. Allowed drop 0.05.\n"
        "  Largest prompt 100 tokens.\n"
        "  Not in the baseline, so left out of the comparison: new.\n"
        "  In the baseline but no longer in the dataset: gone."
    )


def test_a_failed_field_check_names_the_field():
    total = FieldEquals(type="field-equals", path="total", expected="12.50", kind="decimal")
    result = Scored(
        case_id="blue-kettle",
        output='{"total": "1"}',
        prompt_tokens=12,
        checks=(
            AssertionScore(assertion=total, score=0.0, reason="Field total differs."),
            AssertionScore(assertion=CONTAINS, score=1.0, reason="Phrase check."),
        ),
    )

    text = render([result], None, Gate(min_score=0.8))

    assert text.endswith(
        "Cases with the lowest scores\n"
        "  blue-kettle  0.50\n"
        "    0.00  field-equals total    Field total differs."
    )
