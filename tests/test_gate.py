from mizan.gate import baseline_from, evaluate, run_score
from mizan.types import (
    AssertionScore,
    Baseline,
    CaseChange,
    Contains,
    Errored,
    Gate,
    GateFailed,
    Incomplete,
    Passed,
    RunSummary,
    Scored,
)

MODEL = "fake:model.toml"
JUDGE = "fake:judge.toml"


def scored(case_id, score, prompt_tokens=100):
    check = AssertionScore(
        assertion=Contains(type="contains", value="x"), score=score, reason="A reason."
    )
    return Scored(
        case_id=case_id, output="An answer.", prompt_tokens=prompt_tokens, checks=(check,)
    )


def summary(*results):
    return RunSummary(model=MODEL, judge=JUDGE, results=results)


def baseline(**cases):
    return Baseline(model=MODEL, judge=JUDGE, cases=cases)


def failed_checks(verdict):
    assert isinstance(verdict, GateFailed)
    return [failure.check for failure in verdict.failures]


def test_the_run_score_is_the_plain_mean_of_the_case_scores():
    assert run_score([scored("a", 1.0), scored("b", 0.5), scored("c", 0.0)]) == 0.5


def test_a_run_that_matches_its_baseline_passes():
    verdict = evaluate(
        summary(scored("a", 1.0), scored("b", 0.75)),
        baseline(a=1.0, b=0.75),
        Gate(min_score=0.8, max_drop=0.05, max_prompt_tokens=400),
    )

    assert isinstance(verdict, Passed)
    assert verdict.score == 0.875
    assert verdict.notes == ()
    assert verdict.comparison.changes == {
        "a": CaseChange(baseline=1.0, current=1.0),
        "b": CaseChange(baseline=0.75, current=0.75),
    }


def test_a_drop_inside_the_allowance_passes():
    verdict = evaluate(
        summary(scored("a", 1.0), scored("b", 0.9)),
        baseline(a=1.0, b=1.0),
        Gate(max_drop=0.05),
    )

    assert isinstance(verdict, Passed)


def test_a_run_below_the_minimum_fails_even_with_no_baseline():
    verdict = evaluate(summary(scored("a", 1.0), scored("b", 0.5)), None, Gate(min_score=0.8))

    assert failed_checks(verdict) == ["min_score"]
    assert verdict.failures[0].message == "The run score is 0.75. The minimum is 0.80."


def test_a_run_that_falls_further_than_the_allowance_fails():
    verdict = evaluate(
        summary(scored("a", 0.5), scored("b", 1.0)),
        baseline(a=1.0, b=1.0),
        Gate(max_drop=0.05),
    )

    assert failed_checks(verdict) == ["max_drop"]
    assert verdict.failures[0].message == (
        "The run score fell 0.25 below the baseline, from 1.00 to 0.75. The allowed drop is 0.05."
    )


def test_a_new_hard_case_does_not_count_as_a_drop():
    verdict = evaluate(
        summary(scored("a", 1.0), scored("b", 1.0), scored("hard", 0.0)),
        baseline(a=1.0, b=1.0),
        Gate(max_drop=0.05),
    )

    assert isinstance(verdict, Passed)
    assert verdict.score == 0.6667
    assert verdict.comparison.new_cases == ("hard",)
    assert verdict.notes == ("Not in the baseline, so left out of the comparison: hard.",)


def test_a_removed_case_does_not_hide_a_drop_in_the_cases_that_remain():
    verdict = evaluate(
        summary(scored("a", 0.5)),
        baseline(a=1.0, weak=0.0),
        Gate(max_drop=0.05),
    )

    assert failed_checks(verdict) == ["max_drop"]
    assert verdict.comparison.removed_cases == ("weak",)
    assert verdict.notes == ("In the baseline but no longer in the dataset: weak.",)


def test_one_case_that_collapses_fails_when_the_average_hides_it():
    results = [scored(f"steady-{number}", 1.0) for number in range(19)]
    cases = {result.case_id: 1.0 for result in results}

    verdict = evaluate(
        summary(*results, scored("collapsed", 0.2)),
        baseline(**cases, collapsed=1.0),
        Gate(max_drop=0.05, max_case_drop=0.5),
    )

    assert failed_checks(verdict) == ["max_case_drop"]
    assert verdict.failures[0].message == (
        "collapsed fell 0.80 below its baseline, from 1.00 to 0.20. "
        "The allowed drop for one case is 0.50."
    )


def test_a_single_case_drop_is_ignored_unless_the_limit_is_set():
    verdict = evaluate(
        summary(scored("a", 1.0), scored("b", 1.0), scored("c", 0.9)),
        baseline(a=1.0, b=1.0, c=1.0),
        Gate(max_drop=0.05),
    )

    assert isinstance(verdict, Passed)


def test_a_prompt_over_the_token_budget_fails_while_quality_holds():
    verdict = evaluate(
        summary(scored("a", 1.0, prompt_tokens=401), scored("b", 1.0, prompt_tokens=650)),
        baseline(a=1.0, b=1.0),
        Gate(min_score=0.8, max_drop=0.05, max_prompt_tokens=400),
    )

    assert failed_checks(verdict) == ["max_prompt_tokens"]
    assert verdict.score == 1.0
    assert verdict.failures[0].message == (
        "2 cases sent more than 400 prompt tokens. The largest is b with 650."
    )


def test_a_prompt_exactly_at_the_token_budget_passes():
    verdict = evaluate(
        summary(scored("a", 1.0, prompt_tokens=400)), None, Gate(max_prompt_tokens=400)
    )

    assert isinstance(verdict, Passed)


def test_every_broken_limit_is_reported_at_once():
    verdict = evaluate(
        summary(scored("a", 0.0, prompt_tokens=500)),
        baseline(a=1.0),
        Gate(min_score=0.8, max_drop=0.05, max_case_drop=0.5, max_prompt_tokens=400),
    )

    assert failed_checks(verdict) == ["max_prompt_tokens", "min_score", "max_drop", "max_case_drop"]


def test_a_run_with_an_errored_case_has_no_score_and_no_gate_result():
    verdict = evaluate(
        summary(scored("a", 0.0), Errored(case_id="b", reason="The model call failed.")),
        baseline(a=1.0, b=1.0),
        Gate(min_score=0.8),
    )

    assert verdict == Incomplete()
    assert not hasattr(verdict, "score")


def test_a_run_with_no_baseline_checks_the_minimum_and_the_budget_and_says_so():
    verdict = evaluate(
        summary(scored("a", 1.0)), None, Gate(min_score=0.8, max_drop=0.05, max_prompt_tokens=400)
    )

    assert verdict == Passed(
        score=1.0,
        comparison=None,
        notes=("Not compared with a baseline. Checked min_score and the token budget only.",),
    )


def test_a_baseline_that_shares_no_case_is_not_compared():
    verdict = evaluate(summary(scored("new", 0.2)), baseline(old=1.0), Gate(max_drop=0.05))

    assert isinstance(verdict, Passed)
    assert verdict.comparison is None
    assert verdict.notes == (
        "The baseline shares no case with the dataset, so nothing was compared.",
    )


def test_a_baseline_from_another_model_or_judge_is_pointed_out():
    other = Baseline(model="anthropic:claude-opus-5-5", judge=None, cases={"a": 1.0})

    verdict = evaluate(summary(scored("a", 1.0)), other, Gate())

    model_note, judge_note = verdict.notes
    assert model_note == (
        "The baseline was recorded with the model anthropic:claude-opus-5-5. "
        "This run used fake:model.toml."
    )
    assert judge_note == (
        "The baseline was recorded with the judge none. This run used fake:judge.toml."
    )


def test_the_baseline_records_one_score_per_case_in_dataset_order():
    recorded = baseline_from(summary(scored("b", 0.75), scored("a", 1.0)))

    assert recorded == Baseline(model=MODEL, judge=JUDGE, cases={"b": 0.75, "a": 1.0})
    assert list(recorded.cases) == ["b", "a"]
