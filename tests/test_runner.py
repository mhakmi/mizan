import threading

import pytest

from mizan.runner import run_cases
from mizan.types import Case, Contains, Errored, LlmRubric, Prompt, ProviderError, Reply, Scored


def case_with(case_id, question, *assertions):
    return Case(id=case_id, prompt=Prompt(system=None, user=question), assertions=assertions)


CONTAINS_30_DAYS = Contains(type="contains", value="30 days")
RUBRIC = LlmRubric(type="llm-rubric", rubric="Agrees with the reference.", weight=2)


def test_a_case_that_contains_the_phrase_scores_one():
    case = case_with("late-return", "Can I return boots after 40 days?", CONTAINS_30_DAYS)

    def model(prompt: Prompt) -> Reply:
        return Reply(text="Returns are accepted within 30 days.", prompt_tokens=7)

    (result,) = run_cases((case,), complete=model, ask_judge=None, concurrency=1)

    assert isinstance(result, Scored)
    assert result.score == 1.0
    assert result.output == "Returns are accepted within 30 days."
    assert result.prompt_tokens == 7


def test_a_case_score_weighs_the_judge_grade_against_the_other_assertions():
    case = case_with("late-return", "Can I return boots?", RUBRIC, CONTAINS_30_DAYS)

    def model(prompt: Prompt) -> Reply:
        return Reply(text="You have 30 days.", prompt_tokens=7)

    def judge(prompt: Prompt) -> Reply:
        return Reply(text='{"reason": "Leaves out the unused rule.", "score": 3}', prompt_tokens=90)

    (result,) = run_cases((case,), complete=model, ask_judge=judge, concurrency=1)

    # (2 * 0.5 + 1 * 1.0) / 3
    assert result.score == 0.6667


def test_a_failed_model_call_gives_an_errored_case_and_the_others_still_run():
    cases = (
        case_with("works", "First question", CONTAINS_30_DAYS),
        case_with("breaks", "Second question", CONTAINS_30_DAYS),
        case_with("works-too", "Third question", CONTAINS_30_DAYS),
    )

    def model(prompt: Prompt) -> Reply:
        if prompt.user == "Second question":
            raise ProviderError("HTTP 529: overloaded_error, after 3 attempts.")
        return Reply(text="Within 30 days.", prompt_tokens=5)

    results = run_cases(cases, complete=model, ask_judge=None, concurrency=2)

    assert [result.status for result in results] == ["scored", "errored", "scored"]
    assert results[1] == Errored(
        case_id="breaks",
        reason="The model call failed. HTTP 529: overloaded_error, after 3 attempts.",
    )


def test_a_failed_judge_call_gives_an_errored_case_and_not_a_zero():
    case = case_with("late-return", "Can I return boots?", CONTAINS_30_DAYS, RUBRIC)

    def model(prompt: Prompt) -> Reply:
        return Reply(text="You have 30 days.", prompt_tokens=7)

    def judge(prompt: Prompt) -> Reply:
        raise ProviderError("HTTP 401: invalid x-api-key")

    (result,) = run_cases((case,), complete=model, ask_judge=judge, concurrency=1)

    assert result == Errored(
        case_id="late-return", reason="The judge call failed. HTTP 401: invalid x-api-key"
    )


def test_an_unreadable_judge_reply_gives_an_errored_case_and_not_a_zero():
    case = case_with("late-return", "Can I return boots?", RUBRIC)

    def model(prompt: Prompt) -> Reply:
        return Reply(text="You have 30 days.", prompt_tokens=7)

    def judge(prompt: Prompt) -> Reply:
        return Reply(text="Looks good to me!", prompt_tokens=90)

    (result,) = run_cases((case,), complete=model, ask_judge=judge, concurrency=1)

    assert isinstance(result, Errored)
    assert "The judge reply could not be read." in result.reason


def test_results_keep_dataset_order_when_cases_finish_out_of_order():
    first_may_finish = threading.Event()
    cases = (
        case_with("slow", "slow", CONTAINS_30_DAYS),
        case_with("fast", "fast", CONTAINS_30_DAYS),
    )

    def model(prompt: Prompt) -> Reply:
        if prompt.user == "slow":
            assert first_may_finish.wait(timeout=5)
        else:
            first_may_finish.set()
        return Reply(text=f"{prompt.user}: 30 days", prompt_tokens=1)

    results = run_cases(cases, complete=model, ask_judge=None, concurrency=2)

    assert [result.case_id for result in results] == ["slow", "fast"]
    assert [result.output for result in results] == ["slow: 30 days", "fast: 30 days"]


def test_a_bug_in_a_model_function_is_not_hidden_as_an_errored_case():
    case = case_with("late-return", "Can I return boots?", CONTAINS_30_DAYS)

    def model(prompt: Prompt) -> Reply:
        raise KeyError("usage")

    with pytest.raises(KeyError):
        run_cases((case,), complete=model, ask_judge=None, concurrency=1)


def test_a_bug_in_a_judge_function_is_not_hidden_as_an_errored_case():
    case = case_with("late-return", "Can I return boots?", RUBRIC)

    def model(prompt: Prompt) -> Reply:
        return Reply(text="You have 30 days.", prompt_tokens=7)

    def judge(prompt: Prompt) -> Reply:
        raise KeyError("usage")

    with pytest.raises(KeyError):
        run_cases((case,), complete=model, ask_judge=judge, concurrency=1)
