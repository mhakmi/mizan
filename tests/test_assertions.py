import pytest

from mizan.assertions import check, parse_judge_reply
from mizan.types import (
    Case,
    Contains,
    FieldEquals,
    IsJson,
    JudgeReplyError,
    LlmRubric,
    NotContains,
    Prompt,
    ProviderError,
    Regex,
    Reply,
)

ORDER_SCHEMA = {
    "type": "object",
    "properties": {"order_id": {"type": "integer"}},
    "required": ["order_id"],
}


def score(assertion, output, ask_judge=None):
    case = Case(
        id="late-return",
        prompt=Prompt(system="You answer for a shop.", user="Can I return boots after 40 days?"),
        assertions=(assertion,),
    )
    return check(assertion, case, output, ask_judge).score


@pytest.mark.parametrize(
    ("assertion", "output", "expected"),
    [
        (Contains(type="contains", value="30 days"), "Returns close after 30 days.", 1.0),
        (Contains(type="contains", value="30 days"), "Returns close after a month.", 0.0),
        (Contains(type="contains", value="30 Days"), "Returns close after 30 days.", 0.0),
        (Contains(type="contains", value="30 Days", ignore_case=True), "after 30 days.", 1.0),
        (NotContains(type="not-contains", value="cheaper"), "We do not compare prices.", 1.0),
        (NotContains(type="not-contains", value="cheaper"), "Ours is cheaper.", 0.0),
        (NotContains(type="not-contains", value="cheaper", ignore_case=True), "CHEAPER!", 0.0),
        (Regex(type="regex", pattern=r"\d+ business days"), "It takes 3 business days.", 1.0),
        (Regex(type="regex", pattern=r"\d+ business days"), "It takes a few days.", 0.0),
        (IsJson(type="is-json"), '{"order_id": 7}', 1.0),
        (IsJson(type="is-json"), "Your order id is 7.", 0.0),
        (IsJson(type="is-json"), '```json\n{"order_id": 7}\n```', 0.0),
        (IsJson(type="is-json", json_schema=ORDER_SCHEMA), '{"order_id": 7}', 1.0),
        (IsJson(type="is-json", json_schema=ORDER_SCHEMA), '{"order_id": "seven"}', 0.0),
        (IsJson(type="is-json", json_schema=ORDER_SCHEMA), "{}", 0.0),
    ],
)
def test_a_deterministic_assertion_scores_one_or_zero(assertion, output, expected):
    assert score(assertion, output) == expected


def field(path, expected, kind="text"):
    return FieldEquals(type="field-equals", path=path, expected=expected, kind=kind)


MONEY = {"amount": "decimal", "currency": "exact"}
RECEIPT = '{"date": "03 Oct 2026", "total": {"amount": "12.5", "currency": null}, "tip": null}'


@pytest.mark.parametrize(
    ("assertion", "output", "expected"),
    [
        (field("total.amount", "12.50", "decimal"), RECEIPT, 1.0),
        (field("total.amount", "12.51", "decimal"), RECEIPT, 0.0),
        (field("total.amount", "12.50"), RECEIPT, 0.0),
        (field("date", "2026-10-03", "date"), RECEIPT, 1.0),
        (field("date", "2026-10-04", "date"), RECEIPT, 0.0),
        (field("total", {"amount": "12.50", "currency": None}, MONEY), RECEIPT, 1.0),
        (field("total", {"amount": "12.50", "currency": "CAD"}, MONEY), RECEIPT, 0.0),
        (field("tip", None, "decimal"), RECEIPT, 1.0),
        (field("tip", "0.00", "decimal"), RECEIPT, 0.0),
        (field("change", None, "decimal"), RECEIPT, 0.0),
        (field("total.amount", "12.50", "decimal"), "Total: 12.50", 0.0),
    ],
)
def test_a_field_scores_one_only_when_it_means_the_expected_value(assertion, output, expected):
    assert score(assertion, output) == expected


@pytest.mark.parametrize(
    ("assertion", "output", "reason"),
    [
        (field("total.amount", "12.50", "decimal"), RECEIPT, "Field total.amount matches."),
        (
            field("total", {"amount": "1250.00"}, MONEY),
            RECEIPT,
            'Field total differs. amount: expected "1250.00", read "12.5".',
        ),
        (field("change", None), RECEIPT, 'No field "change" in the output.'),
        (field("tip", None), "Tip: none", "The output is not valid JSON."),
    ],
)
def test_a_field_check_says_what_it_read(assertion, output, reason):
    case = Case(id="receipt", prompt=None, assertions=(assertion,))

    assert check(assertion, case, output, None).reason == reason


def test_a_failed_schema_check_says_which_rule_the_output_broke():
    assertion = IsJson(type="is-json", json_schema=ORDER_SCHEMA)
    case = Case(
        id="order", prompt=Prompt(system=None, user="Which order?"), assertions=(assertion,)
    )

    result = check(assertion, case, "{}", None)

    assert "'order_id' is a required property" in result.reason


@pytest.mark.parametrize(
    ("grade", "expected"),
    [(1, 0.0), (2, 0.25), (3, 0.5), (4, 0.75), (5, 1.0)],
)
def test_a_rubric_maps_the_judge_grade_onto_zero_to_one(grade, expected):
    def judge(prompt: Prompt) -> Reply:
        return Reply(text=f'{{"reason": "Graded.", "score": {grade}}}', prompt_tokens=50)

    rubric = LlmRubric(type="llm-rubric", rubric="Agrees with the reference.")

    assert score(rubric, "Returns close after 30 days.", judge) == expected


def test_a_rubric_keeps_the_reason_the_judge_gave():
    def judge(prompt: Prompt) -> Reply:
        return Reply(
            text='{"reason": "It gives the wrong deadline.", "score": 2}', prompt_tokens=50
        )

    rubric = LlmRubric(type="llm-rubric", rubric="Agrees with the reference.")
    case = Case(id="late", prompt=Prompt(system=None, user="Can I?"), assertions=(rubric,))

    assert check(rubric, case, "Sure.", judge).reason == "It gives the wrong deadline."


def test_the_judge_sees_the_question_the_answer_and_the_rubric_but_not_the_system_prompt():
    seen = []

    def judge(prompt: Prompt) -> Reply:
        seen.append(prompt)
        return Reply(text='{"reason": "Fine.", "score": 5}', prompt_tokens=50)

    score(LlmRubric(type="llm-rubric", rubric="Mentions the deadline."), "Within 30 days.", judge)

    (prompt,) = seen
    assert "<input>\nCan I return boots after 40 days?\n</input>" in prompt.user
    assert "<answer>\nWithin 30 days.\n</answer>" in prompt.user
    assert "<rubric>\nMentions the deadline.\n</rubric>" in prompt.user
    assert "You answer for a shop." not in prompt.user


def test_a_failed_judge_call_is_reported_as_a_judge_failure():
    def judge(prompt: Prompt) -> Reply:
        raise ProviderError("HTTP 529: overloaded, after 3 attempts.")

    rubric = LlmRubric(type="llm-rubric", rubric="Agrees with the reference.")

    with pytest.raises(ProviderError, match="The judge call failed. HTTP 529"):
        score(rubric, "Within 30 days.", judge)


def test_a_clean_judge_reply_is_read():
    assert parse_judge_reply('{"reason": "Matches the reference.", "score": 4}') == (
        0.75,
        "Matches the reference.",
    )


def test_a_judge_reply_inside_a_code_fence_is_read():
    text = 'Here is my grade.\n```json\n{"reason": "Wrong deadline.", "score": 1}\n```\n'

    assert parse_judge_reply(text) == (0.0, "Wrong deadline.")


@pytest.mark.parametrize(
    "text",
    [
        "I would give this a 4 out of 5.",
        '{"reason": "Cut off mid-sentence, "score": ',
        '{"reason": "Out of range.", "score": 6}',
        '{"reason": "Zero is not a grade.", "score": 0}',
        '{"reason": "A fraction.", "score": 4.5}',
        '{"reason": "A word.", "score": "5"}',
        '{"reason": "A boolean.", "score": true}',
        '{"score": 5}',
        "",
    ],
)
def test_an_unreadable_judge_reply_is_an_error_and_never_a_score(text):
    with pytest.raises(JudgeReplyError, match="could not be read"):
        parse_judge_reply(text)
