import pytest

from mizan.cases import build_cases, render
from mizan.types import ConfigError, Contains, LlmRubric, NotContains

USER_TEMPLATE = "Customer message:\n\n{{question}}"


def test_render_fills_variables_and_leaves_other_braces_alone():
    template = 'Reply as JSON like {"answer": "..."} to: {{question}}'

    assert render(template, {"question": "Where is my order?"}) == (
        'Reply as JSON like {"answer": "..."} to: Where is my order?'
    )


def test_render_names_the_variable_a_case_does_not_define():
    with pytest.raises(ConfigError, match="refrence"):
        render("Reference: {{refrence}}", {"reference": "30 days"})


def test_a_case_gets_a_rendered_prompt_and_filled_assertions():
    rows = [
        {
            "id": "late-return",
            "question": "Can I return boots after 40 days?",
            "reference": "Returns close after 30 days.",
            "must_include": "30 days",
        }
    ]
    defaults = [
        {"type": "llm-rubric", "rubric": "Agrees with: {{reference}}", "weight": 2},
        {"type": "contains", "value": "{{must_include}}"},
    ]

    (case,) = build_cases(rows, defaults, "You work at {{question}}", USER_TEMPLATE)

    assert case.prompt.system == "You work at Can I return boots after 40 days?"
    assert case.prompt.user == "Customer message:\n\nCan I return boots after 40 days?"
    assert case.assertions == (
        LlmRubric(type="llm-rubric", rubric="Agrees with: Returns close after 30 days.", weight=2),
        Contains(type="contains", value="30 days"),
    )


def test_a_case_runs_the_default_assertions_and_then_its_own():
    rows = [
        {
            "id": "competitor-question",
            "question": "Is your tent cheaper?",
            "assert": [{"type": "not-contains", "value": "cheaper than"}],
        }
    ]
    defaults = [{"type": "contains", "value": "person"}]

    (case,) = build_cases(rows, defaults, None, USER_TEMPLATE)

    assert case.prompt.system is None
    assert case.assertions == (
        Contains(type="contains", value="person"),
        NotContains(type="not-contains", value="cheaper than"),
    )


CONTAINS = [{"type": "contains", "value": "x"}]


@pytest.mark.parametrize(
    ("rows", "defaults", "message"),
    [
        ([], CONTAINS, "no cases"),
        ([{"question": "Hello?"}], CONTAINS, "Case number 1"),
        (
            [{"id": "same", "question": "One?"}, {"id": "same", "question": "Two?"}],
            CONTAINS,
            '"same" is used more than once',
        ),
        ([{"id": "count", "question": 3}], CONTAINS, 'Case "count".*"question" must be a string'),
        ([{"id": "no-question"}], CONTAINS, 'Case "no-question".*question'),
        ([{"id": "unchecked", "question": "Hello?"}], [], 'Case "unchecked".*nothing to check'),
        (
            [{"id": "bad-pattern", "question": "Hello?"}],
            [{"type": "regex", "pattern": "(unclosed"}],
            'Case "bad-pattern".*regular expression',
        ),
        (
            [{"id": "typo", "question": "Hello?"}],
            [{"type": "contains", "value": "{{must_inclde}}"}],
            'Case "typo".*must_inclde',
        ),
    ],
    ids=[
        "empty dataset",
        "missing id",
        "duplicate id",
        "value is not a string",
        "prompt variable not defined",
        "no assertions",
        "bad regex",
        "assertion variable not defined",
    ],
)
def test_a_dataset_that_cannot_run_is_rejected_with_the_case_named(rows, defaults, message):
    with pytest.raises(ConfigError, match=f"(?s){message}"):
        build_cases(rows, defaults, None, USER_TEMPLATE)
