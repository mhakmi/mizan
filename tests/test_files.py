import json

import pytest

from mizan.assertions import check
from mizan.files import load_project, read_folder_rows, write_baseline, write_report
from mizan.types import (
    AssertionScore,
    Baseline,
    ConfigError,
    Contains,
    FakeModel,
    FieldEquals,
    Incomplete,
    Report,
    RunSummary,
    Scored,
)

CONFIG = """\
dataset = "cases.toml"

[prompt]
system = "prompts/system.md"
user = "prompts/user.md"

[model]
provider = "fake"
replies = "fake/model.toml"

[gate]
min_score = 0.8
max_prompt_tokens = 400

[[assert]]
type = "contains"
value = "{{must_include}}"
"""

CASES = """\
[[case]]
id = "late-return"
question = "Can I return boots after 40 days?"
must_include = "30 days"

[[case]]
id = "damaged-item"
question = "My tent arrived torn."
must_include = "replace"
"""


@pytest.fixture
def project_dir(tmp_path):
    (tmp_path / "prompts").mkdir()
    (tmp_path / "mizan.toml").write_text(CONFIG)
    (tmp_path / "cases.toml").write_text(CASES)
    (tmp_path / "prompts" / "system.md").write_text("You answer for a shop.\n")
    (tmp_path / "prompts" / "user.md").write_text("Customer message:\n\n{{question}}\n")
    return tmp_path


def test_a_project_loads_with_its_prompts_rendered_into_each_case(project_dir):
    project = load_project(project_dir / "mizan.toml")

    assert [case.id for case in project.cases] == ["late-return", "damaged-item"]
    assert project.cases[0].prompt.system == "You answer for a shop.\n"
    assert (
        project.cases[0].prompt.user == "Customer message:\n\nCan I return boots after 40 days?\n"
    )
    assert project.cases[1].assertions == (Contains(type="contains", value="replace"),)
    assert project.config.model == FakeModel(provider="fake", replies="fake/model.toml")
    assert project.config.gate.max_prompt_tokens == 400


def test_paths_in_the_config_are_relative_to_the_config_file(project_dir):
    project = load_project(project_dir / "mizan.toml")

    assert project.config_dir == project_dir
    assert project.baseline_path == project_dir / "mizan.baseline.json"
    assert project.report_path == project_dir / "mizan-report.json"


def test_a_project_with_no_baseline_file_has_no_baseline(project_dir):
    assert load_project(project_dir / "mizan.toml").baseline is None


def test_a_committed_baseline_is_loaded(project_dir):
    (project_dir / "mizan.baseline.json").write_text(
        '{"version": 1, "model": "fake:model.toml", "judge": null, "cases": {"late-return": 1.0}}'
    )

    assert load_project(project_dir / "mizan.toml").baseline == Baseline(
        model="fake:model.toml", judge=None, cases={"late-return": 1.0}
    )


def test_a_system_prompt_override_replaces_the_configured_one(project_dir, tmp_path):
    draft = tmp_path / "draft.md"
    draft.write_text("You answer for a different shop.\n")

    project = load_project(project_dir / "mizan.toml", system_prompt_override=draft)

    assert project.cases[0].prompt.system == "You answer for a different shop.\n"
    assert (
        project.cases[0].prompt.user == "Customer message:\n\nCan I return boots after 40 days?\n"
    )


def rewrite(path, old, new):
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new))


def test_an_unknown_template_variable_stops_the_load_and_names_it(project_dir):
    rewrite(project_dir / "prompts" / "user.md", "{{question}}", "{{qestion}}")

    with pytest.raises(ConfigError, match='cases.toml: Case "late-return".*qestion'):
        load_project(project_dir / "mizan.toml")


def test_a_duplicate_case_id_stops_the_load(project_dir):
    rewrite(project_dir / "cases.toml", 'id = "damaged-item"', 'id = "late-return"')

    with pytest.raises(ConfigError, match='cases.toml: The case id "late-return" is used more'):
        load_project(project_dir / "mizan.toml")


def test_a_rubric_without_a_judge_stops_the_load(project_dir):
    rewrite(
        project_dir / "mizan.toml",
        'type = "contains"\nvalue = "{{must_include}}"',
        'type = "llm-rubric"\nrubric = "Mentions {{must_include}}."',
    )

    with pytest.raises(ConfigError, match=r"llm-rubric assertion, but the config has no \[judge\]"):
        load_project(project_dir / "mizan.toml")


def test_a_bad_regex_stops_the_load(project_dir):
    rewrite(
        project_dir / "mizan.toml",
        'type = "contains"\nvalue = "{{must_include}}"',
        'type = "regex"\npattern = "(unclosed"',
    )

    with pytest.raises(ConfigError, match="(?s)cases.toml.*not a valid regular expression"):
        load_project(project_dir / "mizan.toml")


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('dataset = "cases.toml"', 'dataset = "missing.toml"', "Could not read .*missing.toml"),
        ('dataset = "cases.toml"', 'dataset = "cases.toml', "mizan.toml is not valid TOML"),
        ('dataset = "cases.toml"\n', "", "(?s)mizan.toml is not a valid config.*dataset"),
        ("min_score = 0.8", "min_score = 80", "(?s)not a valid config.*gate.min_score"),
        ("min_score = 0.8", "minimum = 0.8", "(?s)not a valid config.*gate.minimum"),
        ('provider = "fake"', 'provider = "bedrock"', "(?s)not a valid config.*model"),
        ('user = "prompts/user.md"', 'user = "prompts/gone.md"', "Could not read .*gone.md"),
        (
            '[prompt]\nsystem = "prompts/system.md"\nuser = "prompts/user.md"\n',
            "",
            r'(?s)not a valid config.*provider = "fake" needs a \[prompt\] table',
        ),
    ],
    ids=[
        "missing dataset",
        "broken TOML",
        "missing key",
        "value out of range",
        "unknown key",
        "unknown provider",
        "missing prompt file",
        "chat model without a prompt",
    ],
)
def test_a_config_mistake_is_reported_with_the_file_and_the_key(project_dir, old, new, message):
    rewrite(project_dir / "mizan.toml", old, new)

    with pytest.raises(ConfigError, match=message):
        load_project(project_dir / "mizan.toml")


def test_a_broken_baseline_is_an_error_and_not_a_missing_baseline(project_dir):
    (project_dir / "mizan.baseline.json").write_text('{"version": 1, "cases": ')

    with pytest.raises(ConfigError, match="mizan.baseline.json is not a valid baseline"):
        load_project(project_dir / "mizan.toml")


def test_writing_the_same_baseline_twice_gives_the_same_bytes(tmp_path):
    path = tmp_path / "mizan.baseline.json"
    baseline = Baseline(
        model="fake:model.toml", judge="fake:judge.toml", cases={"b": 0.8333, "a": 1.0}
    )

    write_baseline(path, baseline)
    first = path.read_bytes()
    write_baseline(path, baseline)

    assert path.read_bytes() == first
    assert first.endswith(b"}\n")
    assert json.loads(first) == {
        "version": 1,
        "model": "fake:model.toml",
        "judge": "fake:judge.toml",
        "cases": {"b": 0.8333, "a": 1.0},
    }
    assert [entry.name for entry in tmp_path.iterdir()] == ["mizan.baseline.json"]


def test_the_report_holds_the_full_output_of_each_case(tmp_path):
    path = tmp_path / "mizan-report.json"
    check = AssertionScore(
        assertion=Contains(type="contains", value="30 days"),
        score=0.0,
        reason='The output does not contain "30 days".',
    )
    result = Scored(
        case_id="late-return", output="Send them back any time.", prompt_tokens=152, checks=(check,)
    )
    summary = RunSummary(model="fake:model.toml", judge=None, results=(result,))

    write_report(path, Report(verdict=Incomplete(), run=summary))

    report = json.loads(path.read_text())
    assert report["version"] == 1
    assert report["verdict"] == {"status": "incomplete"}
    assert report["run"]["results"] == [
        {
            "status": "scored",
            "case_id": "late-return",
            "output": "Send them back any time.",
            "prompt_tokens": 152,
            "checks": [
                {
                    "assertion": {
                        "type": "contains",
                        "value": "30 days",
                        "ignore_case": False,
                        "weight": 1.0,
                    },
                    "score": 0.0,
                    "reason": 'The output does not contain "30 days".',
                }
            ],
            "score": 0.0,
        }
    ]


def write_expected(folder, case_id, text):
    folder.mkdir(exist_ok=True)
    (folder / f"{case_id}.expected.json").write_text(text)


def test_each_expected_file_in_a_folder_is_a_case_with_one_field_check_per_key(tmp_path):
    samples = tmp_path / "samples"
    write_expected(samples, "corner-market", '{"date": "2026-10-03", "total": {"amount": "12.50"}}')
    write_expected(samples, "blue-kettle", '{"correspondent": "Blue Kettle"}')
    (samples / "blue-kettle.png").write_bytes(b"not an image")
    (samples / "scratch.json").write_text("not JSON either")

    rows = read_folder_rows(samples, {"date": "date", "total": {"amount": "decimal"}})

    assert rows == [
        {
            "id": "blue-kettle",
            "assert": [
                FieldEquals(
                    type="field-equals", path="correspondent", expected="Blue Kettle", kind="text"
                )
            ],
        },
        {
            "id": "corner-market",
            "assert": [
                FieldEquals(type="field-equals", path="date", expected="2026-10-03", kind="date"),
                FieldEquals(
                    type="field-equals",
                    path="total",
                    expected={"amount": "12.50"},
                    kind={"amount": "decimal"},
                ),
            ],
        },
    ]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('{"date": ', "corner-market.expected.json is not valid JSON"),
        ('["2026-10-03"]', "corner-market.expected.json must hold one JSON object"),
        ('{"total.amount": "12.50"}', 'the key "total.amount" has a dot'),
        (
            '{"total": "eleven"}',
            '(?s)corner-market.expected.json: the key "total".*"eleven" is not a decimal',
        ),
    ],
    ids=["not JSON", "not an object", "dotted key", "not a decimal"],
)
def test_an_expected_file_that_cannot_make_a_case_stops_the_load(tmp_path, text, message):
    write_expected(tmp_path / "samples", "corner-market", text)

    with pytest.raises(ConfigError, match=message):
        read_folder_rows(tmp_path / "samples", {"total": "decimal"})


def test_an_expected_value_in_a_folder_is_compared_as_written_and_not_as_a_template(tmp_path):
    write_expected(tmp_path / "samples", "coupon", '{"note": "use code {{PROMO}}"}')
    (tmp_path / "mizan.toml").write_text(
        '[dataset]\nfolder = "samples"\n\n[model]\nprovider = "command"\nargv = ["reader"]\n'
    )

    (case,) = load_project(tmp_path / "mizan.toml").cases
    (assertion,) = case.assertions

    assert assertion.expected == "use code {{PROMO}}"
    assert check(assertion, case, '{"note": "use code {{PROMO}}"}', None).score == 1.0


def test_a_missing_dataset_folder_stops_the_load(tmp_path):
    with pytest.raises(ConfigError, match="samples does not exist"):
        read_folder_rows(tmp_path / "samples", {})


def test_a_dataset_folder_with_a_chat_model_stops_the_load(project_dir):
    rewrite(
        project_dir / "mizan.toml", 'dataset = "cases.toml"', 'dataset = { folder = "samples" }'
    )

    with pytest.raises(ConfigError, match='a dataset folder needs provider = "command"'):
        load_project(project_dir / "mizan.toml")
