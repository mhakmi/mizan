import json
import sys

import pytest

from mizan.command import answers_from_command
from mizan.files import load_project
from mizan.providers import label, prepare_answers
from mizan.types import Case, CommandModel, ConfigError, IsJson, ProviderError, Reply

DAFTAR_KEYS = {"id_key": "document", "output_key": "values", "tokens_key": "input_tokens"}


def line(**keys):
    return json.dumps(keys)


def child_program(tmp_path, lines, exit_status=0, before=""):
    (tmp_path / "child.py").write_text(
        f"import os, sys\n{before}\nfor line in {lines!r}:\n    print(line)\nsys.exit({exit_status})\n"
    )
    return (sys.executable, "child.py")


def outcomes(tmp_path, lines, case_ids=("blue-kettle", "corner-market"), exit_status=0, **keys):
    settings = CommandModel(
        provider="command", argv=child_program(tmp_path, lines, exit_status), **keys
    )
    cases = [
        Case(id=case_id, prompt=None, assertions=(IsJson(type="is-json"),)) for case_id in case_ids
    ]
    answer = answers_from_command(settings, tmp_path, None, cases)

    results = {}
    for case in cases:
        try:
            results[case.id] = answer(case)
        except ProviderError as error:
            results[case.id] = str(error)
    return results


def test_each_case_gets_the_output_and_tokens_of_its_own_line(tmp_path):
    results = outcomes(
        tmp_path,
        [
            line(document="corner-market", values={"total": "12.5"}, error=None, input_tokens=11),
            line(document="blue-kettle", values="Total 1,250.00", input_tokens=0),
        ],
        **DAFTAR_KEYS,
    )

    assert results == {
        "blue-kettle": Reply(text="Total 1,250.00", prompt_tokens=0),
        "corner-market": Reply(text='{"total":"12.5"}', prompt_tokens=11),
    }


def test_a_line_with_an_error_is_an_errored_case_and_its_output_is_not_read(tmp_path):
    results = outcomes(
        tmp_path,
        [
            line(id="blue-kettle", output=None, error="bad_key", prompt_tokens=1500),
            line(id="corner-market", output="{}", error={"kind": "timeout"}, prompt_tokens=0),
        ],
    )

    assert results == {
        "blue-kettle": 'The command reported an error for "blue-kettle": bad_key.',
        "corner-market": 'The command reported an error for "corner-market": {"kind":"timeout"}.',
    }


def test_a_case_with_no_line_is_errored_and_names_the_exit_status(tmp_path):
    results = outcomes(
        tmp_path, [line(id="blue-kettle", output="{}", prompt_tokens=3)], exit_status=2
    )

    assert results["blue-kettle"] == Reply(text="{}", prompt_tokens=3)
    assert results["corner-market"] == (
        'The command wrote no line for "corner-market". It exited with status 2.'
    )


def test_a_line_for_no_case_is_ignored(tmp_path):
    results = outcomes(
        tmp_path,
        [
            line(id="scratch", output="{}", prompt_tokens=1),
            line(id="blue-kettle", output="{}", prompt_tokens=1),
        ],
        case_ids=("blue-kettle",),
    )

    assert results == {"blue-kettle": Reply(text="{}", prompt_tokens=1)}


def test_two_lines_for_one_case_is_an_error_and_neither_is_picked(tmp_path):
    results = outcomes(
        tmp_path,
        [
            line(id="blue-kettle", output="first", prompt_tokens=1),
            line(id="blue-kettle", output="second", prompt_tokens=1),
        ],
        case_ids=("blue-kettle",),
    )

    assert results == {"blue-kettle": 'The command wrote 2 lines for "blue-kettle".'}


def test_unusable_lines_are_skipped_and_quoted_only_for_a_case_with_no_line(tmp_path):
    unusable = [
        "Reading receipts" + " ." * 60,
        "[1, 2]",
        line(id=7, output="{}", prompt_tokens=1),
    ]
    results = outcomes(
        tmp_path, [*unusable, line(id="blue-kettle", output="{}", prompt_tokens=1)], exit_status=1
    )

    assert results["blue-kettle"] == Reply(text="{}", prompt_tokens=1)
    assert results["corner-market"] == (
        'The command wrote no line for "corner-market". '
        "It also wrote 3 lines that Mizan could not use. "
        f'The first is "{unusable[0][:80]}". It exited with status 1.'
    )


def test_unusable_lines_are_not_mentioned_when_every_case_has_a_line(tmp_path):
    results = outcomes(
        tmp_path,
        ["warming up", line(id="blue-kettle", output="{}", prompt_tokens=1)],
        case_ids=("blue-kettle",),
    )

    assert results == {"blue-kettle": Reply(text="{}", prompt_tokens=1)}


def test_a_wrong_id_key_is_explained_once_with_the_keys_the_lines_have(tmp_path):
    lines = [
        line(document="blue-kettle", pages=1, values={}, input_tokens=1),
        line(document="corner-market", pages=1, values={}, input_tokens=1),
    ]

    results = outcomes(tmp_path, lines, output_key="values", tokens_key="input_tokens")

    reason = (
        'No line names a case in its "id" key. The command wrote 2 lines. '
        "The first object has the keys: document, pages, values, input_tokens."
    )
    assert results == {"blue-kettle": reason, "corner-market": reason}


@pytest.mark.parametrize("output", ["absent", None])
def test_a_line_with_no_output_and_no_error_is_an_errored_case(tmp_path, output):
    keys = {"id": "blue-kettle", "prompt_tokens": 1}
    if output is None:
        keys["output"] = None

    results = outcomes(tmp_path, [line(**keys)], case_ids=("blue-kettle",))

    assert results == {"blue-kettle": 'The line for "blue-kettle" has no "output".'}


@pytest.mark.parametrize("tokens", ["absent", None, -1, 2.5, "12", True])
def test_a_line_without_a_whole_token_count_is_an_errored_case(tmp_path, tokens):
    keys = {"document": "blue-kettle", "values": {}}
    if tokens != "absent":
        keys["input_tokens"] = tokens

    results = outcomes(tmp_path, [line(**keys)], case_ids=("blue-kettle",), **DAFTAR_KEYS)

    assert results == {
        "blue-kettle": 'The line for "blue-kettle" has no whole number "input_tokens".'
    }


def test_a_non_zero_exit_with_every_line_written_is_still_scored(tmp_path):
    results = outcomes(
        tmp_path,
        [
            line(id="blue-kettle", output="{}", prompt_tokens=1),
            line(id="corner-market", output=None, error="bad_key", prompt_tokens=1),
        ],
        exit_status=2,
    )

    assert results == {
        "blue-kettle": Reply(text="{}", prompt_tokens=1),
        "corner-market": 'The command reported an error for "corner-market": bad_key.',
    }


def test_a_program_that_cannot_start_stops_the_run(tmp_path):
    settings = CommandModel(provider="command", argv=(str(tmp_path / "no-such-program"),))

    with pytest.raises(ConfigError, match='The command ".*no-such-program" could not start'):
        answers_from_command(settings, tmp_path, None, [])


def test_the_program_writes_its_stderr_through_and_it_stays_out_of_the_reasons(tmp_path, capfd):
    settings = CommandModel(
        provider="command",
        argv=child_program(tmp_path, [], 1, before="print('reading receipts', file=sys.stderr)"),
    )
    case = Case(id="blue-kettle", prompt=None, assertions=(IsJson(type="is-json"),))
    answer = answers_from_command(settings, tmp_path, None, [case])

    with pytest.raises(ProviderError) as error:
        answer(case)
    assert "reading receipts" in capfd.readouterr().err
    assert (
        str(error.value) == 'The command wrote no line for "blue-kettle". It exited with status 1.'
    )


def test_the_env_table_is_added_to_the_environment_mizan_runs_in(tmp_path, monkeypatch):
    monkeypatch.setenv("MIZAN_TEST_INHERITED", "kept")
    program = 'print(json.dumps({"id": "blue-kettle", "output": os.environ["MIZAN_TEST_INHERITED"] + " " + os.environ["DAFTAR_MODEL_PROVIDER"], "prompt_tokens": 0}))'
    (tmp_path / "child.py").write_text(f"import json, os\n{program}\n")
    settings = CommandModel(
        provider="command",
        argv=(sys.executable, "child.py"),
        env={"DAFTAR_MODEL_PROVIDER": "fake"},
    )
    case = Case(id="blue-kettle", prompt=None, assertions=(IsJson(type="is-json"),))

    reply = answers_from_command(settings, tmp_path, None, [case])(case)

    assert reply.text == "kept fake"


COMMAND_CONFIG = """\
[dataset]
folder = "samples"
[dataset.kinds]
total = "decimal"

[model]
provider = "command"
argv = ["{python}", "child.py", "{{{{folder}}}}"]
"""


@pytest.fixture
def command_project(tmp_path):
    (tmp_path / "samples").mkdir()
    (tmp_path / "samples" / "blue-kettle.expected.json").write_text('{"total": "1250.00"}')
    (tmp_path / "mizan.toml").write_text(COMMAND_CONFIG.format(python=sys.executable))
    program = (
        "folder = sys.argv[1]\n"
        "print(json.dumps({'id': 'blue-kettle', 'output': {'folder': folder, 'cwd': os.getcwd()}, 'prompt_tokens': 4}))\n"
    )
    (tmp_path / "child.py").write_text(f"import json, os, sys\n{program}")
    return tmp_path


def test_a_command_project_runs_in_the_config_folder_and_gets_the_absolute_dataset_folder(
    command_project, monkeypatch
):
    monkeypatch.chdir(command_project.parent)
    project = load_project(command_project.relative_to(command_project.parent) / "mizan.toml")

    reply = prepare_answers(project)(project.cases[0])

    assert json.loads(reply.text) == {
        "folder": str((command_project / "samples").resolve()),
        "cwd": str(command_project.resolve()),
    }
    assert project.cases[0].prompt is None


def test_the_label_names_the_program_and_its_arguments_as_written():
    settings = CommandModel(provider="command", argv=("python", "reader.py", "{{folder}}"))

    assert label(settings) == "command:python reader.py '{{folder}}'"


def test_folder_in_the_arguments_needs_a_dataset_folder(tmp_path):
    settings = CommandModel(provider="command", argv=("reader", "{{folder}}"))

    with pytest.raises(ConfigError, match=r"argv: .*Only a dataset folder defines \{\{folder\}\}"):
        answers_from_command(settings, tmp_path, None, [])


@pytest.mark.parametrize(
    ("addition", "message"),
    [
        ('[prompt]\nuser = "user.md"\n', r"a command project has no \[prompt\] table"),
        (
            '[judge]\nprovider = "fake"\nreplies = "judge.toml"\n',
            r"cannot use llm-rubric, so \[judge\]",
        ),
        ('[judge]\nprovider = "command"\nargv = ["judge"]\n', "(?s)judge.*command"),
        (
            '[[assert]]\ntype = "llm-rubric"\nrubric = "Reads well."\n',
            "no prompt to show the judge",
        ),
    ],
    ids=["prompt", "judge", "command as judge", "llm-rubric"],
)
def test_a_command_project_refuses_the_parts_that_need_a_prompt(command_project, addition, message):
    config = command_project / "mizan.toml"
    config.write_text(config.read_text() + addition)

    with pytest.raises(ConfigError, match=message):
        load_project(config)


def test_a_draft_system_prompt_is_refused_for_a_command_project(command_project, tmp_path):
    draft = tmp_path / "draft.md"
    draft.write_text("Be brief.")

    with pytest.raises(ConfigError, match=r"--system-prompt needs a \[prompt\] table"):
        load_project(command_project / "mizan.toml", system_prompt_override=draft)


def test_an_empty_dataset_folder_has_no_cases(command_project):
    (command_project / "samples" / "blue-kettle.expected.json").unlink()

    with pytest.raises(ConfigError, match="The dataset has no cases."):
        load_project(command_project / "mizan.toml")
