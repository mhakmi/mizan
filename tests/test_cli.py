import json
import shutil
from pathlib import Path

import pytest

from mizan import cli
from mizan.cli import main

EXAMPLE = Path(__file__).parent.parent / "examples" / "support-bot"


@pytest.fixture
def example(tmp_path):
    without_report = shutil.ignore_patterns("mizan-report.json")
    return Path(shutil.copytree(EXAMPLE, tmp_path / "support-bot", ignore=without_report))


def run(example, *arguments):
    return main(["run", "--config", str(example / "mizan.toml"), *arguments])


def report(example):
    return json.loads((example / "mizan-report.json").read_text())


def committed_files(example):
    return {
        path.relative_to(example): path.read_bytes()
        for path in example.rglob("*")
        if path.is_file() and path.name != "mizan-report.json"
    }


def rewrite(path, old, new):
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new))


def test_the_example_passes_against_its_committed_baseline(example, capsys):
    assert run(example) == 0

    assert report(example)["verdict"]["status"] == "passed"
    assert "\nPASS\n  Score 0.97. Baseline 0.97. Allowed drop 0.05." in capsys.readouterr().out


def test_removing_the_return_policy_fails_the_gate_on_quality(example, capsys):
    before = committed_files(example)

    exit_code = run(example, "--system-prompt", str(example / "drafts/system-without-policy.md"))

    assert exit_code == 1
    verdict = report(example)["verdict"]
    assert verdict["status"] == "gate_failed"
    assert [failure["check"] for failure in verdict["failures"]] == ["min_score", "max_drop"]
    assert verdict["comparison"]["changes"]["late-return"] == {"baseline": 1.0, "current": 0.0}

    late_return = report(example)["run"]["results"][0]
    assert late_return["output"] == "Sure, you can send them back whenever you like."
    assert committed_files(example) == before

    output = capsys.readouterr().out
    assert "late-return  1.00 to 0.00" in output
    assert "The answer says returns have no deadline." in output


def test_pasting_in_the_handbook_fails_the_gate_on_the_token_budget_alone(example):
    assert run(example) == 0
    score_before = report(example)["verdict"]["score"]
    before = committed_files(example)

    exit_code = run(example, "--system-prompt", str(example / "drafts/system-with-handbook.md"))

    assert exit_code == 1
    verdict = report(example)["verdict"]
    assert [failure["check"] for failure in verdict["failures"]] == ["max_prompt_tokens"]
    assert verdict["score"] == score_before
    assert committed_files(example) == before


def test_a_case_that_cannot_be_scored_exits_2_and_is_never_gated(example, capsys):
    rewrite(
        example / "fake/judge.toml", "<answer>\\nGift cards never expire", "<answer>\\nNo match"
    )

    assert run(example) == 2

    written = report(example)
    assert written["verdict"] == {"status": "incomplete"}
    statuses = {result["case_id"]: result["status"] for result in written["run"]["results"]}
    assert statuses["gift-card"] == "errored"
    assert statuses["late-return"] == "scored"
    output = capsys.readouterr().out
    assert "INCOMPLETE\n  1 case could not be scored, so the gate was not checked." in output
    assert "1 case  The judge call failed. No rule in the fake replies file matches" in output


def test_update_baseline_writes_the_baseline_that_is_committed_with_the_example(example):
    committed = (example / "mizan.baseline.json").read_bytes()
    (example / "mizan.baseline.json").unlink()

    assert run(example, "--update-baseline") == 0

    assert (example / "mizan.baseline.json").read_bytes() == committed
    assert run(example) == 0


def test_update_baseline_replaces_a_baseline_the_run_would_fail_against(example):
    stale = {
        "version": 1,
        "model": "fake:model.toml",
        "judge": "fake:judge.toml",
        "cases": {"shipping-time": 1.0},
    }
    (example / "mizan.baseline.json").write_text(json.dumps(stale))
    assert run(example) == 1

    assert run(example, "--update-baseline") == 0

    assert (example / "mizan.baseline.json").read_bytes() == (
        EXAMPLE / "mizan.baseline.json"
    ).read_bytes()


def test_update_baseline_writes_nothing_when_the_run_fails(example, capsys):
    shutil.copy(example / "drafts/system-without-policy.md", example / "prompts/system.md")
    before = (example / "mizan.baseline.json").read_bytes()

    assert run(example, "--update-baseline") == 1

    assert (example / "mizan.baseline.json").read_bytes() == before
    assert "The baseline was not written" in capsys.readouterr().out


def test_a_plain_run_never_writes_a_baseline_and_says_how_to_create_one(example, capsys):
    (example / "mizan.baseline.json").unlink()

    assert run(example) == 0

    assert not (example / "mizan.baseline.json").exists()
    output = capsys.readouterr().out
    assert "Not compared with a baseline." in output
    assert "Create one by running the same command with --update-baseline." in output


def test_a_draft_prompt_cannot_become_the_baseline(example, capsys):
    before = committed_files(example)

    exit_code = run(
        example,
        "--system-prompt",
        str(example / "drafts/system-with-handbook.md"),
        "--update-baseline",
    )

    assert exit_code == 2
    assert "--system-prompt cannot be used with --update-baseline" in capsys.readouterr().err
    assert committed_files(example) == before


def test_a_config_mistake_exits_2_with_one_line_and_no_traceback(example, capsys):
    rewrite(example / "cases.toml", 'must_include = "30 days"', 'must_inclde = "30 days"')

    assert run(example) == 2

    error = capsys.readouterr().err
    assert error.startswith("mizan: ")
    assert 'Case "late-return"' in error
    assert "must_include" in error
    assert "Traceback" not in error


def test_a_missing_api_key_exits_2_and_names_the_variable(example, capsys):
    rewrite(
        example / "mizan.toml",
        '[model]\nprovider = "fake"\nreplies = "fake/model.toml"',
        '[model]\nprovider = "anthropic"\nname = "claude-opus-5-5"',
    )

    assert run(example) == 2

    assert "ANTHROPIC_API_KEY is not set" in capsys.readouterr().err
    assert not (example / "mizan-report.json").exists()


def test_a_bug_in_mizan_exits_2_so_that_1_only_ever_means_a_failed_gate(
    example, capsys, monkeypatch
):
    def broken(*arguments):
        raise RuntimeError("a bug")

    monkeypatch.setattr(cli, "evaluate", broken)

    assert run(example) == 2
    assert "RuntimeError: a bug" in capsys.readouterr().err


def test_a_missing_config_file_exits_2(tmp_path, capsys):
    assert main(["run", "--config", str(tmp_path / "mizan.toml")]) == 2
    assert "Could not read" in capsys.readouterr().err


RECEIPT_EXAMPLE = Path(__file__).parent.parent / "examples" / "receipt-reader"


@pytest.fixture
def receipts(tmp_path):
    without_report = shutil.ignore_patterns("mizan-report.json")
    return Path(
        shutil.copytree(RECEIPT_EXAMPLE, tmp_path / "receipt-reader", ignore=without_report)
    )


def test_the_receipt_reader_passes_against_its_committed_baseline(receipts, capsys):
    before = committed_files(receipts)

    assert run(receipts) == 0

    assert report(receipts)["verdict"]["status"] == "passed"
    assert "\nPASS\n  Score 1.00. Baseline 1.00. Allowed drop 0.05." in capsys.readouterr().out
    assert committed_files(receipts) == before


def test_the_changed_reader_fails_the_gate_and_names_the_wrong_field(receipts, capsys):
    before = committed_files(receipts)

    exit_code = main(["run", "--config", str(receipts / "drafts" / "mizan.toml")])

    assert exit_code == 1
    verdict = report(receipts)["verdict"]
    assert [failure["check"] for failure in verdict["failures"]] == ["max_drop"]
    assert verdict["comparison"]["changes"]["receipt-1041"] == {"baseline": 1.0, "current": 0.75}
    output = capsys.readouterr().out
    assert "receipt-1041  1.00 to 0.75" in output
    assert (
        'field-equals total    Field total differs. amount: expected "1250.00", read "1".' in output
    )
    assert committed_files(receipts) == before


def test_an_expected_file_the_reader_writes_no_line_for_exits_2(receipts, capsys):
    (receipts / "receipts" / "receipt-1045.expected.json").write_text(
        '{"shop": "Blue Kettle Cafe"}'
    )

    assert run(receipts) == 2

    assert report(receipts)["verdict"] == {"status": "incomplete"}
    output = capsys.readouterr().out
    assert "receipt-1045  error" in output
    assert '1 case  The command wrote no line for "receipt-1045".' in output


def test_update_baseline_writes_the_receipt_baseline_that_is_committed(receipts):
    committed = (receipts / "mizan.baseline.json").read_bytes()
    (receipts / "mizan.baseline.json").unlink()

    assert run(receipts, "--update-baseline") == 0

    assert (receipts / "mizan.baseline.json").read_bytes() == committed
