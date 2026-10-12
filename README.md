# Mizan

Mizan runs a dataset against a prompt, scores each answer, and exits non-zero when quality drops or the prompt goes over its token budget. Put it in a CI job and a bad prompt change fails the pull request.

Mizan (ميزان) is Arabic for a weighing scale.

## What it does

You have a prompt in production and a handful of inputs you know the right answers to. Mizan sends each input through the prompt and scores the answer from 0 to 1 with two kinds of assertion.

- Assertions that need no model: `contains`, `not-contains`, `regex`, and `is-json` with an optional JSON Schema.
- `llm-rubric`, where a second model, the judge, grades the answer against a rubric you write.

The run fails when any of these is true.

- The run score fell below the committed baseline by more than `max_drop`.
- The run score is below `min_score`.
- A single case fell by more than `max_case_drop`.
- A case sent more input tokens than `max_prompt_tokens`.

The baseline is a small JSON file in your repository. You update it on purpose, and the change shows up in the pull request next to the prompt change that caused it.

Mizan calls Anthropic or any OpenAI-compatible endpoint with your own API key. It sends no telemetry and needs no account.

## Quick start

You need [uv](https://docs.astral.sh/uv/). The example project uses a scripted fake model, so it needs no API key and no network.

```sh
git clone https://github.com/mhakmi/mizan
cd mizan
uv run mizan run --config examples/support-bot/mizan.toml
```

```
Mizan   examples/support-bot/mizan.toml
Model   fake:model.toml
Judge   fake:judge.toml

Case                 Score  Baseline  Change  Prompt tokens
late-return           1.00      1.00    0.00            119
damaged-item          1.00      1.00    0.00            113
shipping-time         0.83      0.83    0.00            109
gift-card             1.00      1.00    0.00            112
cancel-order          1.00      1.00    0.00            116
competitor-question   1.00      1.00    0.00            113
Run                   0.97      0.97    0.00    largest 119

PASS
  Score 0.97. Baseline 0.97. Allowed drop 0.05.
  Largest prompt 119 tokens. Budget 400.

Report written to examples/support-bot/mizan-report.json
```

The command exits 0. The example is a support assistant for a made-up shop. Every case, answer, and grade in it is synthetic.

Now try a draft of the system prompt that has lost its return policy. The fake model picks its answer from the prompt text, so two answers get worse.

```sh
uv run mizan run --config examples/support-bot/mizan.toml \
  --system-prompt examples/support-bot/drafts/system-without-policy.md
```

```
Case                 Score  Baseline  Change  Prompt tokens
late-return           0.00      1.00   -1.00             91
damaged-item          0.17      1.00   -0.83             85
shipping-time         0.83      0.83    0.00             81
gift-card             1.00      1.00    0.00             84
cancel-order          1.00      1.00    0.00             88
competitor-question   1.00      1.00    0.00             85
Run                   0.67      0.97   -0.31     largest 91

FAIL
  min_score   The run score is 0.67. The minimum is 0.80.
  max_drop    The run score fell 0.31 below the baseline, from 0.97 to 0.67. The allowed drop is 0.05.

Cases that dropped the most
  late-return  1.00 to 0.00
    0.00  llm-rubric (weight 2)    The answer says returns have no deadline. The reference says 30 days.
    0.00  contains "30 days"       The output does not contain "30 days".
  damaged-item  1.00 to 0.17
    0.25  llm-rubric (weight 2)    The answer offers a refund and does not mention a replacement.
    0.00  contains "replace"       The output does not contain "replace".
```

The command exits 1. The numbers are rounded for display, which is why 0.97 minus 0.67 prints as 0.31.

A third draft pastes a handbook into the prompt. The answers stay the same and the run still fails, because every case is over the token budget.

```sh
uv run mizan run --config examples/support-bot/mizan.toml \
  --system-prompt examples/support-bot/drafts/system-with-handbook.md
```

```
Case                 Score  Baseline  Change  Prompt tokens
late-return           1.00      1.00    0.00            477  over
damaged-item          1.00      1.00    0.00            471  over
shipping-time         0.83      0.83    0.00            467  over
gift-card             1.00      1.00    0.00            470  over
cancel-order          1.00      1.00    0.00            474  over
competitor-question   1.00      1.00    0.00            471  over
Run                   0.97      0.97    0.00    largest 477

FAIL
  max_prompt_tokens   6 cases sent more than 400 prompt tokens. The largest is late-return with 477.
```

The command exits 1.

### Set it up for your own prompt

1. Copy `examples/support-bot/` into your repository. Delete `fake/`, `drafts/`, and `mizan.baseline.json`.
2. Put your prompt in `prompts/system.md` and `prompts/user.md`.
3. Replace the cases in `cases.toml` with your own inputs.
4. Point `[model]` and `[judge]` in `mizan.toml` at real models, and export the API key.
5. Run `mizan run`. Read the table, and fix your cases and assertions until the scores match your own reading of the answers.
6. Run `mizan run --update-baseline` and commit `mizan.baseline.json`.
7. Copy `prompt-check.yml` to `.github/workflows/` and add the API key as a repository secret.

## How a run works

```mermaid
flowchart TD
    config["mizan.toml"] --> load
    prompts["prompts/*.md"] --> load
    dataset["cases.toml"] --> load
    load["Read and check the files.<br/>Fill every template."] --> call
    call["Send each case to the model"] --> score
    score["Score each assertion from 0 to 1.<br/>The judge grades llm-rubric assertions."] --> complete{"Was every<br/>case scored?"}
    complete -- "no" --> exit2["Exit 2<br/>could not measure"]
    complete -- "yes" --> gate{"Gate"}
    baseline["mizan.baseline.json"] --> gate
    gate -- "every check holds" --> exit0["Exit 0<br/>pass"]
    gate -- "a check fails" --> exit1["Exit 1<br/>quality drop or over budget"]
    score --> report["Terminal table and<br/>mizan-report.json"]
    gate --> report
```

A case score is the weighted mean of its assertion scores. The run score is the plain mean of the case scores. Only the gate decides pass or fail. An assertion has a score and no threshold of its own.

A failed model call is not a bad answer. A case that cannot be scored gets no score, the gate is not checked, and the run exits 2. A provider outage therefore never reads as a quality drop. The same holds for a judge reply that Mizan cannot read.

Each module owns one part of the run.

| Module | Owns |
|---|---|
| `types.py` | The data shapes. Read this file first. |
| `files.py` | Reading your files, writing the baseline and the report |
| `cases.py` | Turning a dataset row into a case with its templates filled |
| `providers.py` | The Anthropic call, the OpenAI-compatible call, and the fake model |
| `runner.py` | Running the cases |
| `assertions.py` | Scoring one answer, the judge prompt, reading the judge reply |
| `gate.py` | The run score, the comparison with the baseline, what fails |
| `report.py` | The terminal table |
| `cli.py` | Arguments and exit codes |

## Config reference

A project is a `mizan.toml`, the files it names, and a baseline that Mizan writes. Paths in `mizan.toml` are relative to that file. You write TOML and Markdown. Mizan writes JSON.

### Command

```
mizan run [--config PATH] [--system-prompt PATH] [--update-baseline]
```

| Flag | Meaning |
|---|---|
| `--config PATH` | The config file. Defaults to `mizan.toml` in the current directory. |
| `--system-prompt PATH` | Use this file as the system prompt for this run, to try a draft against the committed baseline. The path is relative to the current directory. |
| `--update-baseline` | Run the dataset and save the result as the new baseline. The run is checked against `min_score` and `max_prompt_tokens` only, and the baseline is written only if it passes. Cannot be combined with `--system-prompt`. |

| Exit code | Meaning |
|---|---|
| 0 | The run passed. |
| 1 | The gate failed. The score dropped, or a case went over the token budget. |
| 2 | Mizan could not finish. A file is wrong, a key is missing, or a case could not be scored. |

### `mizan.toml`

| Key | Default | Meaning |
|---|---|---|
| `dataset` | required | The cases file. |
| `baseline` | `mizan.baseline.json` | Where the baseline is read and written. |
| `report` | `mizan-report.json` | Where the report is written on every run. Add it to `.gitignore`. |
| `concurrency` | `4` | How many cases run at the same time. |

`[prompt]`

| Key | Default | Meaning |
|---|---|---|
| `system` | none | A file with the system prompt. |
| `user` | required | A file with the user message. |

Both files are templates. `{{name}}` is replaced with the value of `name` from the case. A name the case does not define stops the run before any model call. Other braces are left alone, so a prompt can contain JSON.

`[model]` is the model your feature runs on. `[judge]` is the model that grades `llm-rubric` assertions. It takes the same keys, has no default, and is required when any `llm-rubric` assertion exists.

| Key | Default | Meaning |
|---|---|---|
| `provider` | required | `anthropic`, `openai`, or `fake`. |
| `name` | required | The model id, sent to the provider as written. |
| `base_url` | the provider's own API | Set it for any OpenAI-compatible server, such as a local one. |
| `api_key_env` | `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` | The environment variable that holds the key. |
| `params` | empty | A table merged into the request body unchanged. Put here what your production call sends. |
| `replies` | required for `fake` | A file of scripted answers. The `fake` provider takes this key and no other. |

Mizan sets no temperature, because some models reject the field. Where your judge model accepts it, set `temperature = 0` under `[judge.params]`. For Anthropic, `max_tokens` defaults to 4096. Set it under `params` to change it.

`[gate]` holds every check that can fail a run.

| Key | Default | Meaning |
|---|---|---|
| `min_score` | none | Fail when the run score is below this. Works before a baseline exists. |
| `max_drop` | `0.05` | Fail when the run score falls this far below the baseline. Only the cases that the baseline and the dataset share are compared, so a new case does not read as a drop. |
| `max_case_drop` | none | Fail when any single case falls this far below its own baseline score. |
| `max_prompt_tokens` | none | Fail when any case sends more input tokens than this. |

With no baseline file, a run checks `min_score` and `max_prompt_tokens` only, and says so.

`[[assert]]` tables in `mizan.toml` apply to every case. A case can add its own. Any string in an assertion can use `{{name}}`, so one rubric can serve the whole dataset with a reference answer from each case.

| Type | Keys | Score |
|---|---|---|
| `contains` | `value`, `ignore_case` | 1 when the answer contains the value, else 0 |
| `not-contains` | `value`, `ignore_case` | 1 when the answer does not contain the value, else 0 |
| `regex` | `pattern` | 1 when the pattern matches anywhere in the answer, else 0 |
| `is-json` | `json_schema` (optional) | 1 when the answer parses as JSON and fits the schema, else 0 |
| `llm-rubric` | `rubric` | The judge grades from 1 to 5, which maps to 0, 0.25, 0.5, 0.75, and 1 |

Every type also takes `weight`, which defaults to 1.

### `cases.toml`

Each `[[case]]` table has an `id`, which must be unique because the baseline is keyed on it. Every other key is a string that templates can use. An optional `assert` list adds assertions to that case.

```toml
[[case]]
id = "late-return"
question = "I bought hiking boots 40 days ago and they don't fit. Can I send them back?"
reference = "Returns are accepted within 30 days of delivery, so this order can no longer be returned."
must_include = "30 days"

[[case]]
id = "competitor-question"
question = "Is your tent cheaper than the ones other shops sell?"
reference = "The assistant says the policies do not cover price comparisons and offers a person."
must_include = "person"
assert = [
  { type = "not-contains", value = "cheaper than", ignore_case = true },
]
```

### `mizan.baseline.json`

Mizan writes this file when you run `mizan run --update-baseline`. Commit it. It holds the model and judge it was recorded with and one score per case. It has no total and no timestamp, so writing the same result twice gives the same bytes.

### `mizan-report.json`

Mizan writes this file on every run. It holds the verdict, each gate failure by name, the comparison with the baseline, and for every case the full answer and each assertion's score and reason.

## Use it in CI

[`examples/support-bot/prompt-check.yml`](examples/support-bot/prompt-check.yml) is a GitHub Actions workflow to copy. It runs Mizan on every pull request that touches the prompt, the cases, the config, or the baseline.

```yaml
- run: uvx --from "git+https://github.com/mhakmi/mizan@v0.1.0" mizan run
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

The job fails when Mizan exits non-zero. The workflow uploads the report as an artifact, so you can read every answer and every judge reason from a failed run without running it again. Pin a release tag, because a floating version can change the result of the check.

## Limits

- Each case runs once. A judge model does not give the same grade every time, and `max_drop` is the only allowance for that. Run an unchanged prompt a few times and set `max_drop` above the spread you see.
- The token budget is the input token count the provider reports for a case. It covers the system prompt and the user message. The count arrives with the answer, so you pay for an over-budget call before the run fails.
- The config and the baseline are files in the pull request. Deleting the baseline or raising `max_drop` weakens the check. Review those changes the way you review the prompt.
- One case that cannot be scored turns the whole run into exit 2. Mizan cannot rerun only the cases that errored.
- The tests cover the Anthropic and OpenAI-compatible calls with recorded reply shapes. They do not call the real services.

## When to use something else

Mizan is not a new idea. Larger tools cover far more ground, and at least two small ones use the same check of a baseline with a margin. I looked at the tools below in October 2026.

| Tool | Pick it when |
|---|---|
| [promptfoo](https://github.com/promptfoo/promptfoo) | You want many providers, red-team tests, a web viewer, and a tool your next hire may already know. `promptfoo eval` exits 100 when a test fails or the pass rate is under a threshold you set. It is now part of OpenAI and remains MIT licensed. |
| [DeepEval](https://github.com/confident-ai/deepeval) | Your tests already live in pytest and you want ready-made metrics for retrieval and agents. |
| [Inspect](https://inspect.aisi.org.uk/) | You evaluate agents, tool use, or sandboxed tasks, or you need repeated runs. |
| [Pydantic Evals](https://github.com/pydantic/pydantic-ai) | Your app is built on Pydantic AI and you are happy to write the pass or fail check in Python. |
| [llmci](https://github.com/llmci-cli/llmci) | You want the same kind of check as Mizan with repeated samples and a significance test, which handles judge noise better than a fixed margin. It keeps baselines under `.llmci/baselines/` and has token metrics. |
| [evalgate](https://github.com/Surge77/evalgate) | You want a floor and a maximum drop against a promoted baseline file, with a separate command to promote a run. |

If you already run promptfoo, a short script over its JSON output can compare a run with a committed copy. That gets you most of what Mizan does.

What Mizan offers next to these is small.

- The default check is a baseline file in git and a margin. You do not write the comparison yourself.
- The prompt's token count is a check of its own. A prompt that doubles in size fails the pull request even when the answers hold.
- It is about 1,400 lines of Python with two runtime dependencies, `pydantic` and `jsonschema`. You can read all of it before you let it block a merge.
- It sends no telemetry and needs no account.

## Development

```sh
uv sync
uv run pytest
uv run ruff check
```

The tests need no network and no API key.

## License

MIT
