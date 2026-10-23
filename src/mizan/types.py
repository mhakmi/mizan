import re
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal, Self, assert_never

import jsonschema.validators
from jsonschema.exceptions import SchemaError
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    computed_field,
    field_validator,
    model_validator,
)

from mizan.fields import Kind, check_expected, split_path


class MizanError(Exception):
    """Base class for every error Mizan reports without a traceback."""


class ConfigError(MizanError):
    """A file the user wrote is wrong, or a key is missing from the environment.

    Raised before any model call. The command line prints the message and exits 2.
    """


class Unscorable(MizanError):
    """One case could not be scored. The runner turns it into an Errored result."""


class ProviderError(Unscorable):
    """A model call failed after its retries, or the model declined to answer."""


class JudgeReplyError(Unscorable):
    """The judge answered, but not with the JSON object the judge prompt asks for."""


def explain(error: ValidationError) -> str:
    lines = []
    for problem in error.errors():
        location = ".".join(str(part) for part in problem["loc"])
        if location:
            lines.append(f"  {location}: {problem['msg']}")
        else:
            lines.append(f"  {problem['msg']}")
    return "\n".join(lines)


class Shape(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


Score = Annotated[float, Field(ge=0.0, le=1.0)]
# Scores are rounded where they are computed, so a stored baseline and a
# fresh run of the same outputs compare as exactly equal.
SCORE_DECIMALS = 4
Weight = Annotated[float, Field(gt=0.0)]


class Prompt(Shape):
    system: str | None
    user: str


class Reply(Shape):
    text: str
    prompt_tokens: int


Complete = Callable[[Prompt], Reply]


class Contains(Shape):
    type: Literal["contains"]
    value: str
    ignore_case: bool = False
    weight: Weight = 1.0


class NotContains(Shape):
    type: Literal["not-contains"]
    value: str
    ignore_case: bool = False
    weight: Weight = 1.0


class Regex(Shape):
    type: Literal["regex"]
    pattern: str
    weight: Weight = 1.0

    @field_validator("pattern")
    @classmethod
    def pattern_compiles(cls, pattern: str) -> str:
        try:
            re.compile(pattern)
        except re.error as error:
            raise ValueError(f"not a valid regular expression ({error})") from error
        return pattern


class IsJson(Shape):
    type: Literal["is-json"]
    json_schema: dict[str, Any] | None = None
    weight: Weight = 1.0

    @field_validator("json_schema")
    @classmethod
    def schema_is_valid(cls, json_schema: dict[str, Any] | None) -> dict[str, Any] | None:
        if json_schema is not None:
            try:
                jsonschema.validators.validator_for(json_schema).check_schema(json_schema)
            except SchemaError as error:
                raise ValueError(f"not a valid JSON Schema ({error.message})") from error
        return json_schema


class LlmRubric(Shape):
    type: Literal["llm-rubric"]
    rubric: str
    weight: Weight = 1.0


class FieldEquals(Shape):
    type: Literal["field-equals"]
    path: str
    # TOML has no null, so a missing expected means the field must be null.
    expected: JsonValue = None
    kind: Kind = "text"
    weight: Weight = 1.0

    @field_validator("path")
    @classmethod
    def path_splits(cls, path: str) -> str:
        split_path(path)
        return path

    @model_validator(mode="after")
    def expected_fits_kind(self) -> Self:
        check_expected(self.kind, self.expected)
        return self


Assertion = Annotated[
    Contains | NotContains | Regex | IsJson | LlmRubric | FieldEquals,
    Field(discriminator="type"),
]


class Case(Shape):
    id: str
    # None exactly when the project runs a command. Config.parts_fit_the_model
    # keeps the pairing, so code that needs a prompt calls require_prompt.
    prompt: Prompt | None
    assertions: Annotated[tuple[Assertion, ...], Field(min_length=1)]

    def require_prompt(self) -> Prompt:
        assert self.prompt is not None, "only a chat project asks for a prompt"
        return self.prompt


Answer = Callable[[Case], Reply]


class AssertionScore(Shape):
    assertion: Assertion
    score: Score
    reason: str


class Scored(Shape):
    status: Literal["scored"] = "scored"
    case_id: str
    output: str
    prompt_tokens: int
    checks: Annotated[tuple[AssertionScore, ...], Field(min_length=1)]

    @computed_field
    @property
    def score(self) -> float:
        weighted_sum = sum(check.assertion.weight * check.score for check in self.checks)
        total_weight = sum(check.assertion.weight for check in self.checks)
        return round(weighted_sum / total_weight, SCORE_DECIMALS)


class Errored(Shape):
    status: Literal["errored"] = "errored"
    case_id: str
    reason: str


CaseResult = Annotated[Scored | Errored, Field(discriminator="status")]


class RunSummary(Shape):
    model: str
    judge: str | None
    results: tuple[CaseResult, ...]

    def scored(self) -> tuple[Scored, ...]:
        return tuple(result for result in self.results if isinstance(result, Scored))

    def errored(self) -> tuple[Errored, ...]:
        return tuple(result for result in self.results if isinstance(result, Errored))


class Baseline(Shape):
    version: Literal[1] = 1
    model: str
    judge: str | None
    cases: dict[str, Score]


# urllib would also open file: and ftp: addresses.
HttpUrl = Annotated[str, Field(pattern=r"^https?://")]


class AnthropicModel(Shape):
    provider: Literal["anthropic"]
    name: str
    base_url: HttpUrl = "https://api.anthropic.com"
    api_key_env: str = "ANTHROPIC_API_KEY"
    params: dict[str, Any] = {}


class OpenAIModel(Shape):
    provider: Literal["openai"]
    name: str
    base_url: HttpUrl = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    params: dict[str, Any] = {}


class FakeModel(Shape):
    provider: Literal["fake"]
    replies: Path


class CommandModel(Shape):
    provider: Literal["command"]
    argv: Annotated[tuple[str, ...], Field(min_length=1)]
    env: dict[str, str] = {}
    id_key: str = "id"
    output_key: str = "output"
    error_key: str = "error"
    tokens_key: str = "prompt_tokens"


ChatModels = AnthropicModel | OpenAIModel | FakeModel
JudgeSettings = Annotated[ChatModels, Field(discriminator="provider")]
ModelSettings = Annotated[ChatModels | CommandModel, Field(discriminator="provider")]


class FolderDataset(Shape):
    folder: Path
    kinds: dict[str, Kind] = {}


class PromptFiles(Shape):
    system: Path | None = None
    user: Path


class Gate(Shape):
    min_score: Score | None = None
    max_drop: Score = 0.05
    max_case_drop: Score | None = None
    max_prompt_tokens: Annotated[int, Field(gt=0)] | None = None


class Config(Shape):
    dataset: Path | FolderDataset
    baseline: Path = Path("mizan.baseline.json")
    report: Path = Path("mizan-report.json")
    concurrency: Annotated[int, Field(ge=1)] = 4
    prompt: PromptFiles | None = None
    model: ModelSettings
    judge: JudgeSettings | None = None
    gate: Gate = Gate()
    assertion_templates: tuple[dict[str, Any], ...] = Field(default=(), alias="assert")

    @model_validator(mode="after")
    def parts_fit_the_model(self) -> Self:
        match self.model:
            case CommandModel():
                if self.prompt is not None:
                    raise ValueError("a command project has no [prompt] table")
                if self.judge is not None:
                    raise ValueError(
                        "a command project cannot use llm-rubric, so [judge] has no use"
                    )
            case AnthropicModel() | OpenAIModel() | FakeModel():
                if self.prompt is None:
                    raise ValueError(f'provider = "{self.model.provider}" needs a [prompt] table')
                if isinstance(self.dataset, FolderDataset):
                    # Pydantic reports a ValueError as a mistake in the file, and a TypeError as a bug.
                    raise ValueError('a dataset folder needs provider = "command"')  # noqa: TRY004
            case _:
                assert_never(self.model)
        return self


class CaseChange(Shape):
    baseline: Score
    current: Score


class Comparison(Shape):
    changes: Annotated[dict[str, CaseChange], Field(min_length=1)]
    new_cases: tuple[str, ...]
    removed_cases: tuple[str, ...]


class GateFailure(Shape):
    check: Literal["min_score", "max_drop", "max_case_drop", "max_prompt_tokens"]
    message: str


class Passed(Shape):
    status: Literal["passed"] = "passed"
    score: Score
    comparison: Comparison | None
    notes: tuple[str, ...]


class GateFailed(Shape):
    status: Literal["gate_failed"] = "gate_failed"
    score: Score
    failures: Annotated[tuple[GateFailure, ...], Field(min_length=1)]
    comparison: Comparison | None
    notes: tuple[str, ...]


class Incomplete(Shape):
    status: Literal["incomplete"] = "incomplete"


Verdict = Annotated[Passed | GateFailed | Incomplete, Field(discriminator="status")]


class Report(Shape):
    version: Literal[1] = 1
    verdict: Verdict
    run: RunSummary
