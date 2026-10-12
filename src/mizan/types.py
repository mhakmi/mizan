import re
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal

import jsonschema.validators
from jsonschema.exceptions import SchemaError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, computed_field, field_validator


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


Assertion = Annotated[
    Contains | NotContains | Regex | IsJson | LlmRubric,
    Field(discriminator="type"),
]


class Case(Shape):
    id: str
    prompt: Prompt
    assertions: Annotated[tuple[Assertion, ...], Field(min_length=1)]


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


ModelSettings = Annotated[
    AnthropicModel | OpenAIModel | FakeModel,
    Field(discriminator="provider"),
]


class PromptFiles(Shape):
    system: Path | None = None
    user: Path


class Gate(Shape):
    min_score: Score | None = None
    max_drop: Score = 0.05
    max_case_drop: Score | None = None
    max_prompt_tokens: Annotated[int, Field(gt=0)] | None = None


class Config(Shape):
    dataset: Path
    baseline: Path = Path("mizan.baseline.json")
    report: Path = Path("mizan-report.json")
    concurrency: Annotated[int, Field(ge=1)] = 4
    prompt: PromptFiles
    model: ModelSettings
    judge: ModelSettings | None = None
    gate: Gate = Gate()
    assertion_templates: tuple[dict[str, Any], ...] = Field(default=(), alias="assert")


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
