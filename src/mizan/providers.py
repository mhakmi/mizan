import http.client
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, assert_never

from pydantic import BaseModel, Field, ValidationError

from mizan.files import Project, read_toml
from mizan.types import (
    Answer,
    AnthropicModel,
    Case,
    ChatModels,
    Complete,
    ConfigError,
    FakeModel,
    ModelSettings,
    OpenAIModel,
    Prompt,
    ProviderError,
    Reply,
    Shape,
    explain,
)

PostJson = Callable[[str, Mapping[str, str], Mapping[str, Any]], dict[str, Any]]

ATTEMPTS = 3
TIMEOUT_SECONDS = 120
# 529 is Anthropic's "overloaded" status.
RETRYABLE_STATUSES = (429, 500, 502, 503, 504, 529)

ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_DEFAULT_MAX_TOKENS = 4096


def prepare_answers(project: Project) -> Answer:
    """Return the function that gives the reply for one case.

    Raises ConfigError, before any case runs, when the model cannot be reached
    at all, such as when its API key is missing.
    """
    settings = project.config.model
    match settings:
        case AnthropicModel() | OpenAIModel() | FakeModel():
            complete = connect(settings, project.config_dir)

            def answer(case: Case) -> Reply:
                try:
                    return complete(case.require_prompt())
                except ProviderError as error:
                    raise ProviderError(f"The model call failed. {error}") from error

            return answer
        case _:
            assert_never(settings)


def connect(settings: ChatModels, config_dir: Path) -> Complete:
    """Return the function that calls the configured model.

    Raises ConfigError when the API key is not in the environment, so a missing
    key stops the run before the first call.
    """
    match settings:
        case AnthropicModel():
            return anthropic_model(settings, read_api_key(settings.api_key_env))
        case OpenAIModel():
            return openai_model(settings, read_api_key(settings.api_key_env))
        case FakeModel():
            return fake_model(load_fake_rules(config_dir / settings.replies))
        case _:
            assert_never(settings)


def label(settings: ModelSettings) -> str:
    match settings:
        case AnthropicModel():
            return f"anthropic:{settings.name}"
        case OpenAIModel():
            return f"openai:{settings.name}"
        case FakeModel():
            return f"fake:{settings.replies.name}"
        case _:
            assert_never(settings)


def read_api_key(variable: str) -> str:
    api_key = os.environ.get(variable, "")
    if api_key == "":
        raise ConfigError(f"The environment variable {variable} is not set.")
    return api_key


def post_json(url: str, headers: Mapping[str, str], body: Mapping[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"content-type": "application/json", **headers},
        method="POST",
    )
    for attempt in range(1, ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            detail = " ".join(error.read().decode("utf-8", errors="replace").split())[:200]
            cause = f"HTTP {error.code}: {detail}"
            if error.code not in RETRYABLE_STATUSES:
                raise ProviderError(cause) from error
        except (OSError, http.client.HTTPException) as error:
            cause = f"The request did not get through ({error})"
        except json.JSONDecodeError as error:
            raise ProviderError("The reply was not JSON.") from error

        if attempt < ATTEMPTS:
            time.sleep(attempt)

    raise ProviderError(f"{cause}, after {ATTEMPTS} attempts.")


class AnthropicBlock(BaseModel):
    type: str
    text: str = ""


class AnthropicUsage(BaseModel):
    input_tokens: int
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None


class AnthropicMessage(BaseModel):
    content: list[AnthropicBlock]
    stop_reason: str | None
    usage: AnthropicUsage


def anthropic_model(settings: AnthropicModel, api_key: str, post: PostJson = post_json) -> Complete:
    url = settings.base_url.rstrip("/") + "/v1/messages"
    headers = {"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION}

    def complete(prompt: Prompt) -> Reply:
        body: dict[str, Any] = {
            "model": settings.name,
            "max_tokens": ANTHROPIC_DEFAULT_MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt.user}],
        }
        if prompt.system is not None:
            body["system"] = prompt.system
        body.update(settings.params)

        try:
            message = AnthropicMessage.model_validate(post(url, headers, body))
        except ValidationError as error:
            raise ProviderError(f"The reply was not a message.\n{explain(error)}") from error
        if message.stop_reason == "refusal":
            raise ProviderError("The model declined to answer (stop reason: refusal).")

        text = "".join(block.text for block in message.content if block.type == "text")
        # input_tokens leaves out the part of the prompt that was read from or
        # written to the cache. The budget is about the whole prompt.
        prompt_tokens = (
            message.usage.input_tokens
            + (message.usage.cache_creation_input_tokens or 0)
            + (message.usage.cache_read_input_tokens or 0)
        )
        return Reply(text=text, prompt_tokens=prompt_tokens)

    return complete


class OpenAIMessage(BaseModel):
    content: str | None = None


class OpenAIChoice(BaseModel):
    message: OpenAIMessage


class OpenAIUsage(BaseModel):
    prompt_tokens: int


class OpenAICompletion(BaseModel):
    choices: list[OpenAIChoice] = Field(min_length=1)
    usage: OpenAIUsage


def openai_model(settings: OpenAIModel, api_key: str, post: PostJson = post_json) -> Complete:
    url = settings.base_url.rstrip("/") + "/chat/completions"
    headers = {"authorization": f"Bearer {api_key}"}

    def complete(prompt: Prompt) -> Reply:
        messages = []
        if prompt.system is not None:
            messages.append({"role": "system", "content": prompt.system})
        messages.append({"role": "user", "content": prompt.user})
        body = {"model": settings.name, "messages": messages, **settings.params}

        try:
            completion = OpenAICompletion.model_validate(post(url, headers, body))
        except ValidationError as error:
            raise ProviderError(f"The reply was not a completion.\n{explain(error)}") from error

        return Reply(
            text=completion.choices[0].message.content or "",
            prompt_tokens=completion.usage.prompt_tokens,
        )

    return complete


class FakeRule(Shape):
    when: Annotated[tuple[str, ...], Field(min_length=1)]
    text: str


class FakeRules(Shape):
    reply: tuple[FakeRule, ...]


def load_fake_rules(path: Path) -> tuple[FakeRule, ...]:
    try:
        return FakeRules.model_validate(read_toml(path)).reply
    except ValidationError as error:
        raise ConfigError(f"{path} is not a valid replies file.\n{explain(error)}") from error


def fake_model(rules: Sequence[FakeRule]) -> Complete:
    def complete(prompt: Prompt) -> Reply:
        searched = (prompt.system or "") + "\n" + prompt.user
        for rule in rules:
            if all(phrase in searched for phrase in rule.when):
                return Reply(text=rule.text, prompt_tokens=len(searched.split()))
        raise ProviderError(
            "No rule in the fake replies file matches this prompt. "
            'Add a [[reply]] table whose "when" phrases all appear in it.'
        )

    return complete
