import io
import json
import urllib.error
import urllib.request

import pytest

from mizan import providers
from mizan.providers import (
    FakeRule,
    anthropic_model,
    connect,
    fake_model,
    label,
    load_fake_rules,
    openai_model,
    post_json,
)
from mizan.types import AnthropicModel, ConfigError, FakeModel, OpenAIModel, Prompt, ProviderError

PROMPT = Prompt(system="You answer for a shop.", user="Can I return boots after 40 days?")

ANTHROPIC_REPLY = {
    "id": "msg_01",
    "type": "message",
    "role": "assistant",
    "model": "claude-opus-5-5",
    "content": [
        {"type": "thinking", "thinking": "", "signature": "abc"},
        {"type": "text", "text": "Returns are accepted "},
        {"type": "text", "text": "within 30 days."},
    ],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 12,
        "cache_creation_input_tokens": 100,
        "cache_read_input_tokens": 30,
        "output_tokens": 9,
    },
}

ANTHROPIC_REFUSAL = {
    "id": "msg_02",
    "type": "message",
    "role": "assistant",
    "model": "claude-opus-5-5",
    "content": [],
    "stop_reason": "refusal",
    "stop_details": {"type": "refusal", "category": "cyber", "explanation": None},
    "usage": {"input_tokens": 12, "output_tokens": 0},
}

OPENAI_REPLY = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "model": "llama3.1",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Returns are accepted within 30 days."},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 41, "completion_tokens": 9, "total_tokens": 50},
}


class CannedPost:
    """Stands in for post_json. Returns one payload and keeps what it was sent."""

    def __init__(self, payload):
        self.payload = payload
        self.requests = []

    def __call__(self, url, headers, body):
        self.requests.append({"url": url, "headers": dict(headers), "body": dict(body)})
        return self.payload


def test_an_anthropic_reply_joins_its_text_blocks_and_counts_cached_prompt_tokens():
    settings = AnthropicModel(provider="anthropic", name="claude-opus-5-5")
    complete = anthropic_model(settings, "test-key", post=CannedPost(ANTHROPIC_REPLY))

    reply = complete(PROMPT)

    assert reply.text == "Returns are accepted within 30 days."
    assert reply.prompt_tokens == 142


def test_an_anthropic_reply_without_cache_fields_counts_the_input_tokens():
    payload = {**ANTHROPIC_REPLY, "usage": {"input_tokens": 12, "output_tokens": 9}}
    settings = AnthropicModel(provider="anthropic", name="claude-opus-5-5")

    reply = anthropic_model(settings, "test-key", post=CannedPost(payload))(PROMPT)

    assert reply.prompt_tokens == 12


def test_an_anthropic_refusal_is_an_error_and_not_an_empty_answer():
    settings = AnthropicModel(provider="anthropic", name="claude-opus-5-5")
    complete = anthropic_model(settings, "test-key", post=CannedPost(ANTHROPIC_REFUSAL))

    with pytest.raises(ProviderError, match="declined"):
        complete(PROMPT)


def test_the_anthropic_request_puts_the_system_prompt_at_the_top_and_merges_params():
    settings = AnthropicModel(
        provider="anthropic",
        name="claude-opus-5-5",
        params={"max_tokens": 900, "output_config": {"effort": "low"}},
    )
    post = CannedPost(ANTHROPIC_REPLY)

    anthropic_model(settings, "test-key", post=post)(PROMPT)

    (request,) = post.requests
    assert request["url"] == "https://api.anthropic.com/v1/messages"
    assert request["headers"] == {"x-api-key": "test-key", "anthropic-version": "2023-06-01"}
    assert request["body"] == {
        "model": "claude-opus-5-5",
        "max_tokens": 900,
        "system": "You answer for a shop.",
        "messages": [{"role": "user", "content": "Can I return boots after 40 days?"}],
        "output_config": {"effort": "low"},
    }


def test_the_anthropic_request_leaves_out_system_and_temperature_unless_they_are_set():
    settings = AnthropicModel(provider="anthropic", name="claude-opus-5-5")
    post = CannedPost(ANTHROPIC_REPLY)

    anthropic_model(settings, "test-key", post=post)(Prompt(system=None, user="Hello"))

    assert post.requests[0]["body"] == {
        "model": "claude-opus-5-5",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": "Hello"}],
    }


def test_an_openai_reply_gives_the_message_text_and_the_prompt_tokens():
    settings = OpenAIModel(provider="openai", name="llama3.1")

    reply = openai_model(settings, "test-key", post=CannedPost(OPENAI_REPLY))(PROMPT)

    assert reply.text == "Returns are accepted within 30 days."
    assert reply.prompt_tokens == 41


def test_the_openai_request_sends_the_system_prompt_as_the_first_message_and_merges_params():
    settings = OpenAIModel(
        provider="openai",
        name="llama3.1",
        base_url="http://localhost:11434/v1/",
        params={"temperature": 0},
    )
    post = CannedPost(OPENAI_REPLY)

    openai_model(settings, "test-key", post=post)(PROMPT)

    (request,) = post.requests
    assert request["url"] == "http://localhost:11434/v1/chat/completions"
    assert request["headers"] == {"authorization": "Bearer test-key"}
    assert request["body"] == {
        "model": "llama3.1",
        "messages": [
            {"role": "system", "content": "You answer for a shop."},
            {"role": "user", "content": "Can I return boots after 40 days?"},
        ],
        "temperature": 0,
    }


def test_the_openai_request_has_no_system_message_when_there_is_no_system_prompt():
    post = CannedPost(OPENAI_REPLY)
    settings = OpenAIModel(provider="openai", name="llama3.1")

    openai_model(settings, "test-key", post=post)(Prompt(system=None, user="Hello"))

    assert post.requests[0]["body"]["messages"] == [{"role": "user", "content": "Hello"}]


@pytest.mark.parametrize(
    ("build", "payload"),
    [
        (
            lambda post: anthropic_model(
                AnthropicModel(provider="anthropic", name="claude-opus-5-5"), "test-key", post=post
            ),
            {key: value for key, value in ANTHROPIC_REPLY.items() if key != "usage"},
        ),
        (
            lambda post: openai_model(
                OpenAIModel(provider="openai", name="llama3.1"), "test-key", post=post
            ),
            {key: value for key, value in OPENAI_REPLY.items() if key != "usage"},
        ),
    ],
    ids=["anthropic", "openai"],
)
def test_a_reply_without_usage_is_an_error_so_the_budget_is_never_unmeasured(build, payload):
    complete = build(CannedPost(payload))

    with pytest.raises(ProviderError, match="usage"):
        complete(PROMPT)


def test_a_missing_api_key_stops_before_any_call_and_names_the_variable(tmp_path):
    settings = OpenAIModel(provider="openai", name="llama3.1", api_key_env="OLLAMA_API_KEY")

    with pytest.raises(ConfigError, match="OLLAMA_API_KEY is not set"):
        connect(settings, tmp_path)

    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY is not set"):
        connect(AnthropicModel(provider="anthropic", name="claude-opus-5-5"), tmp_path)


def test_labels_name_the_provider_and_the_model():
    assert label(AnthropicModel(provider="anthropic", name="claude-opus-5-5")) == (
        "anthropic:claude-opus-5-5"
    )
    assert label(OpenAIModel(provider="openai", name="llama3.1")) == "openai:llama3.1"
    assert label(FakeModel(provider="fake", replies="fake/model.toml")) == "fake:model.toml"


FAKE_RULES = (
    FakeRule(when=("within 30 days", "40 days"), text="This order can no longer be returned."),
    FakeRule(when=("40 days",), text="Send them back whenever you like."),
)


def test_the_fake_answers_with_the_first_rule_whose_phrases_all_appear():
    complete = fake_model(FAKE_RULES)

    with_policy = complete(Prompt(system="Returns are accepted within 30 days.", user="40 days?"))
    without_policy = complete(Prompt(system="Be brief.", user="40 days?"))

    assert with_policy.text == "This order can no longer be returned."
    assert without_policy.text == "Send them back whenever you like."


def test_the_fake_counts_the_words_of_the_whole_prompt_as_tokens():
    complete = fake_model(FAKE_RULES)

    assert complete(Prompt(system="one two three", user="40 days")).prompt_tokens == 5
    assert complete(Prompt(system=None, user="40 days")).prompt_tokens == 2


def test_the_fake_fails_when_no_rule_matches_and_does_not_invent_an_answer():
    with pytest.raises(ProviderError, match="No rule"):
        fake_model(FAKE_RULES)(Prompt(system=None, user="Where is my order?"))


def test_a_fake_connects_from_its_replies_file(tmp_path):
    (tmp_path / "model.toml").write_text('[[reply]]\nwhen = ["order"]\ntext = "It has shipped."\n')

    complete = connect(FakeModel(provider="fake", replies="model.toml"), tmp_path)

    assert complete(Prompt(system=None, user="Where is my order?")).text == "It has shipped."


def test_a_replies_file_with_a_mistake_is_rejected_with_the_file_named(tmp_path):
    path = tmp_path / "model.toml"
    path.write_text('[[reply]]\nwhen = "order"\ntext = "It has shipped."\n')

    with pytest.raises(ConfigError, match="(?s)model.toml is not a valid replies file.*when"):
        load_fake_rules(path)


class CannedHttp:
    """Stands in for urllib.request.urlopen. Each call takes the next outcome."""

    def __init__(self, monkeypatch, *outcomes):
        self.outcomes = list(outcomes)
        self.sleeps = []
        monkeypatch.setattr(urllib.request, "urlopen", self.open)
        monkeypatch.setattr(providers.time, "sleep", self.sleeps.append)

    def open(self, request, timeout):
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return io.BytesIO(json.dumps(outcome).encode())


def http_error(status, body):
    return urllib.error.HTTPError("https://example.test", status, "", {}, io.BytesIO(body.encode()))


def post():
    return post_json("https://example.test/v1/messages", {"x-api-key": "secret-key"}, {"a": 1})


def test_post_json_returns_the_reply_after_a_busy_server_recovers(monkeypatch):
    http = CannedHttp(
        monkeypatch,
        http_error(529, '{"type": "error", "error": {"type": "overloaded_error"}}'),
        urllib.error.URLError("connection refused"),
        {"ok": True},
    )

    assert post() == {"ok": True}
    assert http.sleeps == [1, 2]


def test_post_json_gives_up_after_three_attempts_and_says_why(monkeypatch):
    overloaded = '{"type": "error", "error": {"type": "overloaded_error"}}'
    CannedHttp(
        monkeypatch,
        http_error(529, overloaded),
        http_error(529, overloaded),
        http_error(529, overloaded),
        {"ok": True},
    )

    with pytest.raises(ProviderError, match="HTTP 529: .*overloaded_error.*after 3 attempts"):
        post()


def test_post_json_does_not_retry_a_request_the_server_rejected(monkeypatch):
    http = CannedHttp(monkeypatch, http_error(401, '{"error": "invalid x-api-key"}'), {"ok": True})

    with pytest.raises(ProviderError, match="HTTP 401: .*invalid x-api-key"):
        post()
    assert http.sleeps == []


def test_post_json_never_puts_the_api_key_in_an_error(monkeypatch):
    CannedHttp(monkeypatch, http_error(400, "bad request " + "x" * 500))

    with pytest.raises(ProviderError) as raised:
        post()

    assert "secret-key" not in str(raised.value)
    assert len(str(raised.value)) < 250


def test_post_json_gives_up_on_a_server_that_cannot_be_reached(monkeypatch):
    CannedHttp(monkeypatch, TimeoutError("timed out"), TimeoutError("timed out"), TimeoutError())

    with pytest.raises(ProviderError, match="did not get through.*after 3 attempts"):
        post()
