"""OpenRouter HTTP/config contracts; no authenticated or paid requests."""
from __future__ import annotations

import io
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from Sentry import SentryConfig
from Sentry.clients import ModelOutputTruncatedError, OpenRouterClient, ProviderConfig, create_client


@pytest.fixture
def transport(monkeypatch):
    calls = []
    replies = []

    def send(request, timeout):
        calls.append((request, timeout))
        return io.BytesIO(json.dumps(replies.pop(0)).encode())

    monkeypatch.setattr("Sentry.clients.urlopen", send)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-credential")
    return calls, replies


def completion(content, finish_reason="stop", **message_fields):
    return {
        "model": "qwen/qwen3.5-9b",
        "provider": "Test provider",
        "choices": [{"finish_reason": finish_reason,
                     "message": {"role": "assistant", "content": content, **message_fields}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17,
                  "cost": 0.000002, "completion_tokens_details": {"reasoning_tokens": 0}},
    }


@pytest.mark.parametrize("temperature,max_tokens,structured", [(0.7, 384, False), (0, 4096, True)])
def test_openrouter_preserves_decoding_and_requires_parameter_support(
    transport, temperature, max_tokens, structured,
):
    calls, replies = transport
    expected = '{"valid":true}' if structured else "print(apis.supervisor.show_account_passwords())"
    replies.append(completion(expected, reasoning="hidden text"))
    client = create_client(ProviderConfig(
        provider="openrouter", model="qwen/qwen3.5-9b", reasoning_enabled=False,
    ))
    assert isinstance(client, OpenRouterClient)
    assert client("Return JSON" if structured else "continue", temperature=temperature,
                  max_tokens=max_tokens,
                  response_format={"type": "json_object"} if structured else None) == expected
    request, timeout = calls[0]
    payload = json.loads(request.data)
    assert request.full_url == "https://openrouter.ai/api/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer test-openrouter-credential"
    assert timeout == 120 and payload["stream"] is False
    assert payload["model"] == "qwen/qwen3.5-9b"
    assert payload["max_tokens"] == max_tokens and payload["temperature"] == temperature
    assert payload["provider"] == {"require_parameters": True}
    assert payload["reasoning"] == {"enabled": False}
    assert payload.get("response_format") == ({"type": "json_object"} if structured else None)
    assert client.last_usage == completion(expected)["usage"]
    assert "test-openrouter-credential" not in repr(client)
    assert "test-openrouter-credential" not in request.data.decode()


@pytest.mark.parametrize("reasoning", [None, True, False])
def test_reasoning_is_explicit_and_serializable(transport, reasoning):
    calls, replies = transport
    replies.append(completion("ok"))
    config = ProviderConfig(provider="openrouter", model="exact-model", reasoning_enabled=reasoning)
    assert asdict(config)["reasoning_enabled"] is reasoning
    assert "api_key" not in asdict(config)
    create_client(config)("continue", temperature=0.7, max_tokens=384)
    body = json.loads(calls[0][0].data)
    if reasoning is None:
        assert "reasoning" not in body
    else:
        assert body["reasoning"] == {"enabled": reasoning}


def test_prompt_json_fallback_is_opt_in(transport):
    calls, replies = transport
    replies.append(completion('{"ok":true}'))
    client = create_client(ProviderConfig(provider="openrouter", model="exact-model", json_mode="prompt"))
    client("schema", temperature=0, max_tokens=4096, response_format={"type": "json_object"})
    body = json.loads(calls[0][0].data)
    assert "response_format" not in body
    assert body["provider"] == {"require_parameters": True}
    assert "exactly one valid JSON object" in body["messages"][0]["content"]


@pytest.mark.parametrize("reason", ["length", "error", "content_filter", "tool_calls", None])
def test_openrouter_rejects_incomplete_valid_json_and_retains_usage(transport, reason):
    calls, replies = transport
    replies.append(completion('{"valid":true}', finish_reason=reason))
    client = create_client(ProviderConfig(provider="openrouter", model="exact-model"))
    with pytest.raises(RuntimeError, match="incomplete"):
        client("Return JSON", temperature=0, max_tokens=4096, response_format={"type": "json_object"})
    assert len(calls) == 1
    assert client.last_usage["cost"] == 0.000002


def test_reasoning_is_never_substituted_for_empty_answer(transport):
    _, replies = transport
    replies.append(completion("", reasoning='{"valid":true}'))
    client = create_client(ProviderConfig(provider="openrouter", model="exact-model"))
    with pytest.raises(RuntimeError, match="empty or non-text"):
        client("Return JSON", temperature=0, max_tokens=4096, response_format={"type": "json_object"})


@pytest.mark.parametrize("content", ['[]', '```json\n{"valid":true}\n```', '{"valid":true,"valid":false}'])
def test_native_json_response_is_still_locally_validated(transport, content):
    calls, replies = transport
    replies.append(completion(content))
    client = create_client(ProviderConfig(provider="openrouter", model="exact-model"))
    with pytest.raises(RuntimeError, match="invalid JSON"):
        client("Return JSON", temperature=0, max_tokens=4096, response_format={"type": "json_object"})
    assert len(calls) == 1


@pytest.mark.parametrize("reasoning", [0, 1, "false", "", [], {}])
def test_reasoning_configuration_rejects_non_booleans(reasoning):
    with pytest.raises(ValueError, match="reasoning_enabled"):
        ProviderConfig(provider="openrouter", reasoning_enabled=reasoning)
    with pytest.raises(ValueError, match="reasoning_enabled"):
        OpenRouterClient("https://example.invalid", "model", reasoning_enabled=reasoning)


def test_reasoning_option_is_not_silently_dropped_for_other_providers():
    with pytest.raises(ValueError, match="only by the OpenRouter"):
        ProviderConfig(provider="openai", reasoning_enabled=False)


@pytest.fixture
def openrouter_env(monkeypatch):
    for key in (
        "SENTRY_API_BASE", "SENTRY_API_KEY", "SENTRY_API_KEY_ENV", "SENTRY_TIMEOUT",
        "SENTRY_MAX_RETRIES", "SENTRY_JSON_MODE", "SENTRY_REASONING_ENABLED",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SENTRY_PROVIDER", "openrouter")
    monkeypatch.setenv("SENTRY_MODEL", "qwen/qwen3.5-9b")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-env-credential")
    return monkeypatch


@pytest.mark.parametrize("value,expected", [(None, None), ("true", True), ("false", False)])
def test_openrouter_environment_configuration(openrouter_env, value, expected):
    if value is not None:
        openrouter_env.setenv("SENTRY_REASONING_ENABLED", value)
    config = ProviderConfig.from_env()
    assert config.reasoning_enabled is expected
    assert create_client(config).base_url == "https://openrouter.ai/api/v1"


@pytest.mark.parametrize("value", ["", "0", "False", " false "])
def test_invalid_reasoning_environment_setting_fails(openrouter_env, value):
    openrouter_env.setenv("SENTRY_REASONING_ENABLED", value)
    with pytest.raises(ValueError, match="SENTRY_REASONING_ENABLED"):
        ProviderConfig.from_env()


def test_openrouter_requires_explicitly_available_credential(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    config = ProviderConfig(provider="openrouter", model="exact-model")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        create_client(config)
    monkeypatch.setenv("CUSTOM_ROUTER_TOKEN", "test-override")
    assert create_client(ProviderConfig(provider="openrouter", model="exact-model",
                                        api_key_env="CUSTOM_ROUTER_TOKEN")).api_key == "test-override"


def test_shipped_openrouter_configuration_preserves_backbone_and_paper_budgets():
    config_path = Path(__file__).resolve().parents[1] / "configs/providers/openrouter.yaml"
    config = SentryConfig.from_yaml(config_path)
    assert config.api.provider == "openrouter"
    assert config.api.model == config.model.model == "qwen/qwen3.5-9b"
    assert config.api.reasoning_enabled is False
    assert config.api.max_retries == 0
    assert (config.model.temperature, config.model.max_tokens) == (0.0, 4096)
    assert (config.task_agent.temperature, config.task_agent.max_tokens) == (0.7, 384)


def test_response_metadata_is_allowlisted_and_survives_output_validation(transport):
    _, replies = transport
    reply = completion('{"valid":true}', finish_reason="length")
    reply.update(id="test-generation-id", internal_metadata={"secret": "not retained"})
    replies.append(reply)
    client = create_client(ProviderConfig(provider="openrouter", model="exact-model"))
    with pytest.raises(RuntimeError, match="incomplete"):
        client("Return JSON", temperature=0, max_tokens=4096, response_format={"type": "json_object"})
    assert client.last_response_metadata == {
        "id": "test-generation-id", "model": "qwen/qwen3.5-9b", "provider": "Test provider",
        "finish_reason": "length",
    }


def test_response_metadata_resets_before_input_validation_and_transport_errors(transport):
    _, replies = transport
    replies.append(completion("ok"))
    client = create_client(ProviderConfig(provider="openrouter", model="exact-model"))
    client("continue", temperature=0.7, max_tokens=384)
    assert client.last_response_metadata
    with pytest.raises(ValueError, match="prompt"):
        client("", temperature=0.7, max_tokens=384)
    assert client.last_response_metadata == {} and client.last_usage == {}
    replies.extend([completion("ok"), {"error": {"message": "not retained"}}])
    client("continue", temperature=0.7, max_tokens=384)
    with pytest.raises(RuntimeError, match="API error"):
        client("continue", temperature=0.7, max_tokens=384)
    assert client.last_response_metadata == {} and client.last_usage == {}


def test_response_metadata_excludes_nonstring_or_empty_values(transport):
    _, replies = transport
    reply = completion("ok")
    reply.update(id={"nested": "not retained"}, model="", provider=17)
    replies.append(reply)
    client = create_client(ProviderConfig(provider="openrouter", model="exact-model"))
    assert client.last_response_metadata == {}
    client("continue", temperature=0.7, max_tokens=384)
    assert client.last_response_metadata == {"finish_reason": "stop"}


@pytest.mark.parametrize("partial", ["print('private partial", "print('private but parseable')", ""])
def test_typed_truncation_preserves_rejected_text_without_logging_or_retry(transport, partial):
    calls, replies = transport
    replies.append(completion(partial, finish_reason="length"))
    client = create_client(ProviderConfig(provider="openrouter", model="exact-model", max_retries=2))
    with pytest.raises(ModelOutputTruncatedError) as caught:
        client("continue", temperature=0.7, max_tokens=384)
    error = caught.value
    assert error.partial_text == partial and error.finish_reason == "length"
    assert "private" not in str(error) and "private" not in repr(error)
    assert "private" not in str(error.args)
    assert error.__suppress_context__ is True
    assert client.last_response_metadata["finish_reason"] == "length"
    assert client.last_usage == completion(partial)["usage"]
    assert len(calls) == 1


def test_valid_json_is_never_accepted_when_truncated(transport):
    calls, replies = transport
    replies.append(completion('{"valid":true}', finish_reason="length"))
    client = create_client(ProviderConfig(provider="openrouter", model="exact-model"))
    with pytest.raises(ModelOutputTruncatedError) as caught:
        client("Return JSON", temperature=0, max_tokens=4096, response_format={"type": "json_object"})
    assert caught.value.partial_text == '{"valid":true}' and len(calls) == 1


@pytest.mark.parametrize("fields", [
    {"refusal": "declined"}, {"tool_calls": [{"id": "call-1"}]},
    {"function_call": {"name": "fn"}}, {"content": None},
    {"content": [{"type": "text", "text": "partial"}]},
])
def test_ambiguous_or_refused_truncation_is_not_exposed_as_partial_text(transport, fields):
    _, replies = transport
    reply = completion("partial", finish_reason="length")
    reply["choices"][0]["message"].update(fields)
    replies.append(reply)
    client = create_client(ProviderConfig(provider="openrouter", model="exact-model"))
    with pytest.raises(RuntimeError) as caught:
        client("continue", temperature=0.7, max_tokens=384)
    assert not isinstance(caught.value, ModelOutputTruncatedError)
    assert not hasattr(caught.value, "partial_text")
