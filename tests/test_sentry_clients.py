"""API contract tests: mocked HTTP plus a local server; no paid API calls."""
from __future__ import annotations

import io
import json
import threading
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError

import pytest

from Sentry.clients import (
    AnthropicMessagesClient, ChatCompletionsClient, GeminiClient,
    ModelOutputTruncatedError, OpenAIChatCompletionsClient, OpenAIResponsesClient,
    ProviderConfig, create_client,
)


PROVIDERS = ["openai", "openai_compatible", "openai_responses", "anthropic", "gemini"]


def success(provider, text='{"valid":true}'):
    if provider in {"openai", "openai_compatible"}:
        return {"choices": [{"finish_reason": "stop", "message": {"content": text}}],
                "usage": {"total_tokens": 17}}
    if provider == "openai_responses":
        return {"status": "completed", "output": [
            {"type": "reasoning", "summary": []},
            {"type": "message", "role": "assistant", "status": "completed",
             "content": [{"type": "output_text", "text": text}]}],
            "usage": {"total_tokens": 17}}
    if provider == "anthropic":
        return {"stop_reason": "end_turn", "content": [{"type": "text", "text": text}],
                "usage": {"input_tokens": 12, "output_tokens": 5}}
    return {"candidates": [{"finishReason": "STOP", "content": {
        "parts": [{"thought": True, "text": "private reasoning"}, {"text": text}]}}],
        "usageMetadata": {"totalTokenCount": 17}}


@pytest.fixture
def fake_http(monkeypatch):
    calls = []
    replies = []

    def open_request(request, timeout):
        calls.append((request, timeout))
        reply = replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return io.BytesIO(reply if isinstance(reply, bytes) else json.dumps(reply).encode())

    monkeypatch.setattr("Sentry.clients.urlopen", open_request)
    return calls, replies


def client_for(provider, monkeypatch, **kwargs):
    monkeypatch.setenv("TEST_PROVIDER_KEY", "test-credential")
    config = ProviderConfig(
        provider=provider, model="exact-model", base_url="https://example.invalid/v1",
        api_key_env="TEST_PROVIDER_KEY",
        json_mode=kwargs.pop("json_mode", "prompt" if provider == "anthropic" else "native"),
        **kwargs,
    )
    return create_client(config)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_native_provider_contract_and_usage(provider, fake_http, monkeypatch):
    calls, replies = fake_http
    replies.append(success(provider))
    client = client_for(provider, monkeypatch)
    assert client("Return JSON", temperature=0, max_tokens=4096,
                  response_format={"type": "json_object"}) == '{"valid":true}'
    req, timeout = calls[0]
    body = json.loads(req.data)
    assert req.method == "POST" and timeout == 120
    assert "test-credential" not in repr(client)
    assert "test-credential" not in req.full_url and "test-credential" not in req.data.decode()
    assert client.last_usage == success(provider).get("usage", success(provider).get("usageMetadata"))
    if provider in {"openai", "openai_compatible"}:
        assert req.full_url.endswith("/v1/chat/completions")
        assert req.get_header("Authorization") == "Bearer test-credential"
        token_field = "max_completion_tokens" if provider == "openai" else "max_tokens"
        assert body[token_field] == 4096 and body["temperature"] == 0
        assert body["response_format"] == {"type": "json_object"}
    elif provider == "openai_responses":
        assert req.full_url.endswith("/v1/responses")
        assert body["max_output_tokens"] == 4096 and body["temperature"] == 0
        assert body["text"] == {"format": {"type": "json_object"}}
        assert body["store"] is False
    elif provider == "anthropic":
        assert req.full_url.endswith("/v1/messages")
        assert req.get_header("X-api-key") == "test-credential"
        assert req.get_header("Anthropic-version") == "2023-06-01"
        assert body["max_tokens"] == 4096 and body["temperature"] == 0
        assert "response_format" not in body
        assert "exactly one valid JSON object" in body["messages"][0]["content"]
    else:
        assert req.full_url.endswith("/v1/models/exact-model:generateContent")
        assert req.get_header("X-goog-api-key") == "test-credential"
        assert body["generationConfig"] == {
            "temperature": 0, "maxOutputTokens": 4096,
            "candidateCount": 1, "responseMimeType": "application/json"}


@pytest.mark.parametrize("provider", PROVIDERS)
def test_task_agent_request_keeps_temperature_budget_and_action_text(provider, fake_http, monkeypatch):
    calls, replies = fake_http
    replies.append(success(provider, "click[red]"))
    client = client_for(provider, monkeypatch)
    assert client("continue", temperature=0.7, max_tokens=384) == "click[red]"
    body = json.loads(calls[0][0].data)
    generation = body.get("generationConfig", body)
    assert generation["temperature"] == 0.7
    assert generation.get("maxOutputTokens", generation.get("max_output_tokens",
               generation.get("max_completion_tokens", generation.get("max_tokens")))) == 384
    assert "response_format" not in body and "responseMimeType" not in generation
    assert "text" not in body


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("output", ['[]', 'null', '{} trailing', '```json\n{}\n```',
                                   '{"n":NaN}', '{"n":Infinity}', '{"n":1e999}', '{"a":1,"a":2}', ''])
def test_invalid_structured_output_is_rejected_without_repair(provider, output, fake_http, monkeypatch):
    calls, replies = fake_http
    replies.append(success(provider, output))
    with pytest.raises(RuntimeError):
        client_for(provider, monkeypatch)("Return JSON", temperature=0, max_tokens=4096,
                                         response_format={"type": "json_object"})
    assert len(calls) == 1


@pytest.mark.parametrize("reason", [None, "length", "content_filter", "tool_calls", "function_call"])
def test_chat_rejects_incomplete_or_unknown_finish_reason(reason, fake_http):
    _, replies = fake_http
    response = success("openai")
    response["choices"][0]["finish_reason"] = reason
    replies.append(response)
    with pytest.raises(RuntimeError, match="incomplete"):
        ChatCompletionsClient("https://example.invalid/v1", "test")("x", temperature=0, max_tokens=10)


@pytest.mark.parametrize("provider,field,value", [
    ("openai_responses", "status", "incomplete"),
    ("openai_responses", "status", "failed"),
    ("anthropic", "stop_reason", "max_tokens"),
    ("anthropic", "stop_reason", "tool_use"),
    ("anthropic", "stop_reason", "refusal"),
    ("anthropic", "stop_reason", "model_context_window_exceeded"),
])
def test_native_incomplete_responses_rejected(provider, field, value, fake_http, monkeypatch):
    _, replies = fake_http
    response = success(provider)
    response[field] = value
    replies.append(response)
    with pytest.raises(RuntimeError):
        client_for(provider, monkeypatch)("x", temperature=0, max_tokens=10)


@pytest.mark.parametrize("provider", ["openai_responses", "anthropic", "gemini"])
@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("text", ["", '{"status":"resolved"}'])
def test_native_token_limit_preserves_unaccepted_text_for_host_recovery(
    provider, structured, text, fake_http, monkeypatch,
):
    calls, replies = fake_http
    response = success(provider, text)
    if provider == "openai_responses":
        response.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
        response["output"][1]["status"] = "incomplete"
    elif provider == "anthropic":
        response["stop_reason"] = "max_tokens"
    else:
        response["candidates"][0]["finishReason"] = "MAX_TOKENS"
    replies.append(response)
    client = client_for(provider, monkeypatch)
    with pytest.raises(ModelOutputTruncatedError) as error:
        client("Return JSON", temperature=0, max_tokens=10,
               response_format={"type": "json_object"} if structured else None)
    assert error.value.partial_text == text
    assert client.last_response_metadata["finish_reason"] == error.value.finish_reason
    assert client.last_usage
    assert len(calls) == 1
    if text:
        assert text not in str(error.value) and text not in repr(error.value)


@pytest.mark.parametrize("provider", ["openai_responses", "anthropic", "gemini"])
def test_nontext_token_limited_payload_is_not_treated_as_recoverable_action(
    provider, fake_http, monkeypatch,
):
    _, replies = fake_http
    response = success(provider)
    if provider == "openai_responses":
        response.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
        response["output"].append({"type": "function_call", "name": "finish"})
    elif provider == "anthropic":
        response["stop_reason"] = "max_tokens"
        response["content"].append({"type": "tool_use", "name": "finish"})
    else:
        response["candidates"][0]["finishReason"] = "MAX_TOKENS"
        response["candidates"][0]["content"]["parts"].append({"functionCall": {"name": "finish"}})
    replies.append(response)
    with pytest.raises(RuntimeError) as error:
        client_for(provider, monkeypatch)("x", temperature=0, max_tokens=10)
    assert not isinstance(error.value, ModelOutputTruncatedError)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_refusal_block_and_tool_responses_fail_closed(provider, fake_http, monkeypatch):
    _, replies = fake_http
    response = success(provider)
    if provider in {"openai", "openai_compatible"}:
        response["choices"][0]["message"]["refusal"] = "private message"
    elif provider == "openai_responses":
        response["output"][1]["content"].append({"type": "refusal", "refusal": "private message"})
    elif provider == "anthropic":
        response["stop_details"] = {"type": "refusal", "explanation": "private message"}
    else:
        response["promptFeedback"] = {"blockReason": "SAFETY"}
    replies.append(response)
    with pytest.raises(RuntimeError) as error:
        client_for(provider, monkeypatch)("x", temperature=0, max_tokens=10)
    assert "private message" not in str(error.value)


@pytest.mark.parametrize("finish", ["MAX_TOKENS", "SAFETY", None])
def test_gemini_checks_finish_status(finish, fake_http, monkeypatch):
    _, replies = fake_http
    response = success("gemini")
    response["candidates"][0]["finishReason"] = finish
    replies.append(response)
    with pytest.raises(RuntimeError, match="incomplete"):
        client_for("gemini", monkeypatch)("x", temperature=0, max_tokens=10)


@pytest.mark.parametrize("provider", ["openai", "openai_compatible", "openai_responses", "gemini"])
def test_prompt_fallback_is_explicit(provider, fake_http, monkeypatch):
    calls, replies = fake_http
    replies.append(success(provider))
    client = client_for(provider, monkeypatch, json_mode="prompt")
    client("schema", temperature=0, max_tokens=4096, response_format={"type": "json_object"})
    body = json.loads(calls[0][0].data)
    assert "response_format" not in body and "text" not in body
    assert "responseMimeType" not in body.get("generationConfig", {})
    assert "exactly one valid JSON object" in calls[0][0].data.decode()


@pytest.mark.parametrize("kwargs", [
    {"provider": "unknown"}, {"provider": []}, {"model": ""}, {"model": None},
    {"timeout": True}, {"timeout": 10 ** 1000}, {"timeout": 0}, {"timeout": float("nan")}, {"timeout": float("inf")},
    {"max_retries": -1}, {"max_retries": True}, {"max_retries": 6},
    {"json_mode": "automatic"}, {"api_key_env": "sk-secret-value"},
    {"provider": "anthropic"}, {"provider": "openai_compatible"},
    {"base_url": "https://user:secret@example.invalid"},
    {"base_url": "https://example.invalid/v1?api_key=secret"},
    {"base_url": "https://example.invalid/v1#secret"},
    {"base_url": "https://example.invalid:invalid/v1"},
    {"base_url": "https://example.invalid/\n"}, {"base_url": "file:///tmp/x"},
])
def test_configuration_validation(kwargs):
    with pytest.raises(ValueError):
        ProviderConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"temperature": True}, {"temperature": 10 ** 1000}, {"temperature": float("nan")}, {"temperature": 3},
    {"temperature": -1}, {"max_tokens": True}, {"max_tokens": 0},
    {"max_tokens": 1.5}, {"response_format": {"type": "json_schema"}},
])
def test_call_validation_before_io(kwargs, fake_http):
    calls, _ = fake_http
    with pytest.raises(ValueError):
        ChatCompletionsClient("https://example.invalid", "test")(
            "x", **{"temperature": 0, "max_tokens": 4096, **kwargs})
    assert calls == []


def test_credentials_are_resolved_at_factory_time_and_not_serialized(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MISSING_TOKEN", raising=False)
    config = ProviderConfig(model="exact-model")
    assert "api_key" not in asdict(config)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        create_client(config)
    monkeypatch.setenv("OPENAI_API_KEY", "secret-value")
    client = create_client(config)
    assert isinstance(client, OpenAIChatCompletionsClient)
    assert "secret-value" not in repr(client) and "secret-value" not in repr(config)
    assert create_client(ProviderConfig(provider="openai_compatible", base_url="http://localhost:8000/v1")).api_key == ""
    with pytest.raises(ValueError, match="MISSING_TOKEN"):
        create_client(ProviderConfig(provider="openai_compatible", base_url="http://localhost:8000/v1",
                                     api_key_env="MISSING_TOKEN"))


def test_environment_entry_point_does_not_read_files(monkeypatch):
    for name in ["SENTRY_PROVIDER", "SENTRY_MODEL", "SENTRY_API_BASE", "SENTRY_API_KEY",
                 "SENTRY_API_KEY_ENV", "SENTRY_TIMEOUT", "SENTRY_MAX_RETRIES", "SENTRY_JSON_MODE",
                 "SENTRY_REASONING_ENABLED"]:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="SENTRY_MODEL"):
        ProviderConfig.from_env()
    monkeypatch.setenv("SENTRY_MODEL", "Qwen3.5-9B")
    monkeypatch.setenv("SENTRY_API_BASE", "http://localhost:8000/v1")
    monkeypatch.setenv("SENTRY_API_KEY", "")
    assert ProviderConfig.from_env().provider == "openai_compatible"
    assert create_client(ProviderConfig.from_env()).api_key == ""
    monkeypatch.setenv("SENTRY_API_KEY", "local-test-key")
    assert ProviderConfig.from_env().api_key_env == "SENTRY_API_KEY"
    assert create_client(ProviderConfig.from_env()).api_key == "local-test-key"


@pytest.mark.parametrize("error", [
    HTTPError("https://secret.invalid", 401, "secret", {}, io.BytesIO(b"secret")),
    URLError("secret"), TimeoutError("secret"), ConnectionResetError("secret"),
])
def test_transport_errors_are_redacted_and_usage_reset(error, fake_http):
    calls, replies = fake_http
    replies.extend([success("openai"), error])
    client = ChatCompletionsClient("https://example.invalid", "test", "secret")
    client("x", temperature=0, max_tokens=10)
    assert client.last_usage
    with pytest.raises(RuntimeError) as caught:
        client("x", temperature=0, max_tokens=10)
    assert "secret" not in str(caught.value) and caught.value.__suppress_context__
    assert client.last_usage == {} and len(calls) == 2


@pytest.mark.parametrize("body", [b"[]", b"not json", b'{"error":{"message":"secret"}}',
                                  b'{"choices":[],"choices":[]}', b'\xff'])
def test_malformed_http_response_has_no_content_leak(body, fake_http):
    _, replies = fake_http
    replies.append(body)
    with pytest.raises(RuntimeError) as caught:
        ChatCompletionsClient("https://example.invalid", "test")("x", temperature=0, max_tokens=10)
    assert "secret" not in str(caught.value)


def test_opt_in_retries_only_transient_errors(fake_http, monkeypatch):
    calls, replies = fake_http
    sleeps = []
    monkeypatch.setattr("Sentry.clients.time.sleep", sleeps.append)
    replies.extend([HTTPError("https://example.invalid", 429, "private", {}, io.BytesIO()),
                    success("openai")])
    client = ChatCompletionsClient("https://example.invalid", "test", max_retries=1)
    assert client("x", temperature=0, max_tokens=10) == '{"valid":true}'
    assert len(calls) == 2 and sleeps == [1]
    replies.append(HTTPError("https://example.invalid", 400, "private", {}, io.BytesIO()))
    with pytest.raises(RuntimeError, match="decoding and JSON"):
        client("x", temperature=0, max_tokens=10)
    assert len(calls) == 3


@pytest.fixture
def local_server():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            requests.append((self.path, self.headers.get("Authorization")))
            if self.path.startswith("/redirect/"):
                self.send_response(int(self.path.split("/")[2]))
                self.send_header("Location", "/stolen")
                self.end_headers()
            else:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(success("openai")).encode())

        def do_GET(self):
            requests.append((self.path, self.headers.get("Authorization")))
            self.send_response(200)
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_real_http_transport_and_no_credential_forwarding(local_server):
    base, requests = local_server
    client = ChatCompletionsClient(base + "/v1", "test", "local-test-key", timeout=2)
    assert client("x", temperature=0, max_tokens=10) == '{"valid":true}'
    for code in [301, 302, 303, 307, 308]:
        redirect = ChatCompletionsClient(base + f"/redirect/{code}", "test", "local-test-key", timeout=2)
        with pytest.raises(RuntimeError, match=f"HTTP {code}"):
            redirect("x", temperature=0, max_tokens=10)
    assert len(requests) == 6 and all(path != "/stolen" for path, _ in requests)
    assert all(auth == "Bearer local-test-key" for _, auth in requests)
