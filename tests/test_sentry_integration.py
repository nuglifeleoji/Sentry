from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from Sentry import (
    ProviderConfig,
    SentryRunner,
    agent_step_from_parts,
    create_client,
    model_to_text_judge,
)
from Sentry.clients import ChatCompletionsClient


def test_repairs_use_task_agent_decoding_and_keep_objective_and_observation():
    calls = []
    def model(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return "click[visible]"
    runner = SentryRunner(model)
    runner.start_task(objective="inspect the red product", action_schema="click[visible]")
    repair = runner.step(agent_step_from_parts(
        step_id=0, reasoning="open details", raw_action="bogus",
        observation="visible product costs $12", parsed_ok=False, schema_valid=False,
    ))
    assert runner.generate_repair(repair) == "click[visible]"
    prompt, kwargs = calls.pop()
    assert kwargs == {"temperature": 0.7, "max_tokens": 384}
    assert "inspect the red product" in prompt and "visible product costs $12" in prompt


def test_sdk_shaped_model_response_and_unsupported_response():
    response = SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="{}"))])
    assert model_to_text_judge(lambda *a, **kw: response)("test") == "{}"
    with pytest.raises(TypeError, match="text content"):
        model_to_text_judge(lambda *a, **kw: object())("test")


def test_live_client_preserves_explicit_model_decoding_and_json_format(monkeypatch):
    requests = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            return json.dumps({"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                               "usage": {"total_tokens": 17}}).encode()
    def fake_open(request, timeout):
        requests.append((request, timeout))
        return Response()
    monkeypatch.setattr("Sentry.clients.urlopen", fake_open)
    monkeypatch.setenv("TEST_SENTRY_API_KEY", "test-secret")
    model = create_client(ProviderConfig(
        provider="openai_compatible", base_url="http://localhost:8000/v1/",
        model="exact-model", api_key_env="TEST_SENTRY_API_KEY",
    ))
    assert model_to_text_judge(model)("return JSON") == "{}"
    req, timeout = requests[0]
    body = json.loads(req.data)
    assert req.full_url == "http://localhost:8000/v1/chat/completions"
    assert req.get_header("Authorization") == "Bearer test-secret"
    assert body["model"] == "exact-model"
    assert body["temperature"] == 0 and body["max_tokens"] == 4096
    assert body["response_format"] == {"type": "json_object"}
    assert model.last_usage == {"total_tokens": 17}
    assert "test-secret" not in repr(model)


def test_client_rejects_missing_or_empty_completion(monkeypatch):
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return b'{"choices": []}'
    monkeypatch.setattr("Sentry.clients.urlopen", lambda *a, **kw: Response())
    with pytest.raises(RuntimeError, match="no chat completion"):
        ChatCompletionsClient("http://localhost/v1", "test")("x", temperature=0, max_tokens=10)
