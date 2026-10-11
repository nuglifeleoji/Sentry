"""Explicit, dependency-free HTTP adapters for Sentry's text model contract.

Construction performs no network I/O. Credentials are resolved only by the
factory, never stored in serializable ProviderConfig. No adapter silently
changes decoding settings or retries a malformed model answer.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from http.client import HTTPException
from typing import ClassVar
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


_PROVIDER_DEFAULTS = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "openai_compatible": (None, None),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "openai_responses": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "anthropic": ("https://api.anthropic.com/v1", "ANTHROPIC_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta", "GEMINI_API_KEY"),
}
_TRANSIENT_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


class ModelOutputTruncatedError(RuntimeError):
    """Rejected text stopped at the output limit, retained only for diagnostics.

    The exception message and repr never include ``partial_text``. Callers may
    record an unexecuted failed action and request normal recovery, but this
    exception does not authorize executing or accepting the partial response.
    """

    def __init__(self, partial_text: str, *, finish_reason: str) -> None:
        super().__init__("Model endpoint returned an incomplete completion (output token limit reached)")
        self.partial_text = partial_text
        self.finish_reason = finish_reason


def _validate_base_url(value: str) -> None:
    if not isinstance(value, str) or re.search(r"[\x00-\x20\x7f]", value):
        raise ValueError("base_url must be an HTTP(S) API base URL without whitespace")
    try:
        parts = urlsplit(value)
        valid = parts.scheme in {"http", "https"} and bool(parts.hostname)
        _ = parts.port
    except ValueError:
        raise ValueError("base_url must be a valid HTTP(S) API base URL") from None
    if not valid:
        raise ValueError("base_url must be an HTTP(S) API base URL")
    if parts.username is not None or parts.password is not None or parts.query or parts.fragment:
        raise ValueError("base_url cannot include credentials, query parameters, or fragments")


def _finite_number(value) -> bool:
    if type(value) not in {int, float}:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _validate_options(model: str, timeout: float, json_mode: str, max_retries: int) -> None:
    if not isinstance(model, str) or not model or any(c.isspace() for c in model):
        raise ValueError("Supply the exact nonempty model identifier served by your endpoint")
    if not _finite_number(timeout) or timeout <= 0:
        raise ValueError("timeout must be a finite positive number of seconds")
    if json_mode not in ("native", "prompt"):
        raise ValueError("json_mode must be 'native' or the explicit fallback 'prompt'")
    if type(max_retries) is not int or not 0 <= max_retries <= 5:
        raise ValueError("max_retries must be an integer between 0 and 5")


@dataclass(frozen=True)
class ProviderConfig:
    """Serializable connection settings; model choice is an experimental variable.

    ``api_key_env=None`` selects the provider's standard environment variable,
    except for openai_compatible where it means no authentication. Supply an
    environment-variable name explicitly for an authenticated compatible server.
    """

    provider: str = "openai"
    model: str = "Qwen3.5-9B"
    base_url: str | None = None
    api_key_env: str | None = None
    timeout: float = 120.0
    json_mode: str = "native"
    max_retries: int = 0
    reasoning_enabled: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.provider, str) or self.provider not in _PROVIDER_DEFAULTS:
            raise ValueError("Unsupported model provider")
        _validate_options(self.model, self.timeout, self.json_mode, self.max_retries)
        if self.base_url is not None:
            _validate_base_url(self.base_url)
        if self.provider == "openai_compatible" and self.base_url is None:
            raise ValueError("openai_compatible requires an explicit base_url")
        if self.api_key_env is not None and (
            not isinstance(self.api_key_env, str)
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env) is None
        ):
            raise ValueError("api_key_env must name an environment variable, not contain a key")
        if self.provider == "anthropic" and self.json_mode != "prompt":
            raise ValueError("The Anthropic adapter requires explicit json_mode='prompt'")

        if self.reasoning_enabled is not None:
            if type(self.reasoning_enabled) is not bool:
                raise ValueError("reasoning_enabled must be a boolean or null")
            if self.provider != "openrouter":
                raise ValueError("reasoning_enabled is supported only by the OpenRouter adapter")

    @classmethod
    def from_env(cls) -> "ProviderConfig":
        """Read SENTRY_* connection variables; never read a .env file implicitly."""
        model = os.environ.get("SENTRY_MODEL", "")
        if not model:
            raise ValueError("Set SENTRY_MODEL to the exact model served by the endpoint")
        base_url = os.environ.get("SENTRY_API_BASE") or None
        provider = os.environ.get("SENTRY_PROVIDER") or (
            "openai_compatible" if base_url else "openai"
        )
        key_env = os.environ.get("SENTRY_API_KEY_ENV") or (
            "SENTRY_API_KEY" if os.environ.get("SENTRY_API_KEY") else None
        )
        try:
            timeout = float(os.environ.get("SENTRY_TIMEOUT", "120"))
            max_retries = int(os.environ.get("SENTRY_MAX_RETRIES", "0"))
        except ValueError:
            raise ValueError("SENTRY_TIMEOUT and SENTRY_MAX_RETRIES must be numeric") from None
        reasoning = os.environ.get("SENTRY_REASONING_ENABLED")
        if reasoning is not None and reasoning not in {"true", "false"}:
            raise ValueError("SENTRY_REASONING_ENABLED must be 'true' or 'false' when set")
        return cls(
            provider=provider, model=model, base_url=base_url, api_key_env=key_env,
            timeout=timeout, json_mode=os.environ.get("SENTRY_JSON_MODE", "native"),
            max_retries=max_retries,
            reasoning_enabled=None if reasoning is None else reasoning == "true",
        )


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # urllib can otherwise forward Authorization and API-key headers.
        return None


# Keep one mockable transport entry point for all provider clients.
urlopen = build_opener(_NoRedirect()).open


def _json_object(text: str | bytes) -> dict:
    def reject_constant(value):
        raise ValueError("Non-finite JSON number")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON object key")
            result[key] = value
        return result

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Non-finite JSON number")
        return number

    value = json.loads(text, parse_constant=reject_constant, parse_float=finite_float,
                       object_pairs_hook=unique_object)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


@dataclass
class _HTTPModelClient:
    base_url: str
    model: str
    api_key: str = field(default="", repr=False)
    timeout: float = 120.0
    json_mode: str = "native"
    max_retries: int = 0
    last_usage: dict = field(default_factory=dict, init=False)
    last_response_metadata: dict = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        _validate_base_url(self.base_url)
        _validate_options(self.model, self.timeout, self.json_mode, self.max_retries)
        if not isinstance(self.api_key, str) or not self.api_key.isascii() or any(
            c.isspace() or ord(c) < 32 or ord(c) == 127 for c in self.api_key
        ):
            raise ValueError("api_key must be an ASCII credential without whitespace")

    def __call__(
        self, prompt: str, *, temperature: float, max_tokens: int,
        response_format: dict | None = None,
    ) -> str:
        self.last_usage = {}
        self.last_response_metadata = {}
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a nonempty string")
        if not _finite_number(temperature) or not 0 <= temperature <= 2:
            raise ValueError("temperature must be a finite number between 0 and 2")
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if response_format is not None and response_format != {"type": "json_object"}:
            raise ValueError("Only response_format={'type': 'json_object'} is supported")
        structured = response_format is not None
        if structured and (self.json_mode == "prompt" or "JSON" not in prompt):
            prompt += "\n\nReturn exactly one valid JSON object, without Markdown fences or surrounding text."
        data = self._generate(prompt, temperature, max_tokens, structured)
        self.last_response_metadata = {
            key: data[key] for key in ("id", "model", "provider")
            if isinstance(data.get(key), str) and data[key]
        }
        usage = data.get("usage", data.get("usageMetadata"))
        if isinstance(usage, dict):
            self.last_usage = dict(usage)
        text = self._text(data)
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Model endpoint returned empty or non-text content")
        if structured:
            try:
                _json_object(text)
            except (ValueError, UnicodeError, RecursionError):
                raise RuntimeError("Model endpoint returned an invalid JSON object") from None
        return text

    def _post(self, suffix: str, payload: dict, auth_headers: dict) -> dict:
        request = Request(
            self.base_url.rstrip("/") + suffix,
            data=json.dumps(payload, allow_nan=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json", **auth_headers},
            method="POST",
        )
        for attempt in range(self.max_retries + 1):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    data = _json_object(response.read())
                if data.get("error"):
                    raise RuntimeError("Model endpoint returned an API error")
                return data
            except HTTPError as exc:
                status = exc.code
                exc.close()
                if status not in _TRANSIENT_STATUS or attempt == self.max_retries:
                    detail = "; check model support for the explicit decoding and JSON settings" if status == 400 else ""
                    raise RuntimeError(f"Model endpoint returned HTTP {status}{detail}") from None
            except (URLError, OSError, HTTPException):
                if attempt == self.max_retries:
                    raise RuntimeError("Model endpoint could not be reached or timed out") from None
            except (ValueError, UnicodeError, RecursionError):
                raise RuntimeError("Model endpoint did not return a valid JSON response object") from None
            time.sleep(min(2 ** attempt, 8))
        raise AssertionError("Unreachable retry state")

    def _bearer_headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _generate(self, prompt, temperature, max_tokens, structured):
        raise NotImplementedError

    def _text(self, data):
        raise NotImplementedError


@dataclass
class ChatCompletionsClient(_HTTPModelClient):
    """OpenAI-compatible Chat Completions, including local vLLM/Ollama servers."""

    _token_limit_field: ClassVar[str] = "max_tokens"

    def _generate(self, prompt, temperature, max_tokens, structured):
        payload = {
            "model": self.model, "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature, self._token_limit_field: max_tokens, "stream": False,
        }
        if structured and self.json_mode == "native":
            payload["response_format"] = {"type": "json_object"}
        return self._post("/chat/completions", payload, self._bearer_headers())

    def _text(self, data):
        try:
            choices = data["choices"]
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError
            choice = choices[0]
            message = choice["message"]
            if not isinstance(message, dict):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            raise RuntimeError("Model endpoint returned no chat completion content") from None
        finish_reason = choice.get("finish_reason")
        if isinstance(finish_reason, str) and finish_reason:
            self.last_response_metadata["finish_reason"] = finish_reason
        if message.get("refusal"):
            raise RuntimeError("Model endpoint refused the request")
        if message.get("tool_calls") or message.get("function_call"):
            raise RuntimeError("Model endpoint returned an incomplete or non-text completion")
        content = message.get("content")
        if finish_reason == "length" and isinstance(content, str):
            raise ModelOutputTruncatedError(content, finish_reason=finish_reason) from None
        if finish_reason != "stop":
            raise RuntimeError("Model endpoint returned an incomplete or non-text completion")
        return content


class OpenAIChatCompletionsClient(ChatCompletionsClient):
    """Native OpenAI uses the current max_completion_tokens parameter."""

    _token_limit_field = "max_completion_tokens"


@dataclass
class OpenRouterClient(ChatCompletionsClient):
    """OpenRouter routing requires support for every explicit request parameter."""

    reasoning_enabled: bool | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.reasoning_enabled is not None and type(self.reasoning_enabled) is not bool:
            raise ValueError("reasoning_enabled must be a boolean or null")

    def _generate(self, prompt, temperature, max_tokens, structured):
        payload = {
            "model": self.model, "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature, "max_tokens": max_tokens, "stream": False,
            "provider": {"require_parameters": True},
        }
        if self.reasoning_enabled is not None:
            payload["reasoning"] = {"enabled": self.reasoning_enabled}
        if structured and self.json_mode == "native":
            payload["response_format"] = {"type": "json_object"}
        return self._post("/chat/completions", payload, self._bearer_headers())


class OpenAIResponsesClient(_HTTPModelClient):
    def _generate(self, prompt, temperature, max_tokens, structured):
        payload = {
            "model": self.model, "input": [{"role": "user", "content": prompt}],
            "temperature": temperature, "max_output_tokens": max_tokens,
            "stream": False, "store": False,
        }
        if structured and self.json_mode == "native":
            payload["text"] = {"format": {"type": "json_object"}}
        return self._post("/responses", payload, self._bearer_headers())

    def _text(self, data):
        details = data.get("incomplete_details")
        truncated = (data.get("status") == "incomplete" and isinstance(details, dict)
                     and details.get("reason") == "max_output_tokens")
        if not truncated and (data.get("status") != "completed" or details):
            raise RuntimeError("Model endpoint returned an incomplete response")
        output = data.get("output")
        if not isinstance(output, list):
            raise RuntimeError("Model endpoint returned no response content")
        parts = []
        for item in output:
            if not isinstance(item, dict):
                raise RuntimeError("Model endpoint returned malformed response content")
            if item.get("type") == "reasoning":
                continue
            if (item.get("type") != "message" or item.get("role") != "assistant"
                    or item.get("status") not in (("completed", "incomplete") if truncated else ("completed",))):
                raise RuntimeError("Model endpoint returned an incomplete or non-text response")
            parts.extend(_text_blocks(item.get("content"), "output_text"))
        if truncated:
            self.last_response_metadata["finish_reason"] = "max_output_tokens"
            raise ModelOutputTruncatedError("".join(parts), finish_reason="max_output_tokens")
        return "".join(parts)


class AnthropicMessagesClient(_HTTPModelClient):
    def __post_init__(self) -> None:
        super().__post_init__()
        if self.json_mode != "prompt":
            raise ValueError("The Anthropic adapter requires explicit json_mode='prompt'")

    def _generate(self, prompt, temperature, max_tokens, structured):
        if temperature > 1:
            raise ValueError("Anthropic temperature must be between 0 and 1")
        payload = {
            "model": self.model, "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature, "max_tokens": max_tokens, "stream": False,
        }
        headers = {"anthropic-version": "2023-06-01"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return self._post("/messages", payload, headers)

    def _text(self, data):
        details = data.get("stop_details")
        if data.get("stop_reason") == "refusal" or isinstance(details, dict) and details.get("type") == "refusal":
            raise RuntimeError("Model endpoint refused the request")
        stop_reason = data.get("stop_reason")
        if stop_reason not in ("end_turn", "max_tokens"):
            raise RuntimeError("Model endpoint returned an incomplete or non-text message")
        text = "".join(_text_blocks(data.get("content"), "text"))
        if stop_reason == "max_tokens":
            self.last_response_metadata["finish_reason"] = stop_reason
            raise ModelOutputTruncatedError(text, finish_reason=stop_reason)
        return text


class GeminiClient(_HTTPModelClient):
    def _generate(self, prompt, temperature, max_tokens, structured):
        generation = {"temperature": temperature, "maxOutputTokens": max_tokens, "candidateCount": 1}
        if structured and self.json_mode == "native":
            generation["responseMimeType"] = "application/json"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": generation,
        }
        model = self.model.removeprefix("models/")
        if not model:
            raise ValueError("Gemini requires a model identifier after models/")
        headers = {"x-goog-api-key": self.api_key} if self.api_key else {}
        return self._post(f"/models/{quote(model, safe='')}:generateContent", payload, headers)

    def _text(self, data):
        feedback = data.get("promptFeedback")
        if isinstance(feedback, dict) and feedback.get("blockReason"):
            raise RuntimeError("Model endpoint blocked the prompt")
        candidates = data.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 1 or not isinstance(candidates[0], dict):
            raise RuntimeError("Model endpoint returned no candidate content")
        candidate = candidates[0]
        finish_reason = candidate.get("finishReason")
        if finish_reason not in ("STOP", "MAX_TOKENS"):
            raise RuntimeError("Model endpoint returned an incomplete or blocked candidate")
        content = candidate.get("content")
        if not isinstance(content, dict) or not isinstance(content.get("parts"), list):
            raise RuntimeError("Model endpoint returned no candidate text")
        parts = []
        for block in content["parts"]:
            if not isinstance(block, dict):
                raise RuntimeError("Model endpoint returned malformed candidate text")
            if block.get("thought") is True:
                continue
            if not isinstance(block.get("text"), str):
                raise RuntimeError("Model endpoint returned non-text candidate content")
            parts.append(block["text"])
        if finish_reason == "MAX_TOKENS":
            self.last_response_metadata["finish_reason"] = finish_reason
            raise ModelOutputTruncatedError("".join(parts), finish_reason=finish_reason)
        return "".join(parts)


def _text_blocks(blocks, text_type: str) -> list[str]:
    if not isinstance(blocks, list):
        raise RuntimeError("Model endpoint returned malformed text blocks")
    result = []
    for block in blocks:
        if not isinstance(block, dict):
            raise RuntimeError("Model endpoint returned malformed text blocks")
        if block.get("type") == "refusal":
            raise RuntimeError("Model endpoint refused the request")
        if block.get("type") != text_type or not isinstance(block.get("text"), str):
            raise RuntimeError("Model endpoint returned non-text content")
        result.append(block["text"])
    return result


def create_client(config: ProviderConfig) -> _HTTPModelClient:
    """Resolve environment credentials and construct an adapter without I/O."""
    if not isinstance(config, ProviderConfig):
        raise TypeError("create_client requires ProviderConfig")
    base_url, default_key_env = _PROVIDER_DEFAULTS[config.provider]
    key_env = config.api_key_env or default_key_env
    api_key = os.environ.get(key_env, "") if key_env else ""
    if key_env and not api_key:
        raise ValueError(f"Set the API key environment variable {key_env}")
    clients = {
        "openai": OpenAIChatCompletionsClient,
        "openai_compatible": ChatCompletionsClient,
        "openrouter": OpenRouterClient,
        "openai_responses": OpenAIResponsesClient,
        "anthropic": AnthropicMessagesClient,
        "gemini": GeminiClient,
    }
    extra = {"reasoning_enabled": config.reasoning_enabled} if config.provider == "openrouter" else {}
    return clients[config.provider](
        base_url=config.base_url or base_url, model=config.model, api_key=api_key,
        timeout=config.timeout, json_mode=config.json_mode, max_retries=config.max_retries,
        **extra,
    )
