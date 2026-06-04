from __future__ import annotations

from typing import Any, Callable, Iterable

from .Sentry import Sentry
from .config import SentryConfig
from .models import AgentStep, RescuePrompt


TextJudge = Callable[[str], str]


class SentryRunner:
    def __init__(
        self,
        config: SentryConfig | None = None,
        *,
        model: Any | None = None,
        detector: TextJudge | None = None,
        verifier: TextJudge | None = None,
    ):
        if model is not None:
            judge = model_to_text_judge(model)
            detector = detector or judge
            verifier = verifier or judge
        if detector is None or verifier is None:
            raise ValueError("Provide model or both detector and verifier callbacks.")
        self.guard = Sentry(
            config or SentryConfig(),
            guard_judge_callback=detector,
            verifier_judge_callback=verifier,
        )

    def step(self, step: AgentStep) -> RescuePrompt | None:
        self.guard.observe_step(step)
        return self.guard.maybe_rescue()

    def run_steps(self, steps: Iterable[AgentStep]) -> list[RescuePrompt]:
        prompts: list[RescuePrompt] = []
        for step in steps:
            rescue = self.step(step)
            if rescue is not None:
                prompts.append(rescue)
        return prompts

    def finalize(
        self,
        *,
        final_task_score: float | None = None,
        task_success: bool | None = None,
    ) -> None:
        self.guard.finalize_task(
            final_task_score=final_task_score,
            task_success=task_success,
        )


def model_to_text_judge(model: Any) -> TextJudge:
    if callable(model):
        return lambda prompt: _response_to_text(model(prompt))

    for method_name in ("generate", "invoke", "complete", "predict"):
        method = getattr(model, method_name, None)
        if callable(method):
            return lambda prompt, method=method: _response_to_text(method(prompt))

    raise TypeError(
        "Model must be callable or expose one of: generate, invoke, complete, predict."
    )


def _response_to_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        return _dict_response_to_text(response)
    for attr in ("text", "content", "output_text"):
        value = getattr(response, attr, None)
        if value:
            return _content_to_text(value)
    return str(response)


def _dict_response_to_text(response: dict) -> str:
    for key in ("text", "content", "output_text"):
        value = response.get(key)
        if value:
            return _content_to_text(value)
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message") or {}
            if isinstance(message, dict) and message.get("content"):
                return _content_to_text(message["content"])
            if first.get("text"):
                return _content_to_text(first["text"])
    return str(response)


def _content_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(value)
