from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from .Sentry import Sentry
from .clients import create_client
from .config import ModelConfig, SentryConfig, TaskAgentConfig
from .detector import render_trajectory
from .models import AgentStep, RescuePrompt


TextJudge = Callable[[str], str]


class SentryRunner:
    """Convenience runner that uses one model for every Sentry component.

    Event capture is off by default. Hosts opting in must regularly call
    ``guard.drain_runtime_events()``, including after ``finalize()``.
    """

    def __init__(self, model: Any | None = None, config: SentryConfig | None = None,
                 *, capture_runtime_events: bool = False):
        self.config = config or SentryConfig()
        if model is None:
            if self.config.api is None:
                raise ValueError("Supply a model client or configure api in SentryConfig")
            model = create_client(self.config.api)
        elif self.config.api is not None:
            raise ValueError("Supply either a model client or config.api, so no API settings are ignored")
        served_model = getattr(model, "model", None)
        if isinstance(served_model, str):
            if config is None:
                self.config = replace(self.config, model=replace(self.config.model, model=served_model))
            elif served_model != self.config.model.model:
                raise ValueError("The model client identifier differs from config.model.model")
        self.model = model
        self._structured = model_to_text_judge(model, self.config.model, structured=True)
        self._repair = model_to_text_judge(model, self.config.task_agent, structured=False)
        self.guard = Sentry(
            self.config,
            guard_judge_callback=self._structured,
            verifier_judge_callback=self._structured,
            updater_judge_callback=self._structured,
            capture_runtime_events=capture_runtime_events,
        )

    @classmethod
    def from_yaml(cls, path: str | Path, *, capture_runtime_events: bool = False) -> "SentryRunner":
        """Construct the configured shared client without making an API request."""
        return cls(config=SentryConfig.from_yaml(path), capture_runtime_events=capture_runtime_events)

    def start_task(self, *, objective: str, action_schema: str) -> None:
        self.guard.start_task(objective=objective, action_schema=action_schema)

    def step(self, step: AgentStep, *, terminal: bool = False) -> RescuePrompt | None:
        """Observe one cycle, then offer guidance if the host can continue.

        ``terminal=True`` suppresses a new diagnosis or repair on the last cycle.
        Pending hard verification or H-step soft verification still observes that
        cycle normally. Call ``finalize()`` afterwards to end the task; terminal
        status denotes execution ending, not evaluator success.
        """
        if type(terminal) is not bool:
            raise ValueError("terminal must be a boolean")
        self.guard.observe_step(step)
        return None if terminal else self.guard.maybe_rescue()

    def generate_repair(self, prompt: RescuePrompt) -> str:
        """Continue with task-agent decoding settings (paper Section 4.3).

        This convenience path gives a stateless callback the task and local
        trajectory. Applications with an agent conversation should append
        ``prompt.prompt_text`` there and generate the next action in that context.
        Neither path injects the full playbook.
        """

        task = self.guard.task
        if task is None:
            raise RuntimeError("Call start_task before generating a repair")
        context = (
            f"Task objective.\n{task.objective}\n\n"
            f"Recent task-agent trajectory.\n{render_trajectory(self.guard.window.steps)}\n\n"
        )
        return self._repair(context + prompt.prompt_text)

    def finalize(self) -> None:
        self.guard.finalize_task()


def model_to_text_judge(
    model: Any,
    config: ModelConfig | TaskAgentConfig | None = None,
    *,
    structured: bool = True,
) -> TextJudge:
    """Bind the paper's deterministic generation budget to a model client.

    Supported clients are callable or expose ``generate``, ``invoke``,
    ``complete``, or ``predict`` and must accept the explicit decoding kwargs.
    Refusing those kwargs is an error rather than a silent protocol deviation.
    """

    cfg = config or ModelConfig()
    method = model if callable(model) else None
    if method is None:
        for name in ("generate", "invoke", "complete", "predict"):
            candidate = getattr(model, name, None)
            if callable(candidate):
                method = candidate
                break
    if method is None:
        raise TypeError(
            "Model must be callable or expose generate/invoke/complete/predict."
        )

    kwargs: dict[str, Any] = {
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }
    if structured and cfg.structured_outputs:
        kwargs["response_format"] = {"type": "json_object"}

    def call(prompt: str) -> str:
        # Preserve exceptions from the client itself; a TypeError can indicate
        # a client bug and must not be misreported as an unsupported signature.
        return _response_to_text(method(prompt, **kwargs))

    return call


def _field(value: Any, name: str, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _response_to_text(response: Any) -> str:
    if isinstance(response, str):
        if not response.strip():
            raise TypeError("Model response has empty text content")
        return response
    choices = _field(response, "choices")
    if choices is not None:
        if not isinstance(choices, (list, tuple)) or len(choices) != 1:
            raise TypeError("Model response must contain exactly one chat completion")
        choice = choices[0]
        message = _field(choice, "message")
        if _field(choice, "finish_reason") != "stop":
            raise TypeError("Model response is incomplete or not a text completion")
        if message is not None:
            if _field(message, "refusal") or _field(message, "tool_calls") or _field(message, "function_call"):
                raise TypeError("Model response is a refusal or tool call")
            return _content_to_text(_field(message, "content"))
        return _content_to_text(_field(choice, "text"))
    # SDK Responses objects expose output_text, including for partial results.
    # If a completion status is supplied, it must explicitly indicate success.
    status = _field(response, "status")
    if (status is not None and status != "completed") or _field(response, "incomplete_details") or _field(response, "error"):
        raise TypeError("Model response is incomplete or failed")
    if _field(response, "refusal") or _field(response, "tool_calls"):
        raise TypeError("Model response is a refusal or tool call")
    stop_reason = _field(response, "stop_reason")
    if stop_reason is not None and stop_reason != "end_turn":
        raise TypeError("Model response is incomplete or not a text completion")
    output = _field(response, "output")
    if output is not None:
        if status != "completed" or not isinstance(output, (list, tuple)):
            raise TypeError("Model response is incomplete or malformed")
        parts = []
        for item in output:
            if _field(item, "type") == "reasoning":
                continue
            if (_field(item, "type") != "message" or _field(item, "role") != "assistant"
                    or _field(item, "status") != "completed"):
                raise TypeError("Model response contains incomplete or non-text output")
            parts.append(_content_to_text(_field(item, "content")))
        return _content_to_text("".join(parts))
    for key in ("response_metadata", "metadata"):
        metadata = _field(response, key)
        if isinstance(metadata, dict):
            reason = metadata.get("finish_reason")
            if reason is not None and (not isinstance(reason, str) or reason not in {"stop", "STOP", "end_turn"}):
                raise TypeError("Model response is incomplete or not a text completion")
    for key in ("text", "content", "output_text"):
        value = _field(response, key)
        if value:
            return _content_to_text(value)
    raise TypeError("Model response has no supported text content")


def _content_to_text(value: Any) -> str:
    if isinstance(value, str):
        if not value.strip():
            raise TypeError("Model response has empty text content")
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            kind = _field(item, "type")
            text = _field(item, "text")
            if kind not in {"text", "output_text"} or not isinstance(text, str):
                raise TypeError("Model response contains unsupported non-text content")
            parts.append(text)
        result = "".join(parts)
        if result.strip():
            return result
    raise TypeError("Model response has no supported text content")
