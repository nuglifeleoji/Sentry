from __future__ import annotations

from collections.abc import Callable

from ._structured import json_object, nonempty_text, string_list
from .config import SentryConfig
from .detector import render_trajectory
from .models import AgentStep, EscapeResult, RescueEvent, TaskContext
from .taxonomy import FailureType, resolution_criteria_for


class EscapeVerifier:
    def __init__(
        self,
        judge_callback: Callable[[str], str] | None,
        config: SentryConfig | None = None,
    ):
        self.judge_callback = judge_callback
        self.config = config or SentryConfig()

    def verify(self, event: RescueEvent, task: TaskContext) -> EscapeResult:
        if event.is_hard_repair:
            raise ValueError("Hard repairs use immediate execution verification")
        horizon = self.config.trace.post_rescue_horizon
        if len(event.post_rescue_steps) != horizon:
            raise ValueError(f"Soft repair must be verified after exactly H={horizon} cycles")
        if self.judge_callback is None:
            raise RuntimeError("Soft-repair verification requires the Sentry model callback")
        raw = self.judge_callback(build_verifier_prompt(event, task, self.config))
        return parse_verifier_response(raw, event)


def verify_hard_repair(event: RescueEvent, corrected_step: AgentStep) -> EscapeResult:
    corrected_step.validate()
    action = corrected_step.action
    resolved = bool(
        action.parsed_ok
        and action.schema_valid
        and action.accepted_by_environment is True
    )
    if resolved:
        evidence = [
            "The corrected action satisfied the schema and was accepted by the environment."
        ]
        justification = "Immediate hard-repair verification passed."
    else:
        evidence = [
            action.parser_error
            or action.schema_error
            or "The corrected action was not explicitly accepted by the environment."
        ]
        justification = "Immediate hard-repair verification failed."
    return EscapeResult(
        escaped=resolved,
        failure_type=FailureType.ACTION_VALIDITY_FAILURE,
        evidence=evidence,
        justification=justification,
        resolved_labels=[],
        unresolved_labels=[] if resolved else [FailureType.ACTION_VALIDITY_FAILURE.value],
    )


def build_verifier_prompt(
    event: RescueEvent,
    task: TaskContext,
    config: SentryConfig | None = None,
) -> str:
    cfg = config or SentryConfig()
    horizon = cfg.trace.post_rescue_horizon
    if len(event.post_rescue_steps) != horizon:
        raise ValueError(f"Expected exactly {horizon} post-intervention cycles")
    labels = ", ".join(event.detection.retrieval_labels)
    evidence = "\n".join(f"- {item}" for item in event.detection.local_evidence)
    criteria = resolution_criteria_for(
        event.failure_type, event.detection.retrieval_labels
    )
    criteria_text = "\n".join(f"- {name}: {text}" for name, text in criteria.items())
    return f"""[SENTRY ESCAPE VERIFIER]
Role.
You are the Sentry escape verifier. Your job is to judge whether the rescue intervention fixed the exact failure that triggered the rescue. Judge the specific retrieval labels and resolution criteria, not generic task success alone.

Task objective.
{task.objective}

Triggering failure.
Failure type: {event.failure_type.value}
Retrieval labels: {labels}
Trigger evidence:
{evidence}

Recovery trajectory.
{render_trajectory(event.post_rescue_steps)}

Decision criteria.
A failure is resolved if the agent stops the diagnosed behavior and resumes valid, task-relevant, observation-grounded progress. A failure is not resolved if the same failure pattern persists, the agent drifts further, or the recovery introduces unsupported assumptions.
Label-specific criteria:
{criteria_text}

You have no benchmark reward or final task-success label. Do not infer either.

Output format.
Return only this JSON object with fixed fields:
{{
  "status": "resolved | unresolved",
  "justification": "short justification",
  "evidence": ["evidence from the H-step recovery trajectory"]
}}
"""


def parse_verifier_response(raw: str, event: RescueEvent) -> EscapeResult:
    data = json_object(raw, {"status", "justification", "evidence"})
    if data is None:
        return _unresolved(event, raw, "Verifier output must match the fixed JSON schema.")
    status = data["status"]
    if not isinstance(status, str) or status not in {"resolved", "unresolved"}:
        return _unresolved(event, raw, "Verifier status was not resolved or unresolved.")
    if not string_list(data["evidence"], allow_empty=False) or not nonempty_text(data["justification"]):
        return _unresolved(event, raw, "Verifier evidence and justification must be nonempty strings.")
    evidence = [item.strip() for item in data["evidence"]]
    justification = data["justification"].strip()
    resolved = status == "resolved"
    labels = list(event.detection.retrieval_labels)
    return EscapeResult(
        escaped=resolved,
        failure_type=event.failure_type,
        evidence=evidence or ["No grounded verifier evidence was returned."],
        justification=justification or "Verifier output lacked a justification.",
        raw_response=raw,
        resolved_labels=labels if resolved else [],
        unresolved_labels=[] if resolved else labels,
    )


def _unresolved(event: RescueEvent, raw: str, reason: str) -> EscapeResult:
    return EscapeResult(
        escaped=False,
        failure_type=event.failure_type,
        evidence=[reason],
        justification=reason,
        raw_response=raw,
        unresolved_labels=list(event.detection.retrieval_labels),
    )
