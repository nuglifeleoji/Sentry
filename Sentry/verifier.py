from __future__ import annotations

import json
import re
from typing import Callable

from .config import SentryConfig
from .models import EscapeResult, RescueEvent
from .taxonomy import (
    FailureType,
    normalize_retrieval_label,
    resolution_criteria_for,
)


class EscapeVerifier:
    def __init__(
        self,
        judge_callback: Callable[[str], str] | None = None,
        config: SentryConfig | None = None,
    ):
        self.judge_callback = judge_callback
        self.config = config or SentryConfig()

    def verify(self, event: RescueEvent) -> EscapeResult:
        if self.judge_callback is None:
            raise RuntimeError(
                "EscapeVerifier requires a verifier judge callback; no rule fallback is available."
            )
        prompt = build_verifier_prompt(event, self.config)
        raw = self.judge_callback(prompt)
        return parse_verifier_response(raw, event)


def build_verifier_prompt(
    event: RescueEvent,
    config: SentryConfig | None = None,
) -> str:
    cfg = config or SentryConfig()
    objective = _task_objective_from_event(event) or "No explicit task objective was provided."
    labels = list(event.detection.retrieval_labels or [])
    criteria = _render_resolution_criteria(event.failure_type, labels)
    evidence = "\n".join(
        f"- {str(item).strip()}"
        for item in event.detection.local_evidence
        if item is not None and str(item).strip()
    ) or "- none"
    rescue_text = str(event.rescue_prompt.prompt_text or "")
    if len(rescue_text) > 2500:
        rescue_text = rescue_text[:2500] + "..."
    return f"""You are the Sentry escape verifier.

Your job is to judge whether the rescue intervention fixed the exact failure
that triggered the rescue. Judge the specific retrieval labels and resolution
criteria, not generic task success alone.

Task objective:
{objective}

Detected failure:
- failure_type: {event.failure_type.value}
- retrieval_labels: {_render_labels(labels)}
- detector evidence:
{evidence}
- detector rationale: {_detection_rationale(event)}

Resolution criteria:
{criteria}

Pre-rescue trajectory:
{_render_steps(event.pre_rescue_window.steps, cfg)}

Injected rescue:
{rescue_text}

Post-rescue trajectory:
{_render_steps(event.post_rescue_steps, cfg)}

Return only compact JSON:
{{
  "escaped": true,
  "helped_task": true,
  "resolved_labels": ["repetition_or_looping"],
  "unresolved_labels": [],
  "confidence": 0.8,
  "evidence": ["short concrete evidence from the post-rescue trajectory"],
  "reflection": "When [specific trigger pattern], [specific repair principle grounded in the post-rescue trajectory]."
}}

Rules:
- escaped is true only if the triggering failure is resolved in the post-rescue trajectory.
- helped_task is true only if the repair plausibly helps the original task, not merely changes behavior.
- For soft repairs, resolved_labels must be a subset of the detected retrieval_labels.
- If no detected label was resolved, set escaped=false.
- reflection must be reusable, label-specific, and grounded in the before/after trajectory.
- Make reflection concrete enough for the same label: name the trigger pattern and the repair principle.
- Avoid generic advice such as "click a relevant item" unless the trajectory shows what made the item relevant.
- Do not invent observations, tool results, or task completion.
"""


def parse_verifier_response(raw: str, event: RescueEvent) -> EscapeResult:
    text = str(raw or "").strip()
    data = {}
    try:
        parsed = json.loads(text)
        data = parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match is not None:
            try:
                parsed = json.loads(match.group(0))
                data = parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                data = {}

    detected_labels = set()
    for item in event.detection.retrieval_labels or []:
        label = normalize_retrieval_label(item, event.failure_type)
        if label is not None:
            detected_labels.add(label)

    resolved_labels = []
    raw_resolved = data.get("resolved_labels")
    resolved_items = raw_resolved if isinstance(raw_resolved, list) else [raw_resolved]
    for item in resolved_items:
        label = normalize_retrieval_label(item, event.failure_type)
        if label is not None and label not in resolved_labels:
            resolved_labels.append(label)

    unresolved_labels = []
    raw_unresolved = data.get("unresolved_labels")
    unresolved_items = raw_unresolved if isinstance(raw_unresolved, list) else [raw_unresolved]
    for item in unresolved_items:
        label = normalize_retrieval_label(item, event.failure_type)
        if label is not None and label not in unresolved_labels:
            unresolved_labels.append(label)

    if detected_labels:
        resolved_labels = [label for label in resolved_labels if label in detected_labels]
        unresolved_labels = [label for label in unresolved_labels if label in detected_labels]
    if not unresolved_labels and detected_labels:
        unresolved_labels = [label for label in detected_labels if label not in set(resolved_labels)]

    escaped = bool(data.get("escaped")) and (
        event.failure_type == FailureType.ACTION_VALIDITY_FAILURE or bool(resolved_labels)
    )
    helped_task = bool(data.get("helped_task")) and escaped
    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence"))))
    except (TypeError, ValueError):
        confidence = 0.0
    raw_evidence = data.get("evidence")
    if isinstance(raw_evidence, list):
        evidence = [str(item).strip() for item in raw_evidence if str(item).strip()]
    else:
        evidence = [str(raw_evidence).strip()] if str(raw_evidence or "").strip() else []
    if not evidence:
        evidence = ["Verifier judge returned no evidence."]
    reflection = str(data.get("reflection") or "").strip() or None
    before = float(event.detection.confidence)
    after = before * (1.0 - confidence) if escaped else before
    task_progress_delta = _task_progress_delta(event)
    return EscapeResult(
        escaped=escaped,
        failure_type=event.failure_type,
        confidence=confidence,
        evidence=evidence,
        failure_score_before=before,
        failure_score_after=after,
        progress_delta=float(task_progress_delta or 0.0),
        helped_task=helped_task,
        task_progress_delta=task_progress_delta,
        resolved_labels=resolved_labels,
        unresolved_labels=unresolved_labels,
        reflection=reflection,
        raw_response=raw,
    )


def _render_resolution_criteria(failure_type: FailureType, labels: list[str]) -> str:
    criteria = resolution_criteria_for(failure_type, labels)
    if not criteria:
        return "- No label-specific resolution criteria are available."
    return "\n".join(f"- {label}: {criterion}" for label, criterion in criteria.items())


def _render_steps(steps, config: SentryConfig) -> str:
    lines: list[str] = []
    for step in steps:
        reasoning = " ".join(str(step.reasoning or "").split())
        if len(reasoning) > 500:
            reasoning = reasoning[:500] + "..."
        action = step.action.raw or step.action.tool_name or ""
        observation = " ".join(str(step.observation or "").split())
        if len(observation) > config.guard_judge.max_observation_chars:
            observation = observation[: config.guard_judge.max_observation_chars] + "..."
        metadata = _compact_metadata(step.metadata or {})
        meta = f" metadata={metadata}" if metadata else ""
        lines.append(
            f"- step={step.step_id} reasoning={reasoning!r} action={action!r} "
            f"parsed={step.action.parsed_ok} schema_valid={step.action.schema_valid} "
            f"progress={step.task_progress_score} observation={observation!r}{meta}"
        )
    return "\n".join(lines) or "- none"


def _compact_metadata(metadata: dict) -> dict:
    keys = (
        "failure_type_hint",
        "sentry_failure_type",
        "task_completed",
        "episode_done",
        "finish_executed",
        "commit_executed",
        "action_signature",
        "remaining_action_budget",
        "finish_action_hint",
        "commit_action_hint",
        "task_family",
        "action_space",
    )
    return {key: metadata[key] for key in keys if key in metadata}


def _task_objective_from_event(event: RescueEvent) -> str:
    for step in reversed(list(event.pre_rescue_window.steps) + list(event.post_rescue_steps)):
        metadata = getattr(step, "metadata", {}) or {}
        for key in ("task_objective", "objective", "user_objective", "task_instruction", "instruction"):
            value = str(metadata.get(key) or "").strip()
            if value:
                return value
    return ""


def _detection_rationale(event: RescueEvent) -> str:
    result = event.detection_result
    if result is not None and result.rationale:
        return result.rationale
    return ""


def _render_labels(labels: list[str]) -> str:
    return ", ".join(str(label) for label in labels) if labels else "none"


def _task_progress_delta(event: RescueEvent) -> float | None:
    before_scores = [
        step.task_progress_score
        for step in event.pre_rescue_window.steps
        if step.task_progress_score is not None
    ]
    after_scores = [
        step.task_progress_score
        for step in event.post_rescue_steps
        if step.task_progress_score is not None
    ]
    if not before_scores or not after_scores:
        return None
    return float(after_scores[-1] - before_scores[-1])
