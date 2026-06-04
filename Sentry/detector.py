from __future__ import annotations

import json
import re
from typing import Callable

from .config import SentryConfig
from .models import DetectionResult, LocalTraceWindow
from .taxonomy import (
    FAILURE_TYPE_DEFINITIONS,
    PROGRESS_LABEL_DEFINITIONS,
    REASONING_LABEL_DEFINITIONS,
    FailureType,
    labels_for_failure_type,
    normalize_retrieval_label,
    parse_failure_type,
)


class SentryDetector:
    def __init__(
        self,
        judge_callback: Callable[[str], str] | None = None,
        config: SentryConfig | None = None,
    ):
        self.judge_callback = judge_callback
        self.config = config or SentryConfig()

    def detect(self, window: LocalTraceWindow) -> DetectionResult:
        latest = window.steps[-1] if window.steps else None
        if latest is None:
            return _no_failure("No trajectory steps are available.", source="rule")

        hard_failure = _hard_repair_detection(latest)
        if hard_failure is not None:
            return hard_failure

        if self.judge_callback is not None:
            prompt = build_guard_judge_prompt(window, self.config)
            raw = self.judge_callback(prompt)
            return parse_guard_judge_response(
                raw,
                min_confidence=self.config.guard_judge.min_confidence,
            )

        return _no_failure(
            "No judge callback was configured and no hard interface failure was found.",
            source="rule",
        )


def _hard_repair_detection(step) -> DetectionResult | None:
    metadata = getattr(step, "metadata", {}) or {}
    if not step.action.parsed_ok or not step.action.schema_valid:
        evidence = [
            step.action.parser_error
            or step.action.schema_error
            or "The latest action does not satisfy the current action schema."
        ]
        rationale = "The latest action is not executable under the current schema."
    else:
        hinted_failure = _metadata_failure_type(metadata)
        hard_flags = (
            "action_format_error",
            "invalid_action_schema",
            "unavailable_action_target",
            "no_observable_output",
            "requires_hard_repair",
            "hard_repair",
        )
        triggered = [key for key in hard_flags if metadata.get(key)]
        if not triggered and hinted_failure != FailureType.ACTION_VALIDITY_FAILURE:
            return None
        reason = (
            metadata.get("hard_repair_reason")
            or metadata.get("action_validity_reason")
            or metadata.get("no_observable_output_reason")
            or metadata.get("failure_type_reason")
            or (
                f"Metadata marked hard repair condition: {', '.join(triggered)}."
                if triggered
                else "Metadata classified the latest step as an action-validity failure."
            )
        )
        evidence = [str(reason)]
        rationale = "The latest action requires hard repair before normal progress can resume."

    return DetectionResult(
        should_rescue=True,
        failure_type=FailureType.ACTION_VALIDITY_FAILURE,
        confidence=1.0,
        rationale=rationale,
        source="rule",
        recommended_constraint=_default_constraint(FailureType.ACTION_VALIDITY_FAILURE),
        local_evidence=evidence,
        retrieval_labels=[],
    )


def build_guard_judge_prompt(
    window: LocalTraceWindow,
    config: SentryConfig | None = None,
    *,
    task_objective: str | None = None,
) -> str:
    cfg = config or SentryConfig()
    objective = task_objective or _task_objective_from_window(window)
    step_lines: list[str] = []
    for step in window.steps[-cfg.guard_judge.max_window_steps :]:
        action = step.action.raw or step.action.tool_name or ""
        reasoning = " ".join(str(step.reasoning or "").split())
        obs = " ".join(str(step.observation or "").split())
        if len(obs) > cfg.guard_judge.max_observation_chars:
            obs = obs[: cfg.guard_judge.max_observation_chars] + "..."
        schema = step.metadata.get("schema_description") if step.metadata else None
        schema_part = f" schema={schema!r}" if schema else ""
        metadata_hints = _render_metadata_hints(step.metadata or {})
        metadata_part = f" metadata_hints={metadata_hints}" if metadata_hints else ""
        step_lines.append(
            f"- step={step.step_id} reasoning={reasoning!r} action={action!r} "
            f"parsed={step.action.parsed_ok} schema_valid={step.action.schema_valid} "
            f"progress={step.task_progress_score} observation={obs!r}{schema_part}{metadata_part}"
        )
    recent = "\n".join(step_lines)
    objective_block = objective or "No explicit task objective was provided."
    taxonomy_block = _render_judge_taxonomy()
    return f"""You are the Sentry guard detector.

Hard interface failures are handled before this judge runs. Your job is to
classify only progress failures and reasoning/grounding failures.

Classify the recent task-agent trajectory using only this taxonomy:
{taxonomy_block}

Decision policy:
- Prefer no_failure when the trajectory is plausibly moving toward the objective.
- Use progress_failure when the recent trajectory fails to move the task forward even if the agent's claims are otherwise grounded.
- Use reasoning_grounding_failure when the agent's claims, plan, or action choice is unsupported by the objective or observations.
- Assign the smallest sufficient set of retrieval labels, but do not always default to repetition.
- Evidence must cite concrete recent steps, actions, observations, or missing/contradicted support. Do not invent task state or tool results.

Label choice guide:
- repetition_or_looping: choose this only when the same or near-identical action, query, click, navigation step, or reasoning pattern repeats without new information.
- planning_stall: choose this when the agent talks about the goal or uncertainty but does not convert available information into an executable next action.
- over_exploration: choose this when the agent keeps searching, browsing, comparing, or inspecting even though enough evidence, visible candidates, or integration-recommended actions are available to make a concrete choice.
- termination_miscalibration: choose this when a valid finish/commit action is visible or the task appears locally complete but the agent keeps acting, or when it stops before satisfying stated requirements.
- hallucination: choose this when the agent relies on facts, item attributes, tool results, files, or environment states not supported by the observations.
- objective_drift: choose this when the agent optimizes for a nearby but wrong objective rather than the original user objective.
- reasoning_action_mismatch: choose this when the chosen action does not follow from the agent's stated subgoal or the latest observation.

Task objective:
{objective_block}

Recent trajectory:
{recent or "- none"}

Use objective_drift only when the trajectory clearly optimizes for a goal that
conflicts with the task objective. If no explicit objective is available, use
objective_drift only when the drift is clear from the recent trajectory itself.

Return only compact JSON:
{{
  "should_rescue": true,
  "failure_type": "progress_failure | reasoning_grounding_failure | no_failure",
  "confidence": 0.75,
  "retrieval_labels": ["one_or_more_supported_labels"],
  "evidence": ["short local evidence from the trajectory"],
  "reason": "short reason grounded in the trajectory",
  "recommended_constraint": "optional short constraint for the next action"
}}
"""


def _render_judge_taxonomy() -> str:
    lines = [
        f"- {FailureType.NO_FAILURE.value}: {FAILURE_TYPE_DEFINITIONS[FailureType.NO_FAILURE]}",
        f"- {FailureType.PROGRESS_FAILURE.value}: {FAILURE_TYPE_DEFINITIONS[FailureType.PROGRESS_FAILURE]}",
        "  retrieval labels:",
    ]
    lines.extend(
        f"  - {label}: {PROGRESS_LABEL_DEFINITIONS[label]}"
        for label in labels_for_failure_type(FailureType.PROGRESS_FAILURE)
    )
    lines.extend(
        [
            f"- {FailureType.REASONING_GROUNDING_FAILURE.value}: "
            f"{FAILURE_TYPE_DEFINITIONS[FailureType.REASONING_GROUNDING_FAILURE]}",
            "  retrieval labels:",
        ]
    )
    lines.extend(
        f"  - {label}: {REASONING_LABEL_DEFINITIONS[label]}"
        for label in labels_for_failure_type(FailureType.REASONING_GROUNDING_FAILURE)
    )
    return "\n".join(lines)


def _render_metadata_hints(metadata: dict) -> str:
    keys = (
        "failure_type_hint",
        "retrieval_labels",
        "failure_type_reason",
        "remaining_action_budget",
        "action_mode",
        "state_stage",
        "recommended_rescue_actions",
        "soft_repeat_signal",
        "finish_action_hint",
        "commit_action_hint",
    )
    rendered: list[str] = []
    for key in keys:
        value = metadata.get(key)
        if value in (None, "", [], {}):
            continue
        text = " ".join(str(value or "").split())
        if text:
            rendered.append(f"{key}={text[:240]}")
    return "; ".join(rendered)


def parse_guard_judge_response(
    raw: str,
    *,
    min_confidence: float,
) -> DetectionResult:
    text = str(raw or "").strip()
    try:
        parsed = json.loads(text)
        data = parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        data = None
        if match is not None:
            try:
                parsed = json.loads(match.group(0))
                data = parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                data = None
    if data is None:
        return DetectionResult(
            should_rescue=False,
            failure_type=FailureType.NO_FAILURE,
            confidence=0.0,
            rationale="Detector judge did not return parseable JSON.",
            source="judge",
            raw_response=raw,
        )
    failure_type = parse_failure_type(data.get("failure_type"), FailureType.NO_FAILURE)
    if failure_type is None:
        failure_type = FailureType.NO_FAILURE
    if failure_type == FailureType.ACTION_VALIDITY_FAILURE:
        return DetectionResult(
            should_rescue=False,
            failure_type=FailureType.NO_FAILURE,
            confidence=0.0,
            rationale=(
                "Judge returned action_validity_failure, but hard repair is "
                "handled before the judge and no hard interface failure was detected."
            ),
            source="judge",
            raw_response=raw,
        )
    raw_should = bool(data.get("should_rescue"))
    default_confidence = min_confidence if raw_should and failure_type != FailureType.NO_FAILURE else 0.0
    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence"))))
    except (TypeError, ValueError):
        confidence = default_confidence
    if raw_should and failure_type != FailureType.NO_FAILURE and confidence <= 0.0:
        confidence = min_confidence
    raw_evidence = data.get("evidence")
    raw_items = raw_evidence if isinstance(raw_evidence, list) else [raw_evidence]
    evidence: list[str] = []
    for item in raw_items:
        if item is None:
            continue
        text = str(item).strip()
        if text and text not in evidence:
            evidence.append(text)
    reason = str(data.get("reason") or data.get("rationale") or "").strip()
    if reason and reason not in evidence:
        evidence.insert(0, reason)
    labels = _normalize_labels(data.get("retrieval_labels"), failure_type)
    should = (
        raw_should
        and confidence >= min_confidence
        and failure_type != FailureType.NO_FAILURE
    )
    return DetectionResult(
        should_rescue=should,
        failure_type=failure_type,
        confidence=confidence,
        rationale=reason,
        source="judge",
        recommended_constraint=(
            str(data.get("recommended_constraint"))
            if data.get("recommended_constraint") is not None
            else _default_constraint(failure_type)
        ),
        raw_response=raw,
        local_evidence=evidence,
        retrieval_labels=labels,
    )


def _metadata_failure_type(metadata: dict) -> FailureType | None:
    raw = (
        metadata.get("failure_type_hint")
        or metadata.get("integration_failure_type")
        or metadata.get("sentry_failure_type")
    )
    soft_repeat = metadata.get("soft_repeat_signal")
    if raw is None and isinstance(soft_repeat, dict):
        raw = soft_repeat.get("failure_type")
    return parse_failure_type(raw)


def _task_objective_from_window(window: LocalTraceWindow) -> str:
    keys = (
        "task_objective",
        "objective",
        "user_objective",
        "task_instruction",
        "instruction",
        "user_request",
        "goal",
    )
    for step in reversed(window.steps):
        metadata = step.metadata or {}
        for key in keys:
            value = metadata.get(key)
            text = " ".join(str(value or "").split())
            if text:
                return text
    return ""


def _normalize_labels(value: object, failure_type: FailureType) -> list[str]:
    if failure_type == FailureType.ACTION_VALIDITY_FAILURE:
        return []
    raw_items = value if isinstance(value, list) else [value]
    labels: list[str] = []
    for item in raw_items:
        label = normalize_retrieval_label(item, failure_type)
        if label is not None and label not in labels:
            labels.append(label)
    if labels:
        return labels
    allowed = labels_for_failure_type(failure_type)
    return [allowed[0]] if allowed else []


def _default_constraint(failure_type: FailureType) -> str:
    if failure_type == FailureType.PROGRESS_FAILURE:
        return "Choose a next action that changes state, produces new task evidence, or finishes if the requirements are satisfied."
    if failure_type == FailureType.ACTION_VALIDITY_FAILURE:
        return "Return exactly one valid action that is available in the current state and follows the required schema."
    if failure_type == FailureType.REASONING_GROUNDING_FAILURE:
        return "Ground the next action in recent observations and align it with the task objective."
    return "Continue only if the local trajectory does not support intervention."


def _no_failure(reason: str, *, source: str) -> DetectionResult:
    return DetectionResult(
        should_rescue=False,
        failure_type=FailureType.NO_FAILURE,
        confidence=0.0,
        rationale=reason,
        source=source,
    )
