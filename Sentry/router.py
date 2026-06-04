from __future__ import annotations

from .models import (
    DetectionResult,
    FailureDetection,
    LocalTraceWindow,
    PlaybookSection,
    RescuePrompt,
)
from .playbook import PlaybookManager
from .taxonomy import FailureType, RETRIEVAL_LABEL_TITLES, normalize_retrieval_label


def route_rescue(
    *,
    detection: FailureDetection,
    detection_result: DetectionResult,
    window: LocalTraceWindow,
    playbook: PlaybookManager,
) -> tuple[str, RescuePrompt | None]:
    failure_type = detection_result.failure_type
    if not detection_result.should_rescue or failure_type == FailureType.NO_FAILURE:
        return "no_intervention", None

    if failure_type == FailureType.ACTION_VALIDITY_FAILURE:
        constraint = (
            detection_result.recommended_constraint
            or "Regenerate exactly one valid action in the required schema."
        )
        return "hard_strategy", compose_hard_repair_prompt(
            detection,
            window,
            detection_result,
            constraint,
        )

    if failure_type not in {
        FailureType.PROGRESS_FAILURE,
        FailureType.REASONING_GROUNDING_FAILURE,
    }:
        return "no_intervention", None

    constraint = detection_result.recommended_constraint
    section = playbook.retrieve(detection.failure_type, detection.retrieval_labels)
    return "soft_rescue", compose_soft_repair_prompt(
        detection,
        section,
        window,
        detection_result,
        constraint,
    )


def compose_hard_repair_prompt(
    detection: FailureDetection,
    window: LocalTraceWindow,
    detection_result: DetectionResult,
    constraint: str | None,
) -> RescuePrompt:
    evidence = detection.local_evidence or ["The latest action failed interface validation."]
    prompt = f"""[SENTRY HARD REPAIR]
Detected failure type: {detection.failure_type.value}

Why this was detected:
{_bullets(evidence, empty="No extra evidence.")}

Detector signal:
{_render_diagnosis(detection_result, constraint)}

Repair instructions:
{_hard_repair_instructions(window)}

{_render_context_hints(window)}Now repair the interface violation and continue.
{_output_constraint(detection.failure_type, window, constraint)}
""".strip()

    return RescuePrompt(
        failure_type=detection.failure_type,
        prompt_text=prompt,
        local_evidence=list(evidence),
        playbook_context="",
    )


def compose_soft_repair_prompt(
    detection: FailureDetection,
    playbook_section: PlaybookSection,
    window: LocalTraceWindow,
    detection_result: DetectionResult,
    constraint: str | None,
) -> RescuePrompt:
    evidence = detection.local_evidence or ["The local trace matches this failure pattern."]
    playbook_context = _render_playbook_context(playbook_section, detection.retrieval_labels)
    prompt = f"""[SENTRY SOFT REPAIR]
Detected failure type: {detection.failure_type.value}
Assigned retrieval labels: {_render_labels(detection.retrieval_labels)}

General advice:
{_render_general_advice(detection.failure_type, detection.retrieval_labels, window)}

Why this was detected:
{_bullets(evidence, empty="No extra evidence.")}

Detector signal:
{_render_diagnosis(detection_result, constraint)}

{_render_context_hints(window)}Insights from previous successful recoveries:
{playbook_context}

Continue the original task from the corrected next step.
{_output_constraint(detection.failure_type, window, constraint)}
""".strip()

    return RescuePrompt(
        failure_type=detection.failure_type,
        prompt_text=prompt,
        local_evidence=list(evidence),
        playbook_context=playbook_context,
    )


def _bullets(items, empty: str = "none") -> str:
    lines = [str(item).strip() for item in items if item is not None and str(item).strip()]
    return "\n".join(f"- {line}" for line in lines) or f"- {empty}"


def _render_playbook_context(section: PlaybookSection, labels: list[str] | None) -> str:
    return _render_insights(section, labels) or "- No retrieved playbook insight yet."


def _render_insights(section: PlaybookSection, labels: list[str] | None) -> str:
    selected = _selected_labels(section, labels)
    lines: list[str] = []
    for label in selected:
        insights = section.label_insights.get(label) or []
        if not insights:
            continue
        title = RETRIEVAL_LABEL_TITLES.get(label, label.replace("_", " ").title())
        lines.append(f"- {title}:")
        lines.extend(f"  - {insight.text}" for insight in insights)
    return "\n".join(lines)


def _selected_labels(section: PlaybookSection, labels: list[str] | None) -> list[str]:
    if labels is None:
        labels = section.label_order
    out: list[str] = []
    for item in labels:
        label = normalize_retrieval_label(item, section.failure_type)
        if label is not None and label not in out:
            out.append(label)
    return out


def _render_labels(labels: list[str]) -> str:
    if not labels:
        return "none"
    return ", ".join(
        f"{RETRIEVAL_LABEL_TITLES.get(label, label.replace('_', ' ').title())} (`{label}`)"
        for label in labels
    )


def _render_diagnosis(
    detection_result: DetectionResult,
    constraint: str | None,
) -> str:
    lines: list[str] = []
    if detection_result.rationale:
        lines.append(f"- Detector diagnosis: {detection_result.rationale}")
    if constraint:
        lines.append(f"- Constraint: {constraint}")
    return "\n".join(lines) or "- No additional constraint."


def _render_context_hints(window: LocalTraceWindow) -> str:
    hints = _render_protocol_hints(window) + _render_integration_hints(window)
    return hints


def _render_protocol_hints(window: LocalTraceWindow) -> str:
    for step in reversed(window.steps):
        raw_hints = step.metadata.get("protocol_hints")
        hints = [raw_hints] if isinstance(raw_hints, str) else raw_hints
        if isinstance(hints, list) and hints:
            return "Protocol reminders:\n" + _bullets(hints) + "\n\n"
    return ""


def _render_integration_hints(window: LocalTraceWindow) -> str:
    if not window.steps:
        return ""
    metadata = window.steps[-1].metadata or {}
    lines: list[str] = []
    if metadata.get("remaining_action_budget") is not None:
        lines.append(f"Approximate remaining action budget: {metadata['remaining_action_budget']}.")
    finish_hint = metadata.get("finish_action_hint") or metadata.get("commit_action_hint")
    if finish_hint:
        lines.append(f"Valid finish/commit action when appropriate: `Action: {finish_hint}`.")
    recommended = _latest_recommended_rescue_actions(window)
    if recommended:
        preview = ", ".join(f"`Action: {action}`" for action in recommended)
        lines.append(f"Integration-recommended next action candidates: {preview}.")
    return "Integration hints:\n" + _bullets(lines) + "\n\n" if lines else ""


def _schema_description(window: LocalTraceWindow) -> str:
    if not window.steps:
        return "Use the original task's required action format exactly."
    latest = window.steps[-1]
    return str(
        latest.metadata.get("schema_description")
        or latest.action.schema_error
        or "Use the original task's required action format exactly."
    )


def _latest_action_error_context(window: LocalTraceWindow) -> str:
    if not window.steps:
        return "- No latest action is available."
    latest = window.steps[-1]
    lines = [
        f"- Previous action: {latest.action.raw or latest.action.tool_name or '<empty>'}",
        f"- Parsed: {latest.action.parsed_ok}",
        f"- Schema valid: {latest.action.schema_valid}",
    ]
    if latest.action.parser_error:
        lines.append(f"- Parser error: {latest.action.parser_error}")
    if latest.action.schema_error:
        lines.append(f"- Schema error: {latest.action.schema_error}")
    observation = " ".join(str(latest.observation or "").split())
    if observation:
        lines.append(f"- Latest observation: {observation[:500]}")
    return "\n".join(lines)


def _output_constraint(
    failure_type: FailureType,
    window: LocalTraceWindow,
    constraint: str | None,
) -> str:
    extra = f" Additional constraint: {constraint}" if constraint else ""
    if failure_type == FailureType.ACTION_VALIDITY_FAILURE:
        return f"Return only one valid action. Required schema: {_schema_description(window)}{extra}"
    return f"Keep following the original task protocol. Required action format: {_schema_description(window)}{extra}"


def _hard_repair_instructions(window: LocalTraceWindow) -> str:
    return f"""The previous action was invalid, unavailable, malformed, or failed to expose required output.

Invalid action context:
{_latest_action_error_context(window)}

Required repair:
1. Produce exactly one executable action in the required format.
2. Use only an allowed action/tool name.
3. Include every required field with the correct value types.
4. Select only targets or controls that are valid in the latest observation.
5. If the action is meant to inspect information, make the result visible in the next observation.
6. Do not include reasoning, prose, markdown, or extra text outside the action."""


def _render_general_advice(
    failure_type: FailureType,
    labels: list[str],
    window: LocalTraceWindow,
) -> str:
    advice = _failure_level_advice(failure_type, window)
    label_advice: list[str] = []
    seen: list[str] = []
    for item in labels:
        label = normalize_retrieval_label(item, failure_type)
        if label is None or label in seen:
            continue
        seen.append(label)
        text = _label_advice(label, failure_type)
        if text:
            label_advice.append(text)
    return "\n".join([advice, *label_advice])


def _failure_level_advice(failure_type: FailureType, window: LocalTraceWindow) -> str:
    if failure_type == FailureType.PROGRESS_FAILURE:
        repeat_context = _repeat_context(window)
        return (
            "- Pick one concrete next step that changes state, adds task-relevant "
            "information, or finishes if the task is already satisfied."
            + repeat_context
        )

    if failure_type == FailureType.REASONING_GROUNDING_FAILURE:
        return (
            "- Ground the next action in the original objective and the latest "
            "verified observation, not in unsupported assumptions."
        )

    return "- Use the detector evidence to continue the original task from a corrected next step."


def _label_advice(label: str, failure_type: FailureType) -> str | None:
    normalized = normalize_retrieval_label(label, failure_type)
    if normalized == "repetition_or_looping":
        return (
            "- Do not repeat a stale action unless the next observation would expose "
            "new task-relevant information."
        )
    if normalized == "planning_stall":
        return "- Convert the current subgoal into one executable action instead of continuing to deliberate."
    if normalized == "over_exploration":
        return (
            "- Stop broad exploration when enough evidence is available; inspect, "
            "choose, or commit to a concrete candidate."
        )
    if normalized == "termination_miscalibration":
        return (
            "- If the requirements are satisfied and a valid finish/commit action "
            "exists, use it instead of spending more steps."
        )
    if normalized == "hallucination":
        return (
            "- Separate verified observations from assumptions, and verify unsupported "
            "facts before acting on them."
        )
    if normalized == "objective_drift":
        return "- Restate the original user objective and reject actions that optimize for a nearby but incorrect goal."
    if normalized == "reasoning_action_mismatch":
        return (
            "- Make the next action type and target directly follow from the stated "
            "subgoal and latest observation."
        )
    return None


def _repeat_context(window: LocalTraceWindow) -> str:
    soft_repeat = _latest_soft_repeat_signal(window)
    if not soft_repeat:
        return ""
    signature = soft_repeat.get("action_signature")
    detail = soft_repeat.get("message") or "The latest action repeated a no-progress action."
    if signature:
        return f"\n- Soft repeat signal: `{signature}` already produced no new information."
    return f"\n- Soft repeat signal: {detail}"


def _latest_recommended_rescue_actions(window: LocalTraceWindow) -> list[str]:
    if not window.steps:
        return []
    raw = window.steps[-1].metadata.get("recommended_rescue_actions")
    if not isinstance(raw, list):
        return []
    return [str(item).strip() for item in raw if str(item).strip()][:3]


def _latest_soft_repeat_signal(window: LocalTraceWindow) -> dict | None:
    if not window.steps:
        return None
    value = window.steps[-1].metadata.get("soft_repeat_signal")
    return value if isinstance(value, dict) else None
