from __future__ import annotations

from .detector import render_trajectory
from .models import (
    DetectionResult,
    FailureDetection,
    LocalTraceWindow,
    PlaybookInsight,
    RescuePrompt,
    TaskContext,
)
from .playbook import PlaybookManager
from .taxonomy import FailureType, RETRIEVAL_LABEL_TITLES


def route_rescue(
    *,
    detection: FailureDetection,
    detection_result: DetectionResult,
    window: LocalTraceWindow,
    task: TaskContext,
    playbook: PlaybookManager,
) -> tuple[str, RescuePrompt | None]:
    if not detection_result.should_rescue:
        return "no_intervention", None
    if detection.failure_type == FailureType.ACTION_VALIDITY_FAILURE:
        return "hard_repair", compose_hard_repair_prompt(detection, window, task)
    if detection.failure_type not in {
        FailureType.PROGRESS_FAILURE,
        FailureType.REASONING_GROUNDING_FAILURE,
    }:
        return "no_intervention", None
    retrieved = playbook.retrieve(detection.failure_type, detection.retrieval_labels)
    return "soft_repair", compose_soft_repair_prompt(
        detection, detection_result, retrieved, task
    )


def compose_hard_repair_prompt(
    detection: FailureDetection,
    window: LocalTraceWindow,
    task: TaskContext,
) -> RescuePrompt:
    latest = window.steps[-1]
    relevant = window.steps
    prompt = f"""[SENTRY HARD REPAIR]
Detected issue.
The previous action is invalid or not executable under the required schema. Rewrite the action into one valid action that follows the same task objective and the current observation.

Task objective.
{task.objective}

Recent trajectory.
{render_trajectory(relevant)}

Invalid action.
{latest.action.raw}

Schema error.
{latest.action.parser_error or latest.action.schema_error or next(iter(detection.local_evidence), "The action does not satisfy the required schema.")}

Required schema.
{task.action_schema}

Continuation instruction.
Output only the corrected action in the required action format.
"""
    return RescuePrompt(
        failure_type=detection.failure_type,
        prompt_text=prompt,
        local_evidence=list(detection.local_evidence),
        route="hard_repair",
    )


def compose_soft_repair_prompt(
    detection: FailureDetection,
    detection_result: DetectionResult,
    retrieved: list[PlaybookInsight],
    task: TaskContext,
) -> RescuePrompt:
    guidance = render_playbook_entries(retrieved)
    labels = ", ".join(detection.retrieval_labels)
    evidence = "\n".join(f"- {item}" for item in detection.local_evidence)
    prompt = f"""[SENTRY SOFT REPAIR]
Task objective.
{task.objective}

Detected failure type.
{detection.failure_type.value}

Assigned retrieval labels.
{labels}

General repair advice.
{_general_advice(detection.failure_type)}
{_label_advice(detection.retrieval_labels)}

Why this was detected.
{evidence}

Detector signal.
{detection_result.rationale}

Insights from previous successful recoveries.
{guidance}

Continuation instruction.
Continue the original task from the corrected next step. Keep following the original task protocol.

Required action format.
{task.action_schema}
"""
    return RescuePrompt(
        failure_type=detection.failure_type,
        prompt_text=prompt,
        local_evidence=list(detection.local_evidence),
        playbook_context=guidance,
        route="soft_repair",
        retrieved_entry_ids=[entry.insight_id for entry in retrieved],
    )


def render_playbook_entries(entries: list[PlaybookInsight]) -> str:
    if not entries:
        return "No matching verified recovery is available yet."
    lines: list[str] = []
    for entry in entries:
        labels = ", ".join(
            RETRIEVAL_LABEL_TITLES.get(label, label) for label in entry.retrieval_labels
        ) or "general"
        lines.append(
            f"- [{entry.insight_id}] {labels}. Trigger pattern: "
            f"{entry.trigger_pattern} Repair principle: {entry.repair_principle}"
        )
    return "\n".join(lines)


def _general_advice(failure_type: FailureType) -> str:
    if failure_type == FailureType.PROGRESS_FAILURE:
        return (
            "Stop the diagnosed non-progress pattern and choose a concrete next action "
            "that changes state, obtains new task-relevant information, or terminates "
            "when the objective is satisfied."
        )
    return (
        "Ground the next action in the original objective and observed evidence; "
        "drop unsupported assumptions and make the action follow from the reasoning."
    )


def _label_advice(labels: list[str]) -> str:
    advice = {
        "repetition_or_looping": "Do not repeat an action that produced no new information.",
        "planning_stall": "Convert the current plan into one executable next action.",
        "over_exploration": "Use the evidence already gathered to choose or commit.",
        "termination_miscalibration": "Stop only when requirements are met; terminate once they are met.",
        "hallucination": "Verify or remove every claim not supported by an observation.",
        "objective_drift": "Return to the user's original objective.",
        "reasoning_action_mismatch": "Choose an action that follows from the stated plan and latest observation.",
    }
    return "\n".join(f"- {advice[label]}" for label in labels if label in advice)
