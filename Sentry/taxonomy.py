from __future__ import annotations

from enum import Enum


class FailureType(str, Enum):
    NO_FAILURE = "no_failure"
    PROGRESS_FAILURE = "progress_failure"
    ACTION_VALIDITY_FAILURE = "action_validity_failure"
    REASONING_GROUNDING_FAILURE = "reasoning_grounding_failure"


def parse_failure_type(value: object, fallback: FailureType | None = None) -> FailureType | None:
    if isinstance(value, FailureType):
        return value
    text = str(value or "").strip().lower()
    if not text:
        return fallback
    try:
        return FailureType(text)
    except ValueError:
        mapped = FAILURE_TYPE_ALIASES.get(text)
        return mapped or fallback


FAILURE_TYPE_DEFINITIONS: dict[FailureType, str] = {
    FailureType.NO_FAILURE: "Continue without intervention.",
    FailureType.PROGRESS_FAILURE: (
        "Progress failures occur when the agent's recent trajectory does not "
        "meaningfully move the task toward completion."
    ),
    FailureType.ACTION_VALIDITY_FAILURE: (
        "Action-validity failures occur when the action is malformed, invalid "
        "for the interface, unavailable in the current state, missing required "
        "fields, uses invalid field types, or hides required observable output."
    ),
    FailureType.REASONING_GROUNDING_FAILURE: (
        "Reasoning and grounding failures occur when the agent's claims, plans, "
        "or actions are not adequately supported by the task specification or "
        "by the observations it has received."
    ),
}

PROGRESS_LABEL_DEFINITIONS: dict[str, str] = {
    "repetition_or_looping": (
        "The agent repeats the same or nearly identical reasoning steps, tool "
        "calls, queries, or actions without gaining new task-relevant information."
    ),
    "planning_stall": (
        "The agent recognizes the task objective but fails to convert available "
        "information into a concrete next step, often continuing to deliberate "
        "without taking useful action."
    ),
    "over_exploration": (
        "The agent keeps searching, browsing, inspecting, or gathering information "
        "even after it has enough evidence to make progress toward the task goal."
    ),
    "termination_miscalibration": (
        "The agent stops too early before satisfying the task objective, or "
        "continues acting after the task is already effectively complete."
    ),
}

REASONING_LABEL_DEFINITIONS: dict[str, str] = {
    "hallucination": (
        "The agent introduces unsupported facts, assumptions, tool results, file "
        "contents, environment states, or task constraints that do not appear in "
        "the trajectory."
    ),
    "objective_drift": (
        "The agent shifts away from the user's original objective or optimizes "
        "for a nearby but incorrect goal."
    ),
    "reasoning_action_mismatch": (
        "The proposed action does not follow from the agent's own reasoning or "
        "from the current observation, such as choosing an action that contradicts "
        "its stated plan."
    ),
}

ACTION_VALIDITY_RESOLUTION_CRITERION = (
    "The next action is parseable, schema-valid, uses an available action/tool and target, "
    "includes required fields with valid types, and exposes required observable output."
)

PROGRESS_LABEL_RESOLUTION_CRITERIA: dict[str, str] = {
    "repetition_or_looping": (
        "Resolved when the post-rescue trajectory stops repeating the stale reasoning, "
        "query, tool call, navigation step, or action and obtains new task-relevant "
        "information, changes state, or finishes the task."
    ),
    "planning_stall": (
        "Resolved when the agent converts the recognized objective into a concrete "
        "executable next action that can advance the task."
    ),
    "over_exploration": (
        "Resolved when the agent stops unnecessary searching/inspection and instead "
        "uses available evidence to inspect a concrete candidate, choose, commit, or finish."
    ),
    "termination_miscalibration": (
        "Resolved when the agent either avoids premature stopping before requirements "
        "are met, or uses a valid finish/commit action once the task is effectively complete."
    ),
}

REASONING_LABEL_RESOLUTION_CRITERIA: dict[str, str] = {
    "hallucination": (
        "Resolved when unsupported facts, tool results, file contents, environment "
        "states, or constraints are dropped, corrected, or verified before being used."
    ),
    "objective_drift": (
        "Resolved when the post-rescue trajectory returns to the original user objective "
        "instead of optimizing for a nearby but incorrect goal."
    ),
    "reasoning_action_mismatch": (
        "Resolved when the selected action type and target directly follow from the "
        "agent's stated subgoal and the latest verified observation."
    ),
}

PROGRESS_RETRIEVAL_LABELS: tuple[str, ...] = tuple(PROGRESS_LABEL_DEFINITIONS)
REASONING_RETRIEVAL_LABELS: tuple[str, ...] = tuple(REASONING_LABEL_DEFINITIONS)

RETRIEVAL_LABEL_TITLES: dict[str, str] = {
    "repetition_or_looping": "Repetition or looping",
    "planning_stall": "Planning stalls",
    "over_exploration": "Over-exploration",
    "termination_miscalibration": "Termination miscalibration",
    "hallucination": "Hallucination",
    "objective_drift": "Objective drift",
    "reasoning_action_mismatch": "Reasoning-action mismatch",
}

FAILURE_TYPE_ALIASES: dict[str, FailureType] = {
    "looping": FailureType.PROGRESS_FAILURE,
    "search_stagnation": FailureType.PROGRESS_FAILURE,
    "navigation_cycle": FailureType.PROGRESS_FAILURE,
    "repeated_no_progress": FailureType.PROGRESS_FAILURE,
    "missing_commit": FailureType.PROGRESS_FAILURE,
    "planning_stall": FailureType.PROGRESS_FAILURE,
    "over_exploration": FailureType.PROGRESS_FAILURE,
    "termination_miscalibration": FailureType.PROGRESS_FAILURE,
    "action_format_error": FailureType.ACTION_VALIDITY_FAILURE,
    "unavailable_action_target": FailureType.ACTION_VALIDITY_FAILURE,
    "invalid_action_schema": FailureType.ACTION_VALIDITY_FAILURE,
    "malformed_action": FailureType.ACTION_VALIDITY_FAILURE,
    "no_observable_output": FailureType.ACTION_VALIDITY_FAILURE,
    "unsupported_belief": FailureType.REASONING_GROUNDING_FAILURE,
    "hallucination": FailureType.REASONING_GROUNDING_FAILURE,
    "objective_drift": FailureType.REASONING_GROUNDING_FAILURE,
    "reasoning_action_mismatch": FailureType.REASONING_GROUNDING_FAILURE,
}

LABEL_ALIASES: dict[str, str] = {
    "looping": "repetition_or_looping",
    "search_stagnation": "repetition_or_looping",
    "navigation_cycle": "repetition_or_looping",
    "repeated_no_progress": "repetition_or_looping",
    "repeat_no_progress": "repetition_or_looping",
    "planning_stall": "planning_stall",
    "planning_stalls": "planning_stall",
    "over_exploration": "over_exploration",
    "over-exploration": "over_exploration",
    "missing_commit": "termination_miscalibration",
    "finish_now": "termination_miscalibration",
    "termination_miscalibration": "termination_miscalibration",
    "unsupported_belief": "hallucination",
    "ungrounded_assumption": "hallucination",
    "hallucination": "hallucination",
    "objective_drift": "objective_drift",
    "goal_drift": "objective_drift",
    "reasoning_action_mismatch": "reasoning_action_mismatch",
    "reason_action_mismatch": "reasoning_action_mismatch",
}


def normalize_retrieval_label(
    value: object,
    failure_type: FailureType | None = None,
) -> str | None:
    text = str(value or "").strip().lower().replace(" ", "_")
    if not text:
        return None
    label = LABEL_ALIASES.get(text, text)
    allowed = labels_for_failure_type(failure_type) if failure_type is not None else (
        PROGRESS_RETRIEVAL_LABELS + REASONING_RETRIEVAL_LABELS
    )
    return label if label in allowed else None


def labels_for_failure_type(failure_type: FailureType | None) -> tuple[str, ...]:
    if failure_type == FailureType.PROGRESS_FAILURE:
        return PROGRESS_RETRIEVAL_LABELS
    if failure_type == FailureType.REASONING_GROUNDING_FAILURE:
        return REASONING_RETRIEVAL_LABELS
    return ()


def resolution_criteria_for(
    failure_type: FailureType | None,
    labels: list[str] | tuple[str, ...] | None = None,
) -> dict[str, str]:
    if failure_type == FailureType.ACTION_VALIDITY_FAILURE:
        return {FailureType.ACTION_VALIDITY_FAILURE.value: ACTION_VALIDITY_RESOLUTION_CRITERION}
    if failure_type == FailureType.PROGRESS_FAILURE:
        criteria = PROGRESS_LABEL_RESOLUTION_CRITERIA
    elif failure_type == FailureType.REASONING_GROUNDING_FAILURE:
        criteria = REASONING_LABEL_RESOLUTION_CRITERIA
    else:
        return {}

    selected = [
        label
        for label in (
            normalize_retrieval_label(item, failure_type) for item in (labels or ())
        )
        if label is not None
    ]
    if not selected:
        selected = list(labels_for_failure_type(failure_type))
    return {label: criteria[label] for label in dict.fromkeys(selected) if label in criteria}
