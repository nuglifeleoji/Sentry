from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from ._structured import nonempty_text, string_list
from .taxonomy import FailureType, labels_for_failure_type


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class TaskContext:
    objective: str
    action_schema: str

    def __post_init__(self) -> None:
        if not isinstance(self.objective, str) or not self.objective.strip():
            raise ValueError("A task objective is required by the Sentry detector")
        if not isinstance(self.action_schema, str) or not self.action_schema.strip():
            raise ValueError("An action schema is required by Sentry")


@dataclass
class AgentAction:
    raw: str
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    parsed_ok: bool = True
    schema_valid: bool = True
    parser_error: str | None = None
    schema_error: str | None = None
    accepted_by_environment: bool | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.raw, str):
            raise ValueError("Action raw output must be a string")
        if type(self.parsed_ok) is not bool or type(self.schema_valid) is not bool:
            raise ValueError("Action parsed_ok and schema_valid must be booleans")
        if self.accepted_by_environment is not None and type(self.accepted_by_environment) is not bool:
            raise ValueError("Action accepted_by_environment must be a boolean or None")
        if not isinstance(self.tool_args, dict):
            raise ValueError("Action tool_args must be a dictionary")
        for name in ("tool_name", "parser_error", "schema_error"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"Action {name} must be a string or None")


@dataclass
class AgentStep:
    step_id: int
    timestamp: datetime
    reasoning: str
    action: AgentAction
    observation: str
    # Integration-only. Sentry never reads or renders benchmark rewards.
    task_progress_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if type(self.step_id) is not int or self.step_id < 0:
            raise ValueError("Cycle step_id must be a nonnegative integer")
        if not isinstance(self.timestamp, datetime):
            raise ValueError("Cycle timestamp must be a datetime")
        if not isinstance(self.reasoning, str) or not isinstance(self.observation, str):
            raise ValueError("Cycle reasoning and observation must be strings")
        if not isinstance(self.action, AgentAction):
            raise ValueError("Cycle action must be an AgentAction")
        self.action.validate()
        if not isinstance(self.metadata, dict):
            raise ValueError("Cycle metadata must be a dictionary")


@dataclass
class LocalTraceWindow:
    steps: list[AgentStep] = field(default_factory=list)
    max_size: int = 5

    def __post_init__(self) -> None:
        if type(self.max_size) is not int or self.max_size <= 0:
            raise ValueError("Trace window max_size must be a positive integer")
        initial = list(self.steps)
        self.steps = []
        for step in initial:
            self.add(step)

    def add(self, step: AgentStep) -> None:
        if not isinstance(step, AgentStep):
            raise TypeError("A trace cycle must be an AgentStep")
        step.validate()
        self.steps.append(copy.deepcopy(step))
        self.steps = self.steps[-self.max_size :]

    def clear(self) -> None:
        self.steps.clear()

    def copy(self) -> "LocalTraceWindow":
        return LocalTraceWindow(self.steps, self.max_size)


def agent_step_from_parts(
    *,
    step_id: int,
    reasoning: str,
    raw_action: str,
    tool_name: str | None = None,
    tool_args: dict[str, Any] | None = None,
    observation: str = "",
    parsed_ok: bool = True,
    schema_valid: bool = True,
    accepted_by_environment: bool | None = None,
    parser_error: str | None = None,
    schema_error: str | None = None,
    task_progress_score: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentStep:
    return AgentStep(
        step_id=step_id,
        timestamp=utcnow(),
        reasoning=reasoning,
        action=AgentAction(
            raw=raw_action,
            tool_name=tool_name,
            tool_args={} if tool_args is None else copy.deepcopy(tool_args),
            parsed_ok=parsed_ok,
            schema_valid=schema_valid,
            parser_error=parser_error,
            schema_error=schema_error,
            accepted_by_environment=accepted_by_environment,
        ),
        observation=observation,
        task_progress_score=task_progress_score,
        metadata={} if metadata is None else copy.deepcopy(metadata),
    )


@dataclass
class FailureDetection:
    failure_type: FailureType
    local_evidence: list[str]
    retrieval_labels: list[str] = field(default_factory=list)
    diagnosis: str = ""


@dataclass
class DetectionResult:
    should_rescue: bool
    failure_type: FailureType
    rationale: str
    source: str
    local_evidence: list[str] = field(default_factory=list)
    retrieval_labels: list[str] = field(default_factory=list)
    raw_response: str | None = None

    @property
    def intervention_needed(self) -> bool:
        return self.should_rescue


@dataclass
class PlaybookInsight:
    """A.6.2 recovery fields plus an identifier and the A.6.3 subtype.

    The paper does not define primary-label selection; this implementation uses
    the first assigned retrieval label without reordering the diagnosis.
    """

    insight_id: str
    failure_type: FailureType
    retrieval_labels: list[str]
    trigger_pattern: str
    repair_principle: str
    insertion_step: int = 0
    subtype: str = ""

    @property
    def primary_label(self) -> str | None:
        return self.retrieval_labels[0] if self.retrieval_labels else None

    @property
    def text(self) -> str:
        return f"Trigger: {self.trigger_pattern} Repair: {self.repair_principle}"

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["failure_type"] = self.failure_type.value
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "PlaybookInsight":
        required = {
            "insight_id", "failure_type", "retrieval_labels", "trigger_pattern",
            "repair_principle", "insertion_step",
        }
        if not isinstance(data, dict) or not required <= set(data) or set(data) - required - {"subtype"}:
            raise ValueError("Invalid playbook record fields")
        try:
            failure_type = FailureType(data["failure_type"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid playbook failure_type") from exc
        if failure_type not in {FailureType.PROGRESS_FAILURE, FailureType.REASONING_GROUNDING_FAILURE}:
            raise ValueError("Only soft recoveries belong in the playbook")
        labels = data["retrieval_labels"]
        if not string_list(labels, allow_empty=False):
            raise ValueError("A playbook record requires retrieval labels")
        if len(set(labels)) != len(labels) or any(label not in labels_for_failure_type(failure_type) for label in labels):
            raise ValueError("Invalid or duplicate playbook retrieval labels")
        if not all(nonempty_text(data[key]) for key in ("insight_id", "trigger_pattern", "repair_principle")):
            raise ValueError("Playbook identifiers, triggers and principles must be nonempty strings")
        insertion_step = data["insertion_step"]
        if type(insertion_step) is not int or insertion_step < 0:
            raise ValueError("Playbook insertion_step must be a nonnegative integer")
        subtype = data.get("subtype", "")
        if not isinstance(subtype, str):
            raise ValueError("Playbook subtype must be a string")
        return cls(
            insight_id=data["insight_id"],
            failure_type=failure_type,
            retrieval_labels=list(labels),
            trigger_pattern=data["trigger_pattern"].strip(),
            repair_principle=data["repair_principle"].strip(),
            insertion_step=insertion_step,
            subtype=subtype.strip(),
        )


@dataclass
class RescuePrompt:
    failure_type: FailureType
    prompt_text: str
    local_evidence: list[str]
    playbook_context: str = ""
    route: str = "soft_repair"
    retrieved_entry_ids: list[str] = field(default_factory=list)


@dataclass
class EscapeResult:
    escaped: bool
    failure_type: FailureType
    evidence: list[str]
    justification: str
    raw_response: str | None = None
    resolved_labels: list[str] = field(default_factory=list)
    unresolved_labels: list[str] = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        return self.escaped


@dataclass
class RescueEvent:
    event_id: str
    failure_type: FailureType
    detection: FailureDetection
    rescue_prompt: RescuePrompt
    pre_rescue_window: LocalTraceWindow
    detection_result: DetectionResult | None = None
    post_rescue_steps: list[AgentStep] = field(default_factory=list)

    @property
    def is_hard_repair(self) -> bool:
        return self.failure_type == FailureType.ACTION_VALIDITY_FAILURE
