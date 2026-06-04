from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from typing import Any, Optional

from .taxonomy import FailureType, parse_failure_type


def utcnow() -> datetime:
    return datetime.utcnow()


@dataclass
class AgentAction:
    raw: str
    tool_name: Optional[str] = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    parsed_ok: bool = True
    schema_valid: bool = True
    parser_error: Optional[str] = None
    schema_error: Optional[str] = None


@dataclass
class AgentStep:
    step_id: int
    timestamp: datetime
    reasoning: str
    action: AgentAction
    observation: str
    task_progress_score: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalTraceWindow:
    steps: list[AgentStep] = field(default_factory=list)
    max_size: int = 5

    def add(self, step: AgentStep) -> None:
        self.steps.append(step)
        if len(self.steps) > self.max_size:
            self.steps = self.steps[-self.max_size :]

    def copy(self) -> "LocalTraceWindow":
        return LocalTraceWindow(steps=copy.deepcopy(self.steps), max_size=self.max_size)


def agent_step_from_parts(
    *,
    step_id: int,
    reasoning: str,
    raw_action: str,
    tool_name: str | None,
    tool_args: dict[str, Any] | None = None,
    observation: str = "",
    parsed_ok: bool = True,
    schema_valid: bool = True,
    parser_error: str | None = None,
    schema_error: str | None = None,
    task_progress_score: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentStep:
    return AgentStep(
        step_id=step_id,
        timestamp=utcnow(),
        reasoning=reasoning or "",
        action=AgentAction(
            raw=raw_action or "",
            tool_name=tool_name,
            tool_args=dict(tool_args or {}),
            parsed_ok=parsed_ok,
            schema_valid=schema_valid,
            parser_error=parser_error,
            schema_error=schema_error,
        ),
        observation=observation or "",
        task_progress_score=task_progress_score,
        metadata=dict(metadata or {}),
    )


@dataclass
class FailureDetection:
    failure_type: FailureType
    confidence: float
    local_evidence: list[str]
    retrieval_labels: list[str] = field(default_factory=list)


@dataclass
class DetectionResult:
    should_rescue: bool
    failure_type: FailureType
    confidence: float
    rationale: str
    source: str = "rule"
    recommended_constraint: Optional[str] = None
    raw_response: Optional[str] = None
    local_evidence: list[str] = field(default_factory=list)
    retrieval_labels: list[str] = field(default_factory=list)


@dataclass
class PlaybookInsight:
    insight_id: str
    failure_type: FailureType
    text: str
    created_at: datetime = field(default_factory=utcnow)
    success_count: int = 1
    merge_count: int = 0
    source_summary: Optional[str] = None
    confidence: float = 1.0
    retrieval_label: Optional[str] = None
    task_family: Optional[str] = None
    action_space: Optional[str] = None
    final_score_sum: float = 0.0
    final_score_count: int = 0
    task_success_count: int = 0

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["failure_type"] = self.failure_type.value
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "PlaybookInsight":
        raw = dict(data)
        raw["failure_type"] = parse_failure_type(raw.get("failure_type"), FailureType.NO_FAILURE)
        created = raw.get("created_at")
        raw["created_at"] = (
            datetime.fromisoformat(created)
            if isinstance(created, str) and created
            else utcnow()
        )
        allowed = {item.name for item in fields(cls)}
        raw = {key: value for key, value in raw.items() if key in allowed}
        return cls(**raw)


@dataclass
class PlaybookSection:
    failure_type: FailureType
    title: str
    label_order: list[str] = field(default_factory=list)
    label_definitions: dict[str, str] = field(default_factory=dict)
    label_insights: dict[str, list[PlaybookInsight]] = field(default_factory=dict)
    max_insights: int = 12
    max_chars: int = 2500

    def to_json(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "label_order": list(self.label_order),
            "label_definitions": dict(self.label_definitions),
            "label_insights": {
                label: [i.to_json() for i in insights]
                for label, insights in self.label_insights.items()
            },
            "max_insights": self.max_insights,
            "max_chars": self.max_chars,
        }

    @classmethod
    def from_json(
        cls,
        failure_type: FailureType,
        data: dict[str, Any],
        *,
        max_insights: int,
        max_chars: int,
    ) -> "PlaybookSection":
        label_insights_raw = data.get("label_insights") or {}
        label_insights: dict[str, list[PlaybookInsight]] = {}
        if isinstance(label_insights_raw, dict):
            for label, items in label_insights_raw.items():
                label_insights[str(label)] = [
                    PlaybookInsight.from_json(i)
                    for i in (items or [])
                    if isinstance(i, dict)
                ]
        return cls(
            failure_type=failure_type,
            title=str(data.get("title") or failure_type.value.replace("_", " ").title()),
            label_order=list(data.get("label_order") or []),
            label_definitions=dict(data.get("label_definitions") or {}),
            label_insights=label_insights,
            max_insights=int(data.get("max_insights") or max_insights),
            max_chars=int(data.get("max_chars") or max_chars),
        )


@dataclass
class RescuePrompt:
    failure_type: FailureType
    prompt_text: str
    local_evidence: list[str]
    playbook_context: str


@dataclass
class EscapeResult:
    escaped: bool
    failure_type: FailureType
    confidence: float
    evidence: list[str]
    failure_score_before: float
    failure_score_after: float
    progress_delta: float
    helped_task: bool = False
    task_progress_delta: Optional[float] = None
    resolved_labels: list[str] = field(default_factory=list)
    unresolved_labels: list[str] = field(default_factory=list)
    reflection: Optional[str] = None
    raw_response: Optional[str] = None


@dataclass
class RescueEvent:
    event_id: str
    failure_type: FailureType
    detection: FailureDetection
    rescue_prompt: RescuePrompt
    pre_rescue_window: Any
    detection_result: DetectionResult | None = None
    post_rescue_steps: list[AgentStep] = field(default_factory=list)
