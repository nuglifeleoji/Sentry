from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


@dataclass
class TraceConfig:
    window_size: int = 5
    post_rescue_horizon: int = 10


@dataclass
class InterventionConfig:
    min_confidence: float = 0.65
    max_rescues_per_task: int = 5
    cooldown_steps_after_rescue: int = 2
    allow_action_validity_during_cooldown: bool = True
    max_failed_rescues_per_key: int = 1


@dataclass
class PlaybookConfig:
    enable_updates: bool = True
    load_persisted: bool = True
    max_insights_per_section: int = 12
    max_chars_per_section: int = 2500
    dedup_similarity_threshold: float = 0.82
    min_success_count_for_retrieval: int = 1
    min_average_final_score_for_retrieval: float = 0.0
    min_task_success_count_for_retrieval: int = 0
    persist_path: str = "./rescue_playbook.json"


@dataclass
class GuardJudgeConfig:
    min_confidence: float = 0.65
    max_window_steps: int = 5
    max_observation_chars: int = 900


@dataclass
class LoggingConfig:
    enabled: bool = True
    log_path: str | None = None


@dataclass
class SentryConfig:
    trace: TraceConfig = field(default_factory=TraceConfig)
    intervention: InterventionConfig = field(default_factory=InterventionConfig)
    playbook: PlaybookConfig = field(default_factory=PlaybookConfig)
    guard_judge: GuardJudgeConfig = field(default_factory=GuardJudgeConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "SentryConfig":
        raw = raw or {}
        trace = _dataclass_from_dict(TraceConfig, raw.get("trace"))
        intervention = _dataclass_from_dict(InterventionConfig, raw.get("intervention"))

        playbook = _dataclass_from_dict(PlaybookConfig, raw.get("playbook"))
        guard_judge = _dataclass_from_dict(GuardJudgeConfig, raw.get("guard_judge"))
        logging = _dataclass_from_dict(LoggingConfig, raw.get("logging"))
        return cls(
            trace=trace,
            intervention=intervention,
            playbook=playbook,
            guard_judge=guard_judge,
            logging=logging,
        )


def _dataclass_from_dict(cls, raw: Any):
    if not isinstance(raw, dict):
        return cls()
    allowed = {item.name for item in fields(cls)}
    return cls(**{key: value for key, value in raw.items() if key in allowed})
