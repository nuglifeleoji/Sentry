from __future__ import annotations

import json
import uuid
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .config import PlaybookConfig
from .models import EscapeResult, PlaybookInsight, PlaybookSection, RescueEvent
from .taxonomy import (
    FailureType,
    PROGRESS_LABEL_DEFINITIONS,
    PROGRESS_RETRIEVAL_LABELS,
    REASONING_LABEL_DEFINITIONS,
    REASONING_RETRIEVAL_LABELS,
    normalize_retrieval_label,
    parse_failure_type,
)

SOFT_FAILURE_TYPES = (
    FailureType.PROGRESS_FAILURE,
    FailureType.REASONING_GROUNDING_FAILURE,
)


def string_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def default_sections(config: PlaybookConfig | None = None) -> dict[FailureType, PlaybookSection]:
    cfg = config or PlaybookConfig()
    return {
        FailureType.PROGRESS_FAILURE: _empty_section(
            FailureType.PROGRESS_FAILURE,
            "Progress Failures",
            PROGRESS_RETRIEVAL_LABELS,
            PROGRESS_LABEL_DEFINITIONS,
            cfg,
        ),
        FailureType.REASONING_GROUNDING_FAILURE: _empty_section(
            FailureType.REASONING_GROUNDING_FAILURE,
            "Reasoning and Grounding Failures",
            REASONING_RETRIEVAL_LABELS,
            REASONING_LABEL_DEFINITIONS,
            cfg,
        ),
    }


class PlaybookManager:
    def __init__(self, config: PlaybookConfig | None = None):
        self.config = config or PlaybookConfig()
        self.sections = default_sections(self.config)
        self.load_if_exists()

    def load_if_exists(self) -> None:
        if not self.config.load_persisted:
            return
        path = Path(self.config.persist_path)
        if not path.exists():
            return
        data = json.loads(path.read_text())
        for key, raw_section in (data.get("sections") or {}).items():
            failure_type = parse_failure_type(key)
            if failure_type not in SOFT_FAILURE_TYPES or not isinstance(raw_section, dict):
                continue
            loaded = PlaybookSection.from_json(
                failure_type,
                raw_section,
                max_insights=self.config.max_insights_per_section,
                max_chars=self.config.max_chars_per_section,
            )
            self.sections[failure_type] = _merge_with_defaults(
                self.sections[failure_type],
                loaded,
            )

    def persist(self) -> None:
        path = Path(self.config.persist_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "0.3",
            "format": "sentry_label_playbook",
            "sections": {
                failure_type.value: section.to_json()
                for failure_type, section in self.sections.items()
            },
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def get_section(self, failure_type: FailureType) -> PlaybookSection:
        if failure_type not in SOFT_FAILURE_TYPES:
            raise ValueError(f"{failure_type.value} has no soft playbook section")
        return self.sections[failure_type]

    def retrieve(
        self,
        failure_type: FailureType,
        labels: list[str] | tuple[str, ...] | None,
    ) -> PlaybookSection:
        section = self.get_section(failure_type)
        selected_labels = _selected_labels(section, labels)
        return PlaybookSection(
            failure_type=section.failure_type,
            title=section.title,
            label_order=selected_labels,
            label_definitions={
                label: section.label_definitions[label]
                for label in selected_labels
                if label in section.label_definitions
            },
            label_insights={
                label: [
                    insight
                    for insight in section.label_insights.get(label, [])
                    if self._passes_retrieval_gate(insight)
                ]
                for label in selected_labels
            },
            max_insights=section.max_insights,
            max_chars=section.max_chars,
        )

    def update_section(
        self,
        failure_type: FailureType,
        candidate: PlaybookInsight,
    ) -> dict[str, Any]:
        section = self.get_section(failure_type)
        label = normalize_retrieval_label(candidate.retrieval_label, failure_type)
        if label is None:
            return {"action": "skipped", "reason": "no_valid_retrieval_label"}

        candidate.failure_type = failure_type
        candidate.retrieval_label = label
        candidate.text = _generalize(candidate.text)
        if not candidate.insight_id:
            candidate.insight_id = f"{failure_type.value}-{label}-{uuid.uuid4().hex[:10]}"

        insights = section.label_insights.setdefault(label, [])
        for existing in insights:
            similarity = string_similarity(candidate.text.lower(), existing.text.lower())
            if similarity >= self.config.dedup_similarity_threshold:
                _merge_insight(existing, candidate)
                return {
                    "action": "merged",
                    "retrieval_label": label,
                    "insight_id": existing.insight_id,
                    "similarity": similarity,
                }

        insights.append(candidate)
        self._compress_label(section, label)
        return {
            "action": "appended",
            "retrieval_label": label,
            "insight_id": candidate.insight_id,
        }

    def _passes_retrieval_gate(self, insight: PlaybookInsight) -> bool:
        cfg = self.config
        min_success = max(1, int(cfg.min_success_count_for_retrieval or 1))
        min_task_success = max(0, int(cfg.min_task_success_count_for_retrieval or 0))
        if int(insight.success_count or 0) < min_success:
            return False
        if int(insight.task_success_count or 0) < min_task_success:
            return False
        min_avg = float(cfg.min_average_final_score_for_retrieval or 0.0)
        if min_avg <= 0.0:
            return True
        if int(insight.final_score_count or 0) <= 0:
            return False
        return _average_final_score(insight) >= min_avg

    def _compress_label(self, section: PlaybookSection, label: str) -> None:
        insights = section.label_insights.get(label, [])
        if len(insights) <= section.max_insights and _insight_chars(insights) <= section.max_chars:
            return
        insights.sort(
            key=lambda item: (item.success_count, item.confidence, -item.merge_count),
            reverse=True,
        )
        kept: list[PlaybookInsight] = []
        used = 0
        for item in insights:
            if len(kept) >= section.max_insights:
                break
            if used + len(item.text) > section.max_chars and kept:
                continue
            kept.append(item)
            used += len(item.text)
        section.label_insights[label] = kept


def build_candidate_insights(event: RescueEvent, result: EscapeResult) -> list[PlaybookInsight]:
    reflection = str(result.reflection or "").strip()
    if not reflection:
        return []
    context = _context_labels(event)
    candidates: list[PlaybookInsight] = []
    for label in _resolved_labels(event, result):
        candidates.append(
            PlaybookInsight(
                insight_id=f"{event.failure_type.value}-{label}-{uuid.uuid4().hex[:10]}",
                failure_type=event.failure_type,
                retrieval_label=label,
                text=reflection,
                source_summary="; ".join(result.evidence[:3]),
                confidence=result.confidence,
                task_family=context.get("task_family"),
                action_space=context.get("action_space"),
            )
        )
    return candidates


def _empty_section(
    failure_type: FailureType,
    title: str,
    labels: tuple[str, ...],
    definitions: dict[str, str],
    config: PlaybookConfig,
) -> PlaybookSection:
    return PlaybookSection(
        failure_type=failure_type,
        title=title,
        label_order=list(labels),
        label_definitions=dict(definitions),
        label_insights={},
        max_insights=config.max_insights_per_section,
        max_chars=config.max_chars_per_section,
    )


def _merge_with_defaults(default: PlaybookSection, loaded: PlaybookSection) -> PlaybookSection:
    label_insights: dict[str, list[PlaybookInsight]] = {}
    for label, insights in loaded.label_insights.items():
        normalized = normalize_retrieval_label(label, loaded.failure_type)
        if normalized is not None:
            label_insights[normalized] = list(insights)
    return PlaybookSection(
        failure_type=default.failure_type,
        title=loaded.title or default.title,
        label_order=list(default.label_order),
        label_definitions=dict(default.label_definitions),
        label_insights=label_insights,
        max_insights=loaded.max_insights or default.max_insights,
        max_chars=loaded.max_chars or default.max_chars,
    )


def _selected_labels(
    section: PlaybookSection,
    labels: list[str] | tuple[str, ...] | None,
) -> list[str]:
    source = labels if labels is not None else section.label_order
    out: list[str] = []
    for item in source:
        label = normalize_retrieval_label(item, section.failure_type)
        if label is None or label in out:
            continue
        if label in section.label_definitions or label in section.label_insights:
            out.append(label)
    return out


def _resolved_labels(event: RescueEvent, result: EscapeResult) -> list[str]:
    out: list[str] = []
    for item in result.resolved_labels or []:
        label = normalize_retrieval_label(item, event.failure_type)
        if label is not None and label not in out:
            out.append(label)
    return out


def _merge_insight(existing: PlaybookInsight, candidate: PlaybookInsight) -> None:
    existing.success_count += max(1, int(candidate.success_count or 1))
    existing.merge_count += 1
    existing.confidence = max(existing.confidence, candidate.confidence)
    existing.final_score_sum += float(candidate.final_score_sum or 0.0)
    existing.final_score_count += int(candidate.final_score_count or 0)
    existing.task_success_count += int(candidate.task_success_count or 0)


def _context_labels(event: RescueEvent) -> dict[str, str]:
    steps = list(getattr(event.pre_rescue_window, "steps", []) or [])
    latest = steps[-1] if steps else None
    metadata: dict[str, Any] = dict(getattr(latest, "metadata", {}) or {})
    return {
        key: str(metadata.get(key) or "").strip()
        for key in ("action_space", "task_family")
        if str(metadata.get(key) or "").strip()
    }


def _generalize(text: str) -> str:
    return " ".join(str(text or "").split())[:500]


def _insight_chars(insights: list[PlaybookInsight]) -> int:
    return sum(len(item.text) for item in insights)


def _average_final_score(insight: PlaybookInsight) -> float:
    count = int(insight.final_score_count or 0)
    if count <= 0:
        return 0.0
    return float(insight.final_score_sum or 0.0) / count
