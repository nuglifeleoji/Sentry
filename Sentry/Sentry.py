from __future__ import annotations

import uuid
from typing import Any, Callable

from .config import SentryConfig
from .detector import SentryDetector
from .logging import JSONLLogger
from .models import (
    AgentStep,
    DetectionResult,
    EscapeResult,
    FailureDetection,
    LocalTraceWindow,
    PlaybookInsight,
    RescueEvent,
    RescuePrompt,
)
from .playbook import PlaybookManager, build_candidate_insights
from .router import route_rescue
from .taxonomy import FailureType, labels_for_failure_type, normalize_retrieval_label
from .verifier import EscapeVerifier


class Sentry:
    def __init__(
        self,
        config: SentryConfig | None = None,
        *,
        detector: Any | None = None,
        guard_judge_callback: Callable[[str], str] | None = None,
        verifier_judge_callback: Callable[[str], str] | None = None,
    ):
        self.config = config or SentryConfig()
        self.window = LocalTraceWindow(max_size=self.config.trace.window_size)
        self.detector = detector or SentryDetector(guard_judge_callback, self.config)
        self.playbook = PlaybookManager(self.config.playbook)
        self.escape_verifier = EscapeVerifier(verifier_judge_callback, self.config)
        self.active_rescue_event: RescueEvent | None = None
        self.cooldown_steps_remaining = 0
        self.num_rescues_this_task = 0
        self.failed_rescue_counts: dict[str, int] = {}
        self.applied_playbook_update_keys_this_task: set[str] = set()
        log_path = self.config.logging.log_path if self.config.logging.enabled else None
        self.logger = JSONLLogger(log_path)

    def observe_step(self, step: AgentStep) -> None:
        self.window.add(step)
        self._log({"type": "agent_step", "step_id": step.step_id, "step": step})
        if self.active_rescue_event is not None:
            self.active_rescue_event.post_rescue_steps.append(step)
            self._maybe_verify_active_rescue()

    def maybe_rescue(self) -> RescuePrompt | None:
        if self.active_rescue_event is not None or not self.window.steps:
            return None

        detection_result = self._diagnose_candidate()
        detection = self._detection_from_result(detection_result)
        if not detection_result.should_rescue:
            if self.cooldown_steps_remaining > 0:
                self.cooldown_steps_remaining -= 1
            return None

        if self.num_rescues_this_task >= self.config.intervention.max_rescues_per_task:
            self._log_shadow_decision(detection=detection, reason="rescue_budget_exhausted")
            return None

        rescue_key = self._rescue_key(detection)
        max_failed = self.config.intervention.max_failed_rescues_per_key
        failed_count = self.failed_rescue_counts.get(rescue_key, 0)
        if max_failed >= 0 and failed_count >= max_failed:
            self._log(
                {
                    "type": "rescue_suppressed",
                    "step_id": self.window.steps[-1].step_id,
                    "failure_type": detection.failure_type,
                    "reason": "previous_rescue_failed_for_same_key",
                    "rescue_key": rescue_key,
                    "failed_count": failed_count,
                }
            )
            return None

        critical_format = (
            detection.failure_type == FailureType.ACTION_VALIDITY_FAILURE
            and self.config.intervention.allow_action_validity_during_cooldown
        )
        if self.cooldown_steps_remaining > 0 and not critical_format:
            self.cooldown_steps_remaining -= 1
            self._log_shadow_decision(
                detection=detection,
                reason="cooldown_active",
                extra={"cooldown_steps_remaining": self.cooldown_steps_remaining},
            )
            return None

        intervention_type, rescue_prompt = route_rescue(
            detection=detection,
            detection_result=detection_result,
            window=self.window,
            playbook=self.playbook,
        )
        self._log(
            {
                "type": "intervention_strategy",
                "step_id": self.window.steps[-1].step_id,
                "failure_type": detection.failure_type,
                "intervention_type": intervention_type,
            }
        )
        if rescue_prompt is None:
            self._log_shadow_decision(
                detection=detection,
                reason="strategy_no_intervention",
                extra={
                    "intervention_type": intervention_type,
                    "detection_source": detection_result.source,
                },
            )
            return None

        self.active_rescue_event = RescueEvent(
            event_id=uuid.uuid4().hex,
            failure_type=detection.failure_type,
            detection=detection,
            detection_result=detection_result,
            rescue_prompt=rescue_prompt,
            pre_rescue_window=self.window.copy(),
        )
        self.active_rescue_event.rescue_key = rescue_key
        self.num_rescues_this_task += 1
        self.cooldown_steps_remaining = self.config.intervention.cooldown_steps_after_rescue
        event = {
            "type": "rescue_injected",
            "event_id": self.active_rescue_event.event_id,
            "step_id": self.window.steps[-1].step_id,
            "failure_type": detection.failure_type,
            "confidence": detection.confidence,
            "evidence": detection.local_evidence,
            "retrieval_labels": detection.retrieval_labels,
            "detection_result": detection_result,
            "intervention_type": intervention_type,
            "prompt": rescue_prompt.prompt_text,
        }
        self._log(event)
        return rescue_prompt

    def _diagnose_candidate(self) -> DetectionResult:
        latest_step = self.window.steps[-1]
        detection_result = self.detector.detect(self.window)
        self._log(
            {
                "type": "sentry_diagnosis",
                "step_id": latest_step.step_id,
                "failure_type": detection_result.failure_type,
                "confidence": detection_result.confidence,
                "should_rescue": detection_result.should_rescue,
                "source": detection_result.source,
                "retrieval_labels": detection_result.retrieval_labels,
                "evidence": detection_result.local_evidence,
                "rationale": detection_result.rationale,
            }
        )
        return detection_result

    def _detection_from_result(
        self,
        detection_result: DetectionResult,
    ) -> FailureDetection:
        return FailureDetection(
            failure_type=detection_result.failure_type,
            confidence=detection_result.confidence,
            local_evidence=list(detection_result.local_evidence)
            or ([detection_result.rationale] if detection_result.rationale else []),
            retrieval_labels=self._labels_after_detection_result(
                detection_result.retrieval_labels,
                detection_result.failure_type,
            ),
        )

    def finalize_task(
        self,
        *,
        final_task_score: float | None = None,
        task_success: bool | None = None,
    ) -> None:
        if self.active_rescue_event is not None and self.active_rescue_event.post_rescue_steps:
            self._maybe_verify_active_rescue(
                force=True,
                final_task_score=final_task_score,
                task_success=task_success,
            )

    def _maybe_verify_active_rescue(
        self,
        *,
        force: bool = False,
        final_task_score: float | None = None,
        task_success: bool | None = None,
    ) -> None:
        event = self.active_rescue_event
        if event is None:
            return
        if not force and len(event.post_rescue_steps) < self.config.trace.post_rescue_horizon:
            return
        result = self.escape_verifier.verify(event)
        escape_event = {
            "type": "escape_result",
            "event_id": event.event_id,
            "failure_type": event.failure_type,
            "escaped": result.escaped,
            "confidence": result.confidence,
            "evidence": result.evidence,
            "failure_score_before": result.failure_score_before,
            "failure_score_after": result.failure_score_after,
            "progress_delta": result.progress_delta,
            "helped_task": result.helped_task,
            "task_progress_delta": result.task_progress_delta,
            "resolved_labels": result.resolved_labels,
            "unresolved_labels": result.unresolved_labels,
            "reflection": result.reflection,
        }
        self._log(escape_event)
        if not result.escaped:
            rescue_key = getattr(event, "rescue_key", self._rescue_key(event.detection))
            self.failed_rescue_counts[rescue_key] = (
                self.failed_rescue_counts.get(rescue_key, 0) + 1
            )
        if (
            event.failure_type != FailureType.ACTION_VALIDITY_FAILURE
            and result.escaped
            and self.config.playbook.enable_updates
        ):
            candidates = build_candidate_insights(event, result)
            if not candidates:
                skipped_event = {
                    "type": "playbook_update_skipped",
                    "event_id": event.event_id,
                    "failure_type": event.failure_type,
                    "reason": "no_verifier_reflection_or_resolved_label",
                }
                self._log(skipped_event)
            else:
                for candidate in candidates:
                    self._apply_playbook_update(
                        event,
                        result,
                        candidate,
                        final_task_score=final_task_score,
                        task_success=task_success,
                    )
        elif result.escaped and not self.config.playbook.enable_updates:
            skipped_event = {
                "type": "playbook_update_skipped",
                "event_id": event.event_id,
                "failure_type": event.failure_type,
                "reason": "playbook_updates_disabled",
            }
            self._log(skipped_event)
        self.active_rescue_event = None

    def _apply_playbook_update(
        self,
        event: RescueEvent,
        result: EscapeResult,
        candidate: PlaybookInsight,
        *,
        final_task_score: float | None = None,
        task_success: bool | None = None,
    ) -> None:
        update_key = self._playbook_update_key(event.failure_type, candidate)
        if update_key in self.applied_playbook_update_keys_this_task:
            skipped_event = {
                "type": "playbook_update_skipped",
                "event_id": event.event_id,
                "failure_type": event.failure_type,
                "reason": "duplicate_task_level_pattern",
                "final_task_score": final_task_score,
                "task_success": task_success,
            }
            self._log(skipped_event)
            return
        self.applied_playbook_update_keys_this_task.add(update_key)
        self._record_candidate_task_outcome(
            candidate,
            final_task_score=final_task_score,
            task_success=task_success,
        )
        update = self.playbook.update_section(event.failure_type, candidate)
        self.playbook.persist()
        update_event = {
            "type": "playbook_update",
            "event_id": event.event_id,
            "failure_type": event.failure_type,
            "retrieval_label": candidate.retrieval_label,
            "candidate": candidate,
            "final_task_score": final_task_score,
            "task_success": task_success,
            "task_progress_delta": result.task_progress_delta,
            **update,
        }
        self._log(update_event)

    def _record_candidate_task_outcome(
        self,
        candidate: PlaybookInsight,
        *,
        final_task_score: float | None,
        task_success: bool | None,
    ) -> None:
        if final_task_score is not None:
            try:
                candidate.final_score_sum += float(final_task_score)
                candidate.final_score_count += 1
            except (TypeError, ValueError):
                pass
        if bool(task_success):
            candidate.task_success_count += 1

    def _playbook_update_key(
        self,
        failure_type: FailureType,
        candidate: PlaybookInsight,
    ) -> str:
        text = " ".join(str(candidate.text or "").lower().split())
        label = str(candidate.retrieval_label or "")
        return f"{failure_type.value}:{label}:{text[:240]}"

    def _labels_after_detection_result(
        self,
        labels: list[str],
        failure_type: FailureType,
    ) -> list[str]:
        normalized = []
        for item in labels or []:
            label = normalize_retrieval_label(item, failure_type)
            if label is not None and label not in normalized:
                normalized.append(label)
        if normalized:
            return normalized
        allowed = labels_for_failure_type(failure_type)
        return [allowed[0]] if allowed else []

    def _log(self, event: dict) -> None:
        self.logger.write(event)

    def _log_shadow_decision(
        self,
        *,
        detection: FailureDetection,
        reason: str,
        extra: dict | None = None,
    ) -> None:
        latest = self.window.steps[-1] if self.window.steps else None
        event = {
            "type": "shadow_decision",
            "step_id": latest.step_id if latest is not None else None,
            "failure_type": detection.failure_type,
            "confidence": detection.confidence,
            "reason": reason,
            "would_intervene": True,
            "evidence": list(detection.local_evidence),
        }
        if extra:
            event.update(extra)
        self._log(event)

    def _rescue_key(self, detection: FailureDetection) -> str:
        latest = self.window.steps[-1] if self.window.steps else None
        if latest is None:
            return detection.failure_type.value
        metadata = latest.metadata or {}
        signature = metadata.get("action_signature")
        soft_repeat = metadata.get("soft_repeat_signal")
        if not signature and isinstance(soft_repeat, dict):
            signature = soft_repeat.get("action_signature")
        if signature:
            return f"{detection.failure_type.value}:{signature}"
        tool = latest.action.tool_name or "raw"
        raw = (latest.action.raw or "")[:160]
        return f"{detection.failure_type.value}:{tool}:{raw}"
