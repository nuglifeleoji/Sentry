from __future__ import annotations

import copy
import uuid
from collections.abc import Callable
from typing import Any

from .config import SentryConfig
from .detector import SentryDetector, detect_action_validity
from .logging import JSONLLogger, to_jsonable
from .models import (
    AgentStep,
    DetectionResult,
    EscapeResult,
    FailureDetection,
    LocalTraceWindow,
    RescueEvent,
    RescuePrompt,
    TaskContext,
)
from .playbook import PlaybookManager
from .router import route_rescue
from .taxonomy import FailureType
from .updater import PlaybookUpdater
from .verifier import EscapeVerifier, verify_hard_repair


class Sentry:
    """External failure manager implementing Sections 3.1--3.3.

    Task-local state is reset by :meth:`start_task` / :meth:`finalize_task`.
    The playbook is intentionally retained across tasks and changes only after a
    verified soft recovery.

    In-memory event capture is opt-in. Hosts enabling ``capture_runtime_events``
    must drain events regularly; file logging is independent of this buffer.
    """

    def __init__(
        self,
        config: SentryConfig | None = None,
        *,
        detector: Any | None = None,
        guard_judge_callback: Callable[[str], str] | None = None,
        verifier_judge_callback: Callable[[str], str] | None = None,
        updater_judge_callback: Callable[[str], str] | None = None,
        capture_runtime_events: bool = False,
    ):
        if type(capture_runtime_events) is not bool:
            raise ValueError("capture_runtime_events must be a boolean")
        self.config = config or SentryConfig()
        self.window = LocalTraceWindow(max_size=self.config.trace.window_size)
        self.detector = detector or SentryDetector(guard_judge_callback, self.config)
        self.verifier = EscapeVerifier(verifier_judge_callback, self.config)
        self.updater = PlaybookUpdater(updater_judge_callback)
        self.playbook = PlaybookManager(self.config.playbook, self.config.retrieval)
        self.task: TaskContext | None = None
        self.active_rescue_event: RescueEvent | None = None
        self._pending_soft_event: RescueEvent | None = None
        self._cycle_count = 0
        self._last_diagnosed_cycle = -1
        self.last_escape_result: EscapeResult | None = None
        self._runtime_events: list[dict[str, Any]] | None = [] if capture_runtime_events else None
        log_path = self.config.logging.log_path if self.config.logging.enabled else None
        self.logger = JSONLLogger(log_path)

    def start_task(self, *, objective: str, action_schema: str) -> None:
        task = TaskContext(objective=objective, action_schema=action_schema)
        if self.task is not None:
            self.finalize_task()
        self.task = task
        self.window.clear()
        self.active_rescue_event = None
        self._pending_soft_event = None
        self._cycle_count = 0
        self._last_diagnosed_cycle = -1
        self.last_escape_result = None
        self._log({"type": "task_started", "objective": objective})

    def observe_step(self, step: AgentStep) -> None:
        self._require_task()
        self.window.add(step)
        self._cycle_count += 1
        self._log({"type": "agent_step", "step_id": step.step_id, "step": step})
        active = self.active_rescue_event
        hard = active if active is not None and active.is_hard_repair else None
        soft = self._pending_soft_event or (active if hard is None else None)
        if soft is not None:
            soft.post_rescue_steps.append(copy.deepcopy(step))
        if hard is not None:
            hard.post_rescue_steps.append(copy.deepcopy(step))
            self._complete_hard_verification(step)
        if soft is not None:
            horizon = self.config.trace.post_rescue_horizon
            if len(soft.post_rescue_steps) == horizon:
                self._complete_soft_verification(soft)
            elif len(soft.post_rescue_steps) > horizon:
                raise RuntimeError("Sentry missed the exact H-step verification boundary")

    def maybe_rescue(self) -> RescuePrompt | None:
        task = self._require_task()
        if not self.window.steps or self._last_diagnosed_cycle == self._cycle_count:
            return None
        active = self.active_rescue_event
        if active is not None:
            if active.is_hard_repair:
                return None
            # While soft verification is pending, schema-first hard repair must
            # remain available. Do not inject another soft intervention.
            result = detect_action_validity(self.window.steps[-1])
            if result is None:
                return None
        else:
            # Schema-first routing is a runtime invariant, including when a
            # caller supplies its own soft-failure detector.
            result = detect_action_validity(self.window.steps[-1])
            if result is None:
                result = self.detector.detect(self.window, task)
        self._last_diagnosed_cycle = self._cycle_count
        self._log_detection(result)
        if not result.should_rescue:
            return None
        detection = FailureDetection(
            failure_type=result.failure_type,
            local_evidence=list(result.local_evidence),
            retrieval_labels=list(result.retrieval_labels),
            diagnosis=result.rationale,
        )
        route, prompt = route_rescue(
            detection=detection,
            detection_result=result,
            window=self.window,
            task=task,
            playbook=self.playbook,
        )
        if prompt is None:
            return None
        event = RescueEvent(
            event_id=uuid.uuid4().hex,
            failure_type=detection.failure_type,
            detection=detection,
            detection_result=result,
            rescue_prompt=prompt,
            pre_rescue_window=self.window.copy(),
        )
        if active is not None:
            self._pending_soft_event = active
        self.active_rescue_event = event
        self._log(
            {
                "type": "rescue_injected",
                "event_id": event.event_id,
                "failure_type": event.failure_type,
                "retrieval_labels": detection.retrieval_labels,
                "route": route,
                "prompt": prompt.prompt_text,
                "retrieved_entry_ids": list(prompt.retrieved_entry_ids),
            }
        )
        return prompt

    def finalize_task(self) -> None:
        """End the task without exposing task outcomes to Sentry.

        An incomplete H-step recovery is not verified and can never update the
        playbook.
        """

        events = [self.active_rescue_event, self._pending_soft_event]
        for event in events:
            if event is not None:
                self._log({
                    "type": "verification_incomplete",
                    "event_id": event.event_id,
                    "observed_cycles": len(event.post_rescue_steps),
                    "required_cycles": 1 if event.is_hard_repair else self.config.trace.post_rescue_horizon,
                })
        self.active_rescue_event = None
        self._pending_soft_event = None
        self.last_escape_result = None
        self._cycle_count = 0
        self._last_diagnosed_cycle = -1
        self.window.clear()
        self.task = None
        self._log({"type": "task_finalized"})

    def reset_playbook(self) -> None:
        self.playbook.clear()

    def _complete_hard_verification(self, corrected_step: AgentStep) -> None:
        event = self.active_rescue_event
        if event is None:
            return
        result = verify_hard_repair(event, corrected_step)
        self.last_escape_result = result
        self._log_escape(event, result)
        # Hard repair is never written to the playbook.
        self.active_rescue_event = self._pending_soft_event
        self._pending_soft_event = None

    def _complete_soft_verification(self, event: RescueEvent) -> None:
        task = self._require_task()
        # Detach at H even if a model provider fails: future cycles must not
        # extend an already completed recovery window.
        if self.active_rescue_event is event:
            self.active_rescue_event = None
        if self._pending_soft_event is event:
            self._pending_soft_event = None
        try:
            result = self.verifier.verify(event, task)
        except Exception:
            self._log({"type": "verification_error", "event_id": event.event_id})
            raise
        self.last_escape_result = result
        self._log_escape(event, result)
        if result.escaped:
            try:
                candidate = self.updater.summarize(event, result)
            except Exception:
                self._log({
                    "type": "playbook_update_skipped",
                    "event_id": event.event_id,
                    "reason": "updater_error",
                })
                raise
            if candidate is None:
                self._log(
                    {
                        "type": "playbook_update_skipped",
                        "event_id": event.event_id,
                        "reason": "invalid_structured_update",
                    }
                )
            else:
                inserted = self.playbook.add(candidate)
                self._log(
                    {
                        "type": "playbook_update",
                        "event_id": event.event_id,
                        "entry": inserted,
                    }
                )

    def _require_task(self) -> TaskContext:
        if self.task is None:
            raise RuntimeError("Call start_task(objective=..., action_schema=...) first")
        return self.task

    def _log_detection(self, result: DetectionResult) -> None:
        self._log(
            {
                "type": "sentry_diagnosis",
                "failure_type": result.failure_type,
                "intervention_needed": result.should_rescue,
                "retrieval_labels": result.retrieval_labels,
                "local_evidence": result.local_evidence,
                "diagnosis": result.rationale,
                "source": result.source,
            }
        )

    def _log_escape(self, event: RescueEvent, result: EscapeResult) -> None:
        self._log(
            {
                "type": "escape_result",
                "event_id": event.event_id,
                "failure_type": event.failure_type,
                "resolved": result.escaped,
                "observed_cycles": len(event.post_rescue_steps),
                "evidence": result.evidence,
                "justification": result.justification,
            }
        )

    def _log(self, event: dict[str, Any]) -> None:
        if self._runtime_events is not None:
            self._runtime_events.append(to_jsonable(event))
        self.logger.write(event)

    def drain_runtime_events(self) -> list[dict[str, Any]]:
        """Transfer captured events to the caller, or return [] if capture is off.

        With ``capture_runtime_events=True``, drain after each cycle and after
        finalization. Events are retained until drained, including across tasks.
        """
        if self._runtime_events is None:
            return []
        events = self._runtime_events
        self._runtime_events = []
        return events
