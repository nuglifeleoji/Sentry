from __future__ import annotations

import json
import uuid
from collections.abc import Callable

from ._structured import json_object, nonempty_text, string_list
from .detector import render_trajectory
from .models import EscapeResult, PlaybookInsight, RescueEvent
from .taxonomy import labels_for_failure_type


class PlaybookUpdater:
    def __init__(self, judge_callback: Callable[[str], str] | None):
        self.judge_callback = judge_callback

    def summarize(
        self, event: RescueEvent, result: EscapeResult
    ) -> PlaybookInsight | None:
        if not result.escaped or event.is_hard_repair or result.failure_type != event.failure_type:
            return None
        if self.judge_callback is None:
            raise RuntimeError("Playbook updating requires the Sentry model callback")
        raw = self.judge_callback(build_update_prompt(event, result))
        return parse_update_response(raw, event)


def build_update_prompt(event: RescueEvent, result: EscapeResult) -> str:
    trigger = "\n".join(f"- {item}" for item in event.detection.local_evidence)
    evidence = "\n".join(f"- {item}" for item in result.evidence)
    return f"""[SENTRY PLAYBOOK UPDATE]
Role.
You summarize only verified successful rescues into reusable playbook entries. Do not store generic reflections or advice that is not supported by the local trajectory.

Verified rescue episode.
Failure type: {event.failure_type.value}
Retrieval labels: {', '.join(event.detection.retrieval_labels)}
Trigger pattern:
{trigger}
Repair guidance:
{event.rescue_prompt.prompt_text}
Post-repair trajectory:
{render_trajectory(event.post_rescue_steps)}
Post-repair evidence:
{evidence}

Output format.
Return only this JSON object with fixed fields:
{{
  "failure_type": "{event.failure_type.value}",
  "retrieval_labels": {json.dumps(event.detection.retrieval_labels)},
  "subtype": "concise human-readable subtype",
  "trigger_pattern": "reusable trigger pattern grounded in this episode",
  "repair_principle": "reusable repair principle grounded in this episode"
}}
The entry must be specific enough to help a future similar failure, but not so narrow that it applies only to the original trajectory.
"""


def parse_update_response(raw: str, event: RescueEvent) -> PlaybookInsight | None:
    data = json_object(raw, {
        "failure_type", "retrieval_labels", "subtype", "trigger_pattern", "repair_principle",
    })
    if data is None or event.is_hard_repair:
        return None
    if data["failure_type"] != event.failure_type.value:
        return None
    labels = data["retrieval_labels"]
    if not string_list(labels, allow_empty=False):
        return None
    if len(set(labels)) != len(labels):
        return None
    if any(label not in labels_for_failure_type(event.failure_type) for label in labels):
        return None
    if set(labels) != set(event.detection.retrieval_labels):
        return None
    if not all(nonempty_text(data[key]) for key in ("subtype", "trigger_pattern", "repair_principle")):
        return None
    return PlaybookInsight(
        insight_id=f"sentry-{uuid.uuid4().hex[:10]}",
        failure_type=event.failure_type,
        retrieval_labels=list(event.detection.retrieval_labels),
        trigger_pattern=data["trigger_pattern"].strip(),
        repair_principle=data["repair_principle"].strip(),
        subtype=data["subtype"].strip(),
    )
