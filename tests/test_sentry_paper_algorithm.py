from __future__ import annotations

import json

from Sentry import (
    FailureType,
    PlaybookInsight,
    Sentry,
    SentryConfig,
    agent_step_from_parts,
    model_to_text_judge,
)
from Sentry.playbook import PlaybookManager


SCHEMA = "Action: search[query] | click[id] | finish[answer]"


def step(
    number: int,
    *,
    action: str = "search[x]",
    observation: str = "new evidence",
    parsed: bool = True,
    valid: bool = True,
    accepted: bool | None = True,
    score: float | None = None,
):
    return agent_step_from_parts(
        step_id=number,
        reasoning=f"reasoning-{number}",
        raw_action=action,
        observation=observation,
        parsed_ok=parsed,
        schema_valid=valid,
        accepted_by_environment=accepted,
        parser_error=None if parsed else "parse failed",
        schema_error=None if valid else "schema failed",
        task_progress_score=score,
        metadata={"benchmark_reward": 999, "final_task_success": True},
    )


def test_full_sentry_constants_are_the_paper_defaults():
    config = SentryConfig()
    assert config.trace.window_size == 5
    assert config.trace.post_rescue_horizon == 10
    assert config.retrieval.top_k == 5
    assert config.retrieval.primary_label_cap == 2
    assert config.model.model == "Qwen3.5-9B"
    assert config.model.temperature == 0
    assert config.model.max_tokens == 4096
    assert config.playbook.persist_path is None
    assert Sentry(config).playbook.entries == []


def test_hard_repair_is_schema_first_immediate_and_never_updates_playbook():
    def forbidden(_: str) -> str:
        raise AssertionError("hard repair must not invoke soft detector/verifier/updater")

    sentry = Sentry(
        guard_judge_callback=forbidden,
        verifier_judge_callback=forbidden,
        updater_judge_callback=forbidden,
    )
    sentry.start_task(objective="find x", action_schema=SCHEMA)
    sentry.observe_step(step(0, action="bad", parsed=False, valid=False, accepted=False))

    prompt = sentry.maybe_rescue()
    assert prompt is not None and prompt.route == "hard_repair"
    assert "Output only the corrected action" in prompt.prompt_text

    sentry.observe_step(step(1, action="search[x]", accepted=True))
    assert sentry.last_escape_result is not None
    assert sentry.last_escape_result.resolved
    assert sentry.active_rescue_event is None
    assert sentry.playbook.entries == []


def test_hard_repair_requires_explicit_environment_acceptance():
    sentry = Sentry(guard_judge_callback=lambda _: "")
    sentry.start_task(objective="find x", action_schema=SCHEMA)
    sentry.observe_step(step(0, parsed=False, valid=False, accepted=False))
    assert sentry.maybe_rescue() is not None
    sentry.observe_step(step(1, accepted=None))
    assert sentry.last_escape_result is not None
    assert not sentry.last_escape_result.resolved


def test_soft_repair_waits_exactly_h_and_updates_only_after_resolution():
    calls: dict[str, list[str]] = {"detector": [], "verifier": [], "updater": []}

    def detector(prompt: str) -> str:
        calls["detector"].append(prompt)
        return json.dumps(
            {
                "intervention_needed": True,
                "failure_type": "progress_failure",
                "retrieval_labels": ["repetition_or_looping"],
                "local_evidence": ["search[x] repeated without new information"],
                "diagnosis": "The trajectory is looping.",
            }
        )

    def verifier(prompt: str) -> str:
        calls["verifier"].append(prompt)
        return json.dumps(
            {
                "status": "resolved",
                "justification": "The repeated search stopped and new grounded evidence appeared.",
                "evidence": ["The agent clicked a result and observed its details."],
            }
        )

    def updater(prompt: str) -> str:
        calls["updater"].append(prompt)
        return json.dumps(
            {
                "failure_type": "progress_failure",
                "retrieval_labels": ["repetition_or_looping"],
                "subtype": "Repetition or looping",
                "trigger_pattern": "A search is repeated after returning no new evidence.",
                "repair_principle": "Open a grounded candidate instead of repeating the search.",
            }
        )

    sentry = Sentry(
        guard_judge_callback=detector,
        verifier_judge_callback=verifier,
        updater_judge_callback=updater,
    )
    sentry.start_task(objective="find x", action_schema=SCHEMA)
    for index in range(5):
        sentry.observe_step(
            step(index, observation="same", score=12345 + index)
        )
    rescue = sentry.maybe_rescue()
    assert rescue is not None and rescue.route == "soft_repair"
    assert len(calls["detector"]) == 1
    assert "12345" not in calls["detector"][0]
    assert "benchmark_reward" not in calls["detector"][0]

    for index in range(5, 14):
        sentry.observe_step(step(index, action=f"click[{index}]", score=99999))
    assert calls["verifier"] == []
    assert sentry.playbook.entries == []

    sentry.observe_step(step(14, action="finish[x]", score=99999))
    assert len(calls["verifier"]) == 1
    assert len(calls["updater"]) == 1
    verifier_prompt = calls["verifier"][0]
    assert "reasoning-0" not in verifier_prompt
    assert "[SENTRY SOFT REPAIR]" not in verifier_prompt
    assert "99999" not in verifier_prompt
    assert "benchmark_reward" not in verifier_prompt
    for prompt in calls["detector"] + calls["verifier"] + calls["updater"]:
        assert "task_progress_score" not in prompt
        assert "benchmark_reward" not in prompt
        assert "final_task_success" not in prompt
    assert len(sentry.playbook.entries) == 1
    entry = sentry.playbook.entries[0]
    assert entry.failure_type == FailureType.PROGRESS_FAILURE
    assert entry.retrieval_labels == ["repetition_or_looping"]
    assert entry.insertion_step == 1


def test_incomplete_soft_repair_is_not_verified_or_stored():
    detector_output = json.dumps(
        {
            "intervention_needed": True,
            "failure_type": "reasoning_grounding_failure",
            "retrieval_labels": ["hallucination"],
            "local_evidence": ["unsupported claim"],
            "diagnosis": "unsupported",
        }
    )
    calls = []
    sentry = Sentry(
        guard_judge_callback=lambda _: detector_output,
        verifier_judge_callback=lambda prompt: calls.append(prompt) or "{}",
        updater_judge_callback=lambda _: "{}",
    )
    sentry.start_task(objective="verify x", action_schema=SCHEMA)
    sentry.observe_step(step(0))
    assert sentry.maybe_rescue() is not None
    for index in range(1, 10):
        sentry.observe_step(step(index, score=1.0))
    sentry.finalize_task()
    assert calls == []
    assert sentry.playbook.entries == []


def test_retrieval_uses_overlap_recency_diversity_and_same_type_fallback():
    manager = PlaybookManager()

    def add(labels: list[str], name: str):
        manager.add(
            PlaybookInsight(
                insight_id=name,
                failure_type=FailureType.PROGRESS_FAILURE,
                retrieval_labels=labels,
                trigger_pattern=f"trigger-{name}",
                repair_principle=f"repair-{name}",
            )
        )

    add(["repetition_or_looping", "planning_stall"], "both-old")
    add(["repetition_or_looping", "planning_stall"], "both-new")
    add(["repetition_or_looping"], "repeat-only")
    add(["planning_stall"], "plan-1")
    add(["planning_stall"], "plan-2")
    add(["planning_stall"], "plan-3")
    result = manager.retrieve(
        FailureType.PROGRESS_FAILURE,
        ["repetition_or_looping", "planning_stall"],
    )
    assert [item.insight_id for item in result] == [
        "both-new",
        "both-old",
        "plan-3",
        "plan-2",
        "plan-1",
    ]

    fallback = PlaybookInsight(
        insight_id="grounding",
        failure_type=FailureType.REASONING_GROUNDING_FAILURE,
        retrieval_labels=["hallucination"],
        trigger_pattern="grounding failure",
        repair_principle="use observed evidence",
    )
    manager.add(fallback)
    assert manager.retrieve(
        FailureType.REASONING_GROUNDING_FAILURE, ["objective_drift"]
    ) == [fallback]


def test_task_reset_clears_trace_but_preserves_cross_task_playbook():
    manager_entry = PlaybookInsight(
        insight_id="seed",
        failure_type=FailureType.PROGRESS_FAILURE,
        retrieval_labels=["planning_stall"],
        trigger_pattern="planning without action",
        repair_principle="take one executable action",
    )
    sentry = Sentry(guard_judge_callback=lambda _: json.dumps({
        "intervention_needed": False,
        "failure_type": "no_failure",
        "retrieval_labels": [],
        "local_evidence": [],
        "diagnosis": "none",
    }))
    sentry.playbook.add(manager_entry)
    sentry.start_task(objective="task one", action_schema=SCHEMA)
    sentry.observe_step(step(0))
    sentry.start_task(objective="task two", action_schema=SCHEMA)
    assert sentry.window.steps == []
    assert sentry.active_rescue_event is None
    assert [entry.insight_id for entry in sentry.playbook.entries] == ["seed"]


def test_model_adapter_enforces_generation_budget_and_structured_output():
    calls = []

    def model(prompt: str, **kwargs):
        calls.append((prompt, kwargs))
        return {"content": "{}"}

    judge = model_to_text_judge(model)
    assert judge("prompt") == "{}"
    assert calls[0][1] == {
        "temperature": 0.0,
        "max_tokens": 4096,
        "response_format": {"type": "json_object"},
    }


def test_rolling_five_cycle_detection_and_ten_cycle_recovery_exclude_evaluator_fields():
    """The detector and recovery models receive different, bounded evidence windows."""
    prompts: dict[str, str] = {}
    outputs = {
        "detector": {
            "intervention_needed": True,
            "failure_type": "progress_failure",
            "retrieval_labels": ["repetition_or_looping"],
            "local_evidence": ["A search repeats without new evidence."],
            "diagnosis": "Repeated search requires a new grounded action.",
        },
        "verifier": {
            "status": "resolved",
            "justification": "New visible candidates were inspected.",
            "evidence": ["Candidate details supplied useful new information."],
        },
        "updater": {
            "failure_type": "progress_failure",
            "retrieval_labels": ["repetition_or_looping"],
            "subtype": "Repeated search",
            "trigger_pattern": "A search repeats without new information.",
            "repair_principle": "Inspect a visible candidate before repeating the search.",
        },
    }

    def judge(stage):
        def call(prompt):
            assert stage not in prompts
            prompts[stage] = prompt
            return json.dumps(outputs[stage])
        return call

    sentry = Sentry(
        guard_judge_callback=judge("detector"),
        verifier_judge_callback=judge("verifier"),
        updater_judge_callback=judge("updater"),
    )
    sentry.start_task(objective="Inspect a matching candidate", action_schema=SCHEMA)
    for index in range(8):
        observed = step(index, observation=f"observed candidate {index}", score=987654321.0)
        observed.metadata["evaluator_only_secret"] = "NEVER_EXPOSE_EVALUATOR_STATE"
        sentry.observe_step(observed)
    assert [item.step_id for item in sentry.window.steps] == list(range(3, 8))
    assert sentry.maybe_rescue() is not None
    detector_cycles = [line for line in prompts["detector"].splitlines() if line.startswith("Cycle ")]
    assert detector_cycles == [f"Cycle {index}:" for index in range(3, 8)]

    for index in range(8, 18):
        observed = step(index, action=f"click[{index}]", score=987654321.0)
        observed.metadata["evaluator_only_secret"] = "NEVER_EXPOSE_EVALUATOR_STATE"
        sentry.observe_step(observed)
    assert set(prompts) == {"detector", "verifier", "updater"}
    for stage in ("verifier", "updater"):
        recovery_cycles = [line for line in prompts[stage].splitlines() if line.startswith("Cycle ")]
        assert recovery_cycles == [f"Cycle {index}:" for index in range(8, 18)]
    for prompt in prompts.values():
        assert "987654321" not in prompt
        assert "NEVER_EXPOSE_EVALUATOR_STATE" not in prompt
        assert "evaluator_only_secret" not in prompt
        assert "benchmark_reward" not in prompt
    assert len(sentry.playbook.entries) == 1
