from __future__ import annotations

import json

import pytest

from Sentry import FailureType, PlaybookInsight, Sentry, SentryConfig, agent_step_from_parts
from Sentry.config import PlaybookConfig, TaskAgentConfig
from Sentry.detector import parse_detector_response
from Sentry.models import FailureDetection, LocalTraceWindow, RescueEvent, RescuePrompt
from Sentry.playbook import PlaybookManager
from Sentry.updater import parse_update_response
from Sentry.verifier import parse_verifier_response


DETECTION = {
    "intervention_needed": True,
    "failure_type": "progress_failure",
    "retrieval_labels": ["repetition_or_looping"],
    "local_evidence": ["The same search returned the same observation."],
    "diagnosis": "The agent repeats a search without new evidence.",
}
NO_FAILURE = {
    "intervention_needed": False,
    "failure_type": "no_failure",
    "retrieval_labels": [],
    "local_evidence": [],
    "diagnosis": "A grounded action advances the objective.",
}
RESOLVED = {
    "status": "resolved",
    "justification": "A visible candidate was inspected instead of repeating the search.",
    "evidence": ["The candidate's details supplied new task-relevant information."],
}
UPDATE = {
    "failure_type": "progress_failure",
    "retrieval_labels": ["repetition_or_looping"],
    "subtype": "Repeated search",
    "trigger_pattern": "The same search returns no new information.",
    "repair_principle": "Inspect a visible candidate before repeating a search.",
}


def cycle(index: int, *, valid: bool = True, accepted: bool | None = True):
    return agent_step_from_parts(
        step_id=index,
        reasoning="Use a visible candidate.",
        raw_action=f"click[{index}]",
        observation=f"visible candidate details {index}",
        schema_valid=valid,
        schema_error=None if valid else "The candidate is unavailable.",
        accepted_by_environment=accepted,
    )


def event() -> RescueEvent:
    failure = FailureType.PROGRESS_FAILURE
    return RescueEvent(
        "test-event", failure,
        FailureDetection(failure, DETECTION["local_evidence"], ["repetition_or_looping"]),
        RescuePrompt(failure, "Inspect a visible candidate.", DETECTION["local_evidence"]),
        LocalTraceWindow(),
    )


def start(**kwargs) -> Sentry:
    sentry = Sentry(guard_judge_callback=lambda _: json.dumps(DETECTION), **kwargs)
    sentry.start_task(objective="Find a matching item", action_schema="click[visible candidate]")
    sentry.observe_step(cycle(0))
    assert sentry.maybe_rescue() is not None
    return sentry


def entry(identifier: str, label: str = "planning_stall", failure=FailureType.PROGRESS_FAILURE):
    return PlaybookInsight(identifier, failure, [label], f"trigger {identifier}", f"repair {identifier}")


@pytest.mark.parametrize("patch", [
    {"intervention_needed": "true"},
    {"failure_type": "looping"},
    {"failure_type": "action_validity_failure"},
    {"failure_type": "no_failure"},
    {"retrieval_labels": ["repetition_or_looping", "unknown"]},
    {"retrieval_labels": ["looping"]},
    {"retrieval_labels": ["repetition_or_looping", "repetition_or_looping"]},
    {"retrieval_labels": ["hallucination"]},
    {"local_evidence": [17]},
    {"local_evidence": [{}]},
    {"local_evidence": []},
    {"diagnosis": {"text": "looping"}},
    {"diagnosis": " "},
    {"extra": True},
])
def test_detector_fails_closed_on_schema_errors(patch):
    result = parse_detector_response(json.dumps(DETECTION | patch))
    assert not result.should_rescue
    assert result.source == "invalid_model_output"


def test_detector_requires_complete_unambiguous_json():
    missing = dict(DETECTION)
    del missing["diagnosis"]
    valid_json = json.dumps(DETECTION)
    duplicate = valid_json[:-1] + ', "intervention_needed": false}'
    for raw in (json.dumps(missing), f"```json\n{valid_json}\n```", duplicate):
        assert parse_detector_response(raw).source == "invalid_model_output"
    assert not parse_detector_response(json.dumps(NO_FAILURE)).should_rescue
    assert parse_detector_response(json.dumps(DETECTION)).should_rescue


@pytest.mark.parametrize("patch", [
    {"status": ["resolved"]},
    {"status": "Resolved"},
    {"justification": True},
    {"evidence": [{}]},
    {"evidence": []},
    {"extra": "unsupported"},
])
def test_verifier_schema_errors_cannot_authorize_learning(patch):
    assert not parse_verifier_response(json.dumps(RESOLVED | patch), event()).resolved


@pytest.mark.parametrize("patch", [
    {"failure_type": "looping"},
    {"retrieval_labels": ["repetition_or_looping", "unknown"]},
    {"retrieval_labels": []},
    {"retrieval_labels": ["planning_stall"]},
    {"trigger_pattern": {"text": "repeat"}},
    {"repair_principle": True},
    {"subtype": ""},
    {"extra": 1},
])
def test_updater_rejects_malformed_or_relabeled_recoveries(patch):
    assert parse_update_response(json.dumps(UPDATE | patch), event()) is None


def test_updater_retains_subtype_in_serializable_record():
    learned = parse_update_response(json.dumps(UPDATE), event())
    assert learned is not None
    assert learned.subtype == "Repeated search"
    assert PlaybookInsight.from_json(learned.to_json()) == learned


def test_schema_valid_execution_rejection_is_not_automatically_hard_failure():
    calls = []
    sentry = Sentry(guard_judge_callback=lambda prompt: calls.append(prompt) or json.dumps(NO_FAILURE))
    sentry.start_task(objective="Find a matching item", action_schema="click[id]")
    sentry.observe_step(cycle(0, accepted=False))
    assert sentry.maybe_rescue() is None
    assert len(calls) == 1


def test_hard_prompt_contains_current_observation_even_for_first_cycle():
    sentry = Sentry()
    sentry.start_task(objective="Find a matching item", action_schema="click[id]")
    sentry.observe_step(cycle(0, valid=False, accepted=False))
    rescue = sentry.maybe_rescue()
    assert rescue is not None
    assert "visible candidate details 0" in rescue.prompt_text
    assert "Find a matching item" in rescue.prompt_text


def test_no_overlap_falls_back_to_five_recent_entries_of_the_same_type():
    manager = PlaybookManager()
    for index in range(7):
        manager.add(entry(str(index), "planning_stall"))
    manager.add(entry("grounding", "hallucination", FailureType.REASONING_GROUNDING_FAILURE))
    result = manager.retrieve(FailureType.PROGRESS_FAILURE, ["over_exploration"])
    assert [item.insight_id for item in result] == ["6", "5", "4", "3", "2"]


def test_matching_candidates_do_not_get_padded_with_nonmatching_entries():
    manager = PlaybookManager()
    manager.add(entry("match", "repetition_or_looping"))
    manager.add(entry("unrelated", "planning_stall"))
    result = manager.retrieve(FailureType.PROGRESS_FAILURE, ["repetition_or_looping"])
    assert [item.insight_id for item in result] == ["match"]


def test_diversity_does_not_reorder_when_all_candidates_fit():
    manager = PlaybookManager()
    manager.add(entry("older", "planning_stall"))
    for index in range(3):
        manager.add(entry(str(index), "repetition_or_looping"))
    result = manager.retrieve(FailureType.PROGRESS_FAILURE, ["repetition_or_looping", "planning_stall"])
    assert [item.insight_id for item in result] == ["2", "1", "0", "older"]


def test_repeated_polling_does_not_rediagnose_unchanged_trajectory():
    calls = []
    sentry = Sentry(guard_judge_callback=lambda prompt: calls.append(prompt) or json.dumps(NO_FAILURE))
    sentry.start_task(objective="Find a matching item", action_schema="click[id]")
    sentry.observe_step(cycle(0))
    assert sentry.maybe_rescue() is None
    assert sentry.maybe_rescue() is None
    assert len(calls) == 1


def test_hard_repair_remains_available_during_soft_recovery_horizon():
    verifications = []
    sentry = start(
        verifier_judge_callback=lambda prompt: verifications.append(prompt) or json.dumps(RESOLVED),
        updater_judge_callback=lambda _: json.dumps(UPDATE),
    )
    original = sentry.active_rescue_event
    sentry.observe_step(cycle(1, valid=False, accepted=False))
    hard = sentry.maybe_rescue()
    assert hard is not None and hard.route == "hard_repair"
    sentry.observe_step(cycle(2))
    assert sentry.active_rescue_event is original
    assert sentry.last_escape_result.resolved
    assert verifications == []
    for index in range(3, 11):
        sentry.observe_step(cycle(index))
    assert len(verifications) == 1
    assert len(original.post_rescue_steps) == 10
    assert sentry.active_rescue_event is None
    assert len(sentry.playbook.entries) == 1


def test_provider_failure_at_h_does_not_wedge_the_next_cycle():
    def unavailable(_):
        raise RuntimeError("provider unavailable")

    sentry = start(verifier_judge_callback=unavailable, capture_runtime_events=True)
    for index in range(1, 10):
        sentry.observe_step(cycle(index))
    with pytest.raises(RuntimeError, match="provider unavailable"):
        sentry.observe_step(cycle(10))
    assert sentry.active_rescue_event is None
    assert sentry.playbook.entries == []
    sentry.observe_step(cycle(11))
    assert any(item["type"] == "verification_error" for item in sentry.drain_runtime_events())


def test_unresolved_recoveries_never_call_updater():
    def forbidden(_):
        raise AssertionError("This recovery cannot update the playbook")

    result = RESOLVED | {"status": "unresolved"}
    sentry = start(
        verifier_judge_callback=lambda _: json.dumps(result),
        updater_judge_callback=forbidden,
    )
    for index in range(1, 11):
        sentry.observe_step(cycle(index))
    assert sentry.playbook.entries == []


def test_new_manager_starts_empty_and_explicit_restore_preserves_insertion_order(tmp_path):
    path = str(tmp_path / "playbook.json")
    manager = PlaybookManager(PlaybookConfig(persist_path=path))
    manager.add(entry("first"))
    manager.add(entry("second"))
    loaded = PlaybookManager(PlaybookConfig(persist_path=path))
    assert loaded.entries == []
    loaded.load()
    loaded.add(entry("third"))
    assert [item.insight_id for item in loaded.retrieve(FailureType.PROGRESS_FAILURE, ["planning_stall"])] == ["third", "second", "first"]
    assert [item.insertion_step for item in loaded.entries] == [1, 2, 3]


def test_configuration_rejects_malformed_sections_and_task_agent_deviation():
    assert SentryConfig.from_dict({"task_agent": {"temperature": 0.7, "max_tokens": 384}}).task_agent == TaskAgentConfig()
    with pytest.raises(ValueError, match="mapping"):
        SentryConfig.from_dict({"trace": "W=5"})
    with pytest.raises(ValueError, match="task agent"):
        SentryConfig(task_agent=TaskAgentConfig(temperature=0.0))


@pytest.mark.parametrize("contents", ["[]", "false", "0", "a string"])
def test_yaml_scalar_and_sequence_configs_are_rejected(tmp_path, contents):
    path = tmp_path / "sentry.yaml"
    path.write_text(contents)
    with pytest.raises(ValueError, match="mapping"):
        SentryConfig.from_yaml(path)


def test_finalize_resets_task_local_verification_result():
    sentry = Sentry()
    sentry.start_task(objective="Find item", action_schema="click[id]")
    sentry.observe_step(cycle(0, valid=False, accepted=False))
    assert sentry.maybe_rescue() is not None
    sentry.observe_step(cycle(1))
    assert sentry.last_escape_result.resolved
    sentry.finalize_task()
    assert sentry.last_escape_result is None
    assert sentry.active_rescue_event is None
    assert sentry.window.steps == []


def test_observations_are_snapshots_in_both_windows():
    sentry = start()
    observed = cycle(1)
    sentry.observe_step(observed)
    observed.observation = "mutated after observation"
    assert sentry.window.steps[-1].observation == "visible candidate details 1"
    assert sentry.active_rescue_event.post_rescue_steps[-1].observation == "visible candidate details 1"


def test_explicit_snapshot_loading_requires_an_existing_artifact(tmp_path):
    with pytest.raises(ValueError, match="persist_path"):
        PlaybookManager().load()
    with pytest.raises(FileNotFoundError):
        PlaybookManager(PlaybookConfig(persist_path=str(tmp_path / "absent.json"))).load()


@pytest.mark.parametrize("mutate", [
    lambda data: data.update(format="unknown"),
    lambda data: data.update(version=True),
    lambda data: data.update(insertion_step=42),
    lambda data: data["entries"][0].update(retrieval_labels=["planning_stall", "unknown"]),
    lambda data: data["entries"][0].update(trigger_pattern={"text": "stall"}),
    lambda data: data["entries"][0].update(insertion_step=-1),
    lambda data: data["entries"].append(data["entries"][0]),
])
def test_corrupt_persisted_memory_is_rejected_without_partial_reload(tmp_path, mutate):
    path = tmp_path / "playbook.json"
    manager = PlaybookManager(PlaybookConfig(persist_path=str(path)))
    manager.add(entry("existing"))
    artifact = json.loads(path.read_text())
    mutate(artifact)
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError):
        manager.load()
    assert [item.insight_id for item in manager.entries] == ["existing"]


def test_failed_atomic_save_leaves_previous_artifact_intact(tmp_path, monkeypatch):
    path = tmp_path / "playbook.json"
    manager = PlaybookManager(PlaybookConfig(persist_path=str(path)))
    manager.add(entry("existing"))
    before = path.read_text()

    def unavailable_replace(source, destination):
        raise OSError("filesystem unavailable")

    monkeypatch.setattr("Sentry.playbook.os.replace", unavailable_replace)
    with pytest.raises(OSError, match="filesystem unavailable"):
        manager.add(entry("new"))
    assert path.read_text() == before
    assert [item.insight_id for item in manager.entries] == ["existing"]
    assert list(tmp_path.glob(".*.tmp")) == []



def test_hard_verification_and_soft_h_boundary_can_share_one_cycle():
    sentry = start(
        capture_runtime_events=True,
        verifier_judge_callback=lambda _: json.dumps(RESOLVED),
        updater_judge_callback=lambda _: json.dumps(UPDATE),
    )
    soft_event_id = sentry.active_rescue_event.event_id
    for index in range(1, 9):
        sentry.observe_step(cycle(index))
    sentry.observe_step(cycle(9, valid=False, accepted=False))
    assert sentry.maybe_rescue().route == "hard_repair"
    hard_event_id = sentry.active_rescue_event.event_id
    sentry.observe_step(cycle(10))
    results = [item for item in sentry.drain_runtime_events() if item["type"] == "escape_result"]
    assert [item["event_id"] for item in results] == [hard_event_id, soft_event_id]
    assert all(item["resolved"] for item in results)
    assert sentry.active_rescue_event is None
    assert len(sentry.playbook.entries) == 1


def test_updater_provider_failure_does_not_reopen_verified_episode():
    def unavailable(_):
        raise RuntimeError("updater unavailable")

    sentry = start(
        capture_runtime_events=True,
        verifier_judge_callback=lambda _: json.dumps(RESOLVED),
        updater_judge_callback=unavailable,
    )
    for index in range(1, 10):
        sentry.observe_step(cycle(index))
    with pytest.raises(RuntimeError, match="updater unavailable"):
        sentry.observe_step(cycle(10))
    assert sentry.last_escape_result.resolved
    assert sentry.active_rescue_event is None
    assert sentry.playbook.entries == []
    sentry.observe_step(cycle(11))
    assert any(item.get("reason") == "updater_error" for item in sentry.drain_runtime_events())


def test_custom_soft_detector_cannot_bypass_schema_first_routing():
    class SoftDetector:
        def detect(self, window, task):
            raise AssertionError("Invalid actions must not reach the soft detector")

    sentry = Sentry(detector=SoftDetector())
    sentry.start_task(objective="Find item", action_schema="click[id]")
    sentry.observe_step(cycle(0, valid=False, accepted=False))
    prompt = sentry.maybe_rescue()
    assert prompt is not None and prompt.route == "hard_repair"


@pytest.mark.parametrize("patch", [
    {"parsed_ok": "false"},
    {"parsed_ok": 1},
    {"schema_valid": "true"},
    {"schema_valid": 0},
    {"accepted_by_environment": "true"},
    {"accepted_by_environment": 1},
    {"step_id": True},
    {"step_id": -1},
    {"raw_action": False},
    {"reasoning": []},
    {"observation": {}},
])
def test_adapter_inputs_cannot_bypass_validation_through_python_truthiness(patch):
    arguments = {
        "step_id": 0,
        "reasoning": "Inspect the candidate",
        "raw_action": "click[1]",
        "observation": "Candidate details",
    }
    with pytest.raises(ValueError):
        agent_step_from_parts(**(arguments | patch))


def test_mutated_invalid_cycle_is_rejected_before_observation():
    sentry = Sentry()
    sentry.start_task(objective="Find item", action_schema="click[id]")
    observed = cycle(0)
    observed.action.schema_valid = "false"
    with pytest.raises(ValueError, match="booleans"):
        sentry.observe_step(observed)
    assert sentry.window.steps == []


def test_playbook_input_and_output_mutation_cannot_change_validated_memory():
    manager = PlaybookManager()
    submitted = entry("verified")
    inserted = manager.add(submitted)
    submitted.trigger_pattern = "changed original input"
    inserted.retrieval_labels[:] = ["hallucination"]
    retrieved = manager.retrieve(FailureType.PROGRESS_FAILURE, ["planning_stall"])
    retrieved[0].repair_principle = "changed retrieval output"
    public = manager.entries
    public[0].insertion_step = 999
    public.clear()
    stored = manager.entries[0]
    assert stored.trigger_pattern == "trigger verified"
    assert stored.repair_principle == "repair verified"
    assert stored.retrieval_labels == ["planning_stall"]
    assert stored.insertion_step == 1


def test_failed_save_rolls_back_memory_counter_and_submitted_record(tmp_path, monkeypatch):
    path = tmp_path / "playbook.json"
    manager = PlaybookManager(PlaybookConfig(persist_path=str(path)))
    manager.add(entry("existing"))
    submitted = entry("new")
    original_replace = __import__("os").replace

    def unavailable_replace(source, destination):
        raise OSError("filesystem unavailable")

    monkeypatch.setattr("Sentry.playbook.os.replace", unavailable_replace)
    with pytest.raises(OSError):
        manager.add(submitted)
    assert submitted.insertion_step == 0
    assert [item.insight_id for item in manager.entries] == ["existing"]
    with pytest.raises(OSError):
        manager.clear()
    assert [item.insight_id for item in manager.entries] == ["existing"]
    monkeypatch.setattr("Sentry.playbook.os.replace", original_replace)
    manager.add(submitted)
    assert submitted.insertion_step == 2
    loaded = PlaybookManager(PlaybookConfig(persist_path=str(path)))
    loaded.load()
    assert loaded.entries == manager.entries


def test_updater_preserves_assigned_label_order_for_primary_label():
    recovery = event()
    recovery.detection.retrieval_labels = ["repetition_or_looping", "planning_stall"]
    output = UPDATE | {"retrieval_labels": ["planning_stall", "repetition_or_looping"]}
    learned = parse_update_response(json.dumps(output), recovery)
    assert learned is not None
    assert learned.retrieval_labels == recovery.detection.retrieval_labels
    assert learned.primary_label == "repetition_or_looping"
