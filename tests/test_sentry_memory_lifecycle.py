"""Exercise learned-memory lifecycle without a demo environment or model API."""
from __future__ import annotations

import gc
import json
import tracemalloc

import pytest

from Sentry import FailureType, Sentry, SentryConfig, SentryRunner, agent_step_from_parts


NO_FAILURE = {
    'intervention_needed': False, 'failure_type': 'no_failure',
    'retrieval_labels': [], 'local_evidence': [], 'diagnosis': 'Visible progress.',
}
FAILURE = {
    'intervention_needed': True, 'failure_type': 'progress_failure',
    'retrieval_labels': ['repetition_or_looping'],
    'local_evidence': ['A search repeats without revealing another record.'],
    'diagnosis': 'Repeated search is not making progress.',
}
LESSON = {
    'failure_type': 'progress_failure', 'retrieval_labels': ['repetition_or_looping'],
    'subtype': 'repeated search',
    'trigger_pattern': 'A search returns the same visible records.',
    'repair_principle': 'Inspect an unchecked visible record before repeating the search.',
}


def cycle(step_id, *, action='search[records]', observation='The same records remain visible.'):
    return agent_step_from_parts(
        step_id=step_id, reasoning='', raw_action=action, observation=observation,
        parsed_ok=True, schema_valid=True, accepted_by_environment=True,
    )


@pytest.mark.parametrize('many_tasks', [False, True])
@pytest.mark.parametrize('logging_enabled', [False, True])
def test_default_runner_memory_does_not_grow_with_unconsumed_events(tmp_path, many_tasks, logging_enabled):
    path = tmp_path / 'events.jsonl'
    config = SentryConfig.from_dict({'logging': {'enabled': logging_enabled, 'log_path': str(path)}})
    no_failure = json.dumps(NO_FAILURE)
    runner = SentryRunner(lambda *a, **kw: no_failure, config)

    def batches(start, stop):
        for task in range(start, stop):
            if many_tasks or task == 0:
                runner.start_task(objective=f'Task {task}', action_schema='search[query]')
            for index in range(10):
                # Allocate distinct strings so retained history consumes real memory.
                runner.step(cycle(task * 10 + index, observation=f'{task}/{index}:' + 'x' * 20000),
                            terminal=many_tasks and index == 9)
            if many_tasks:
                runner.finalize()

    tracemalloc.start()
    try:
        batches(0, 10)
        gc.collect()
        baseline, _ = tracemalloc.get_traced_memory()
        batches(10, 50)
        gc.collect()
        retained, _ = tracemalloc.get_traced_memory()
        # The previous event queue retained >8 MiB here. Allow allocator noise
        # while requiring bounded memory both within a task and across tasks.
        assert retained - baseline < 1024 * 1024
    finally:
        runner.finalize()
        tracemalloc.stop()
    assert runner.guard.drain_runtime_events() == []
    if logging_enabled:
        events = [json.loads(line) for line in path.read_text().splitlines()]
        assert sum(e['type'] == 'agent_step' for e in events) == 500
        assert events[-1]['type'] == 'task_finalized'
    else:
        assert not path.exists()


def test_opt_in_capture_preserves_snapshots_across_tasks_until_drained():
    runner = SentryRunner(lambda *a, **kw: '{}',
                          SentryConfig.from_dict({'logging': {'enabled': False}}),
                          capture_runtime_events=True)
    for task in range(2):
        runner.start_task(objective=f'Task {task}', action_schema='search[query]')
        step = cycle(task, observation=f'Original observation {task}')
        runner.step(step, terminal=True)
        step.observation = 'Caller changed the observation'
        runner.finalize()
    events = runner.guard.drain_runtime_events()
    assert [e['type'] for e in events] == ['task_started', 'agent_step', 'task_finalized'] * 2
    assert [e['step']['observation'] for e in events if e['type'] == 'agent_step'] == [
        'Original observation 0', 'Original observation 1',
    ]
    assert runner.guard.drain_runtime_events() == []
    runner.start_task(objective='Next task', action_schema='search[query]')
    runner.finalize()
    assert len(events) == 6
    assert [e['type'] for e in runner.guard.drain_runtime_events()] == ['task_started', 'task_finalized']


@pytest.mark.parametrize('capture', [False, True])
def test_yaml_runner_forwards_event_capture_without_model_calls(tmp_path, monkeypatch, capture):
    path = tmp_path / 'config.yaml'
    path.write_text('api:\n  provider: openai\n  model: test-model\n')
    def unused_model(*args, **kwargs):
        pytest.fail('Task lifecycle must not invoke the model')
    monkeypatch.setattr('Sentry.integration.create_client', lambda config: unused_model)
    runner = SentryRunner.from_yaml(path, capture_runtime_events=capture)
    runner.start_task(objective='Task', action_schema='search[query]')
    runner.finalize()
    assert [e['type'] for e in runner.guard.drain_runtime_events()] == (
        ['task_started', 'task_finalized'] if capture else []
    )


@pytest.mark.parametrize('capture', ['false', 1, None])
def test_event_capture_rejects_non_boolean_flags(capture):
    with pytest.raises(ValueError, match='capture_runtime_events'):
        Sentry(capture_runtime_events=capture)


def test_verified_lesson_is_reused_only_when_a_later_task_needs_intervention(monkeypatch):
    # Model decisions are explicit fixtures; retrieval and learning use the real runtime.
    detect_failure = False
    model_calls = []

    def model(prompt, **kwargs):
        model_calls.append(prompt)
        if prompt.startswith('[SENTRY DETECTOR]'):
            return json.dumps(FAILURE if detect_failure else NO_FAILURE)
        if prompt.startswith('[SENTRY ESCAPE VERIFIER]'):
            return json.dumps({
                'status': 'resolved', 'justification': 'Repeated search stopped.',
                'evidence': ['Distinct visible records were inspected.'],
            })
        if prompt.startswith('[SENTRY PLAYBOOK UPDATE]'):
            return json.dumps(LESSON)
        assert '[SENTRY SOFT REPAIR]' in prompt
        return 'inspect[next-record]'

    runner = SentryRunner(model, capture_runtime_events=True)
    retrievals = []
    retrieve = runner.guard.playbook.retrieve

    def tracked_retrieve(failure_type, labels):
        entries = retrieve(failure_type, labels)
        retrievals.append((failure_type, list(labels), [entry.insight_id for entry in entries]))
        return entries

    monkeypatch.setattr(runner.guard.playbook, 'retrieve', tracked_retrieve)

    runner.start_task(objective='Healthy task before learning', action_schema='search[query] | inspect[id]')
    assert runner.step(cycle(0, observation='A new visible record was found.')) is None
    assert retrievals == [] and runner.guard.playbook.entries == []
    runner.finalize()

    detect_failure = True
    runner.start_task(objective='First task learns a recovery', action_schema='search[query] | inspect[id]')
    initial_repair = runner.step(cycle(0))
    assert initial_repair is not None and initial_repair.route == 'soft_repair'
    assert retrievals == [(FailureType.PROGRESS_FAILURE, ['repetition_or_looping'], [])]
    for step_id in range(1, 11):
        assert runner.step(cycle(step_id, action=f'inspect[record-{step_id}]',
                                 observation=f'Record {step_id} supplied new evidence.'),
                           terminal=step_id == 10) is None
    learned = runner.guard.playbook.entries[0]
    assert learned.repair_principle == LESSON['repair_principle']
    updates = [e for e in runner.guard.drain_runtime_events() if e['type'] == 'playbook_update']
    assert [e['entry']['insight_id'] for e in updates] == [learned.insight_id]
    runner.finalize()

    detect_failure = False
    runner.start_task(objective='Healthy task after learning', action_schema='search[query] | inspect[id]')
    assert runner.guard.window.steps == [] and runner.guard.active_rescue_event is None
    assert runner.guard.playbook.entries == [learned]
    assert runner.step(cycle(0, observation='Another new visible record was found.')) is None
    assert len(retrievals) == 1  # Existing memory does not cause speculative retrieval.
    healthy_prompt = model_calls[-1]
    assert learned.insight_id not in healthy_prompt
    assert learned.repair_principle not in healthy_prompt
    assert 'First task learns a recovery' not in healthy_prompt
    runner.finalize()

    detect_failure = True
    runner.start_task(objective='Later task reuses a recovery', action_schema='search[query] | inspect[id]')
    later_repair = runner.step(cycle(0))
    assert later_repair is not None and later_repair.route == 'soft_repair'
    assert retrievals[-1] == (FailureType.PROGRESS_FAILURE, ['repetition_or_looping'], [learned.insight_id])
    assert learned.insight_id in later_repair.playbook_context
    assert later_repair.retrieved_entry_ids == [learned.insight_id]
    assert learned.repair_principle in later_repair.prompt_text
    assert 'First task learns a recovery' not in later_repair.prompt_text
    # The actual continuation receives the selected learned guidance.
    assert runner.generate_repair(later_repair) == 'inspect[next-record]'
    assert learned.insight_id in model_calls[-1]
    assert 'Later task reuses a recovery' in model_calls[-1]
    assert runner.guard.playbook.entries == [learned]
    later_events = runner.guard.drain_runtime_events()
    injected = [event for event in later_events if event['type'] == 'rescue_injected']
    assert injected[-1]['retrieved_entry_ids'] == [learned.insight_id]
    runner.finalize()
