from __future__ import annotations

import json

import pytest

from Sentry import SentryRunner, agent_step_from_parts

DETECTION = {'intervention_needed': True, 'failure_type': 'progress_failure',
             'retrieval_labels': ['repetition_or_looping'], 'local_evidence': ['repeated action'],
             'diagnosis': 'No new information.'}
RESOLVED = {'status': 'resolved', 'justification': 'Loop stopped.', 'evidence': ['New grounded action.']}
UPDATE = {'failure_type': 'progress_failure', 'retrieval_labels': ['repetition_or_looping'],
          'subtype': 'repetition', 'trigger_pattern': 'Same request without new information.',
          'repair_principle': 'Inspect a different visible resource.'}


def cycle(step_id, *, valid=True):
    return agent_step_from_parts(step_id=step_id, reasoning='', raw_action='inspect[sample]',
        observation='Visible resource inspected.', parsed_ok=valid, schema_valid=valid,
        accepted_by_environment=valid)


def test_terminal_cycle_skips_new_diagnosis_and_still_verifies_pending_hard_repair():
    calls = []
    runner = SentryRunner(lambda *a, **kw: calls.append(a) or '{}')
    runner.start_task(objective='Inspect sample', action_schema='inspect[sample]')
    assert runner.step(cycle(0, valid=False)) is not None
    assert runner.step(cycle(1), terminal=True) is None
    assert calls == []
    assert runner.guard.last_escape_result.resolved
    assert runner.guard.playbook.entries == []
    runner.finalize()


def test_terminal_tenth_recovery_cycle_verifies_and_learns_without_new_diagnosis():
    phases = []
    def model(prompt, **kwargs):
        if prompt.startswith('[SENTRY DETECTOR]'):
            phases.append('detector')
            return json.dumps(DETECTION)
        if prompt.startswith('[SENTRY ESCAPE VERIFIER]'):
            phases.append('verifier')
            return json.dumps(RESOLVED)
        assert prompt.startswith('[SENTRY PLAYBOOK UPDATE]')
        phases.append('updater')
        return json.dumps(UPDATE)
    runner = SentryRunner(model)
    runner.start_task(objective='Inspect sample', action_schema='inspect[sample]')
    assert runner.step(cycle(0)) is not None
    for i in range(1, 10):
        assert runner.step(cycle(i)) is None
    assert phases == ['detector']
    assert runner.step(cycle(10), terminal=True) is None
    assert phases == ['detector', 'verifier', 'updater']
    assert len(runner.guard.playbook.entries) == 1
    runner.finalize()


def test_early_terminal_episode_is_not_verified_or_learned():
    calls = []
    def model(prompt, **kwargs):
        calls.append(prompt)
        assert prompt.startswith('[SENTRY DETECTOR]')
        return json.dumps(DETECTION)
    runner = SentryRunner(model, capture_runtime_events=True)
    runner.start_task(objective='Inspect sample', action_schema='inspect[sample]')
    runner.step(cycle(0))
    runner.step(cycle(1), terminal=True)
    runner.finalize()
    assert len(calls) == 1 and not runner.guard.playbook.entries
    incomplete = [e for e in runner.guard.drain_runtime_events() if e['type'] == 'verification_incomplete']
    assert incomplete[0]['observed_cycles'] == 1 and incomplete[0]['required_cycles'] == 10


@pytest.mark.parametrize('terminal', ['false', 1, None])
def test_invalid_terminal_flag_is_rejected_before_observation(terminal):
    runner = SentryRunner(lambda *a, **kw: '{}')
    runner.start_task(objective='Inspect sample', action_schema='inspect[sample]')
    with pytest.raises(ValueError, match='terminal'):
        runner.step(cycle(0), terminal=terminal)
    assert not runner.guard.window.steps


@pytest.mark.parametrize('generation_fails', [False, True])
def test_repair_generation_alone_never_observes_or_verifies_a_replacement(generation_fails):
    def model(prompt, **kwargs):
        if generation_fails:
            raise RuntimeError('transport failed')
        return 'inspect[sample]'

    runner = SentryRunner(model, capture_runtime_events=True)
    runner.start_task(objective='Inspect sample', action_schema='inspect[sample]')
    repair = runner.step(cycle(0, valid=False))
    assert repair is not None
    runner.guard.drain_runtime_events()
    if generation_fails:
        with pytest.raises(RuntimeError, match='transport failed'):
            runner.generate_repair(repair)
    else:
        assert runner.generate_repair(repair) == 'inspect[sample]'
    assert [step.step_id for step in runner.guard.window.steps] == [0]
    assert runner.guard.last_escape_result is None
    assert runner.guard.active_rescue_event.post_rescue_steps == []
    assert runner.guard.drain_runtime_events() == []
    runner.finalize()
    assert runner.guard.task is None
    incomplete = [e for e in runner.guard.drain_runtime_events() if e['type'] == 'verification_incomplete']
    assert incomplete[0]['observed_cycles'] == 0


def test_detector_failure_keeps_the_observed_cycle_until_host_finalization():
    calls = []
    def model(prompt, **kwargs):
        calls.append(prompt)
        raise RuntimeError('transport failed')

    runner = SentryRunner(model, capture_runtime_events=True)
    runner.start_task(objective='Inspect sample', action_schema='inspect[sample]')
    with pytest.raises(RuntimeError, match='transport failed'):
        runner.step(cycle(0))
    assert len(calls) == 1
    assert [step.step_id for step in runner.guard.window.steps] == [0]
    events = runner.guard.drain_runtime_events()
    assert [e['type'] for e in events] == ['task_started', 'agent_step']
    assert not runner.guard.playbook.entries
    runner.finalize()
    assert runner.guard.task is None and runner.guard.window.steps == []
