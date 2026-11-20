"""Offline AppWorld host/CLI checks; no official world or model requests."""
from __future__ import annotations

import json
import io
from pathlib import Path

import pytest

from Sentry import SentryConfig, TaskContext, agent_step_from_parts
from Sentry.clients import (
    AnthropicMessagesClient, GeminiClient, ModelOutputTruncatedError, OpenAIResponsesClient,
)
from benchmarks.appworld import run

NO_FAILURE = json.dumps({'intervention_needed': False, 'failure_type': 'no_failure',
                         'retrieval_labels': [], 'local_evidence': [], 'diagnosis': 'Visible progress.'})


def config(**kwargs):
    return SentryConfig.from_dict({'api': {'provider': 'openai_compatible', 'model': 'test-model',
                                           'base_url': 'http://localhost:9999/v1'}, **kwargs})


class Client:
    model = 'test-model'
    max_retries = 0

    def __init__(self, actions=None, detector=NO_FAILURE):
        self.actions = iter(actions or ["print('visible')", 'finish'])
        self.detector = detector
        self.calls = []
        self.last_usage = {'cost': 999}
        self.last_response_metadata = {'id': 'stale'}

    def __call__(self, prompt, **kwargs):
        assert self.last_usage == self.last_response_metadata == {}
        self.calls.append((prompt, kwargs))
        value = self.detector if prompt.startswith('[SENTRY DETECTOR]') else next(self.actions)
        if isinstance(value, Exception):
            raise value
        self.last_usage = {'cost': 0.001, 'prompt_tokens': 10, 'completion_tokens': 2}
        self.last_response_metadata = {'model': self.model, 'id': f'response-{len(self.calls)}'}
        return value


class Environment:
    version = 'offline-test'
    task_context = TaskContext(objective='Inspect the visible record, then finish.', action_schema='Python code')
    initial_observation = 'A visible record is available.'

    def __init__(self, task_id, **kwargs):
        self.task_id, self.kwargs = task_id, kwargs
        self.executed = []
        self.closed = self.evaluated = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def execute(self, raw_action, *, step_id):
        self.executed.append(raw_action)
        return agent_step_from_parts(step_id=step_id, reasoning='', raw_action=raw_action,
            observation='Execution ended.' if raw_action == 'finish' else 'Record has public details.',
            parsed_ok=True, schema_valid=True, accepted_by_environment=True,
            metadata={'completion_requested': raw_action == 'finish'})

    def task_completed(self):
        return self.executed and self.executed[-1] == 'finish'

    def evaluate(self):
        self.evaluated = True
        return {'task_success': bool(self.task_completed()), 'score_sentinel': 78123}


@pytest.fixture
def execute(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'load_task_ids', lambda dataset, *, root: ['dev-task'])
    environments = []

    def invoke(client=None, environment_class=Environment, sentry_config=None, **kwargs):
        def factory(*args, **settings):
            env = environment_class(*args, **settings)
            environments.append(env)
            return env
        result = run.run_appworld(sentry_config or config(), root=tmp_path, output_dir=tmp_path / 'runs',
            client=client or Client(), environment_factory=factory, progress=lambda _: None, **kwargs)
        events = [json.loads(line) for line in (Path(result['output_dir']) / 'events.jsonl').read_text().splitlines()]
        return result, environments[-1], events
    return invoke


def test_complete_host_preserves_history_decoding_and_evaluates_after_finalization(execute, tmp_path):
    class CheckedEnvironment(Environment):
        def evaluate(self):
            events = next((tmp_path / 'runs').glob('*/events.jsonl')).read_text().splitlines()
            assert json.loads(events[-1])['type'] == 'task_finalized'
            return super().evaluate()

    client = Client()
    result, env, events = execute(client, environment_class=CheckedEnvironment)
    assert result['error'] is None and result['stop_reason'] == 'task_completed'
    assert result['metrics']['task_success'] is True
    assert env.closed and env.evaluated and len(env.executed) == 2
    assert result['observed_cycles'] == 2 and result['model_calls'] == 4
    assert [call['phase'] for call in result['calls']] == ['task_agent', 'detector', 'task_agent', 'detector']
    assert result['known_cost_usd'] == pytest.approx(0.004)
    assert 'print(\'visible\')' in client.calls[-1][0]
    assert 'Record has public details.' in client.calls[-1][0]
    assert all('score_sentinel' not in prompt and '78123' not in prompt for prompt, _ in client.calls)
    assert client.calls[0][1] == {'temperature': 0.7, 'max_tokens': 384}
    assert client.calls[1][1] == {'temperature': 0.0, 'max_tokens': 4096,
                                 'response_format': {'type': 'json_object'}}
    assert '384 tokens per action' in client.calls[0][0]
    assert len([e for e in events if e['type'] == 'agent_step']) == 2
    assert json.loads((Path(result['output_dir']) / 'playbook.json').read_text())['entries'] == []
    assert result['effective_config']['playbook']['persist_path'].startswith(result['output_dir'])


def test_truncated_task_action_is_unexecuted_and_next_action_verifies_hard_repair(execute):
    partial = "print('syntactically valid but incomplete generation')"
    client = Client([ModelOutputTruncatedError(partial, finish_reason='length'), 'finish'])
    result, env, events = execute(client)
    assert env.executed == ['finish']
    assert result['model_calls'] == 3 and result['observed_cycles'] == 2
    assert result['interventions'] == result['verified_recoveries'] == 1
    assert result['learned_entries'] == 0 and result['error'] is None
    first = next(e['step'] for e in events if e['type'] == 'agent_step')
    assert first['action']['raw'] == partial
    assert first['action']['parsed_ok'] is False and first['action']['accepted_by_environment'] is None
    assert '[SENTRY HARD REPAIR]' in client.calls[-2][0]


def test_repeated_hard_repairs_keep_only_current_guidance_in_agent_context(execute):
    client = Client([
        ModelOutputTruncatedError('first_partial()', finish_reason='length'),
        ModelOutputTruncatedError('second_partial()', finish_reason='length'),
        'finish',
    ])
    result, env, _ = execute(client)
    assert result['error'] is None and env.executed == ['finish']
    prompts = [prompt for prompt, _ in client.calls]
    assert '[SENTRY HARD REPAIR]' not in prompts[0]
    assert prompts[1].count('[SENTRY HARD REPAIR]') == 1
    assert prompts[2].count('[SENTRY HARD REPAIR]') == 1
    assert 'first_partial()' in prompts[2] and 'second_partial()' in prompts[2]
    assert '384-token limit' in prompts[2]


@pytest.mark.parametrize('provider', ['openai_responses', 'anthropic', 'gemini'])
def test_native_provider_truncation_reaches_appworld_hard_repair(execute, monkeypatch, provider):
    partial = "print('complete syntax but truncated generation')"
    if provider == 'openai_responses':
        client = OpenAIResponsesClient('https://example.invalid/v1', 'test-model')
        def response(text, truncated):
            status = 'incomplete' if truncated else 'completed'
            return {'status': status,
                    'incomplete_details': {'reason': 'max_output_tokens'} if truncated else None,
                    'output': [{'type': 'message', 'role': 'assistant', 'status': status,
                                'content': [{'type': 'output_text', 'text': text}]}]}
    elif provider == 'anthropic':
        client = AnthropicMessagesClient('https://example.invalid/v1', 'test-model', json_mode='prompt')
        def response(text, truncated):
            return {'stop_reason': 'max_tokens' if truncated else 'end_turn',
                    'content': [{'type': 'text', 'text': text}]}
    else:
        client = GeminiClient('https://example.invalid/v1', 'test-model')
        def response(text, truncated):
            return {'candidates': [{'finishReason': 'MAX_TOKENS' if truncated else 'STOP',
                                    'content': {'parts': [{'text': text}]}}]}
    replies = iter([response(partial, True), response('finish', False), response(NO_FAILURE, False)])
    monkeypatch.setattr('Sentry.clients.urlopen',
                        lambda *a, **kw: io.BytesIO(json.dumps(next(replies)).encode()))
    native_config = SentryConfig.from_dict({'api': {
        'provider': provider, 'model': 'test-model',
        'json_mode': 'prompt' if provider == 'anthropic' else 'native',
    }})
    result, env, _ = execute(client, sentry_config=native_config)
    assert env.executed == ['finish'] and env.evaluated
    assert result['error'] is None and result['stop_reason'] == 'task_completed'
    assert result['interventions'] == result['verified_recoveries'] == 1
    assert result['learned_entries'] == 0


def test_auxiliary_truncation_aborts_without_extra_cycles_or_evaluation(execute):
    client = Client(detector=ModelOutputTruncatedError('{}', finish_reason='length'))
    result, env, events = execute(client)
    assert result['error'] == {'type': 'ModelOutputTruncatedError'}
    assert result['metrics'] is None and not env.evaluated and env.closed
    assert len(env.executed) == result['observed_cycles'] == 1
    assert result['calls'][-1]['error_type'] == 'ModelOutputTruncatedError'
    assert events[-1]['type'] == 'task_finalized'


def test_adapter_exception_is_never_recast_as_an_unexecuted_generation(execute):
    class UncertainExecution(Environment):
        def execute(self, raw_action, *, step_id):
            self.executed.append(raw_action)
            raise ModelOutputTruncatedError('adapter exception', finish_reason='length')

    result, env, events = execute(environment_class=UncertainExecution)
    assert len(env.executed) == 1 and result['observed_cycles'] == 0
    assert result['metrics'] is None and not env.evaluated
    assert not any(e['type'] == 'agent_step' for e in events)


@pytest.mark.parametrize('limit,reason', [({'max_model_calls': 1}, 'model_call_limit'),
                                         ({'max_cost_usd': 0.0001}, 'cost_threshold')])
def test_request_and_cost_limits_stop_before_auxiliary_request_but_score_partial_task(execute, limit, reason):
    client = Client()
    result, env, events = execute(client, **limit)
    assert result['stop_reason'] == reason and result['error']['type'] == 'RunLimitError'
    assert len(client.calls) == result['model_calls'] == 1
    assert len(env.executed) == 1 and env.evaluated and env.closed
    assert result['metrics']['task_success'] is False
    assert events[-1]['type'] == 'task_finalized'


def test_last_budgeted_action_is_terminal_without_new_detector(execute):
    result, env, events = execute(max_steps=1)
    assert result['stop_reason'] == 'max_steps' and result['error'] is None
    assert result['model_calls'] == 1 and env.evaluated
    assert result['metrics']['task_success'] is False
    assert not any(e['type'] == 'sentry_diagnosis' for e in events)


class StickyCompletionEnvironment(Environment):
    def task_completed(self):
        # Like official AppWorld, later actions do not reset the completion flag.
        return 'finish' in self.executed


PREMATURE_COMPLETION = json.dumps({
    'intervention_needed': True, 'failure_type': 'progress_failure',
    'retrieval_labels': ['termination_miscalibration'],
    'local_evidence': ['The agent finished without inspecting the visible record.'],
    'diagnosis': 'Inspect the requested record before completing the task.',
})


def test_premature_completion_can_be_repaired_in_same_world_without_rechecking_during_h(execute):
    client = Client(['finish', 'inspect', 'finish'], detector=PREMATURE_COMPLETION)
    result, env, events = execute(client, environment_class=StickyCompletionEnvironment)
    assert result['error'] is None and result['stop_reason'] == 'task_completed'
    assert env.executed == ['finish', 'inspect', 'finish'] and env.evaluated
    assert result['observed_cycles'] == 3 and result['interventions'] == 1
    assert [call['phase'] for call in result['calls']] == ['task_agent', 'detector', 'task_agent', 'task_agent']
    assert '[SENTRY SOFT REPAIR]' in client.calls[2][0]
    assert 'call apis.supervisor.complete_task(...) again' in client.calls[3][0]
    assert 'No official evaluation has been performed.' in client.calls[2][0]
    assert all('score_sentinel' not in prompt and '78123' not in prompt for prompt, _ in client.calls)
    assert not any(e['type'] == 'escape_result' for e in events)
    assert result['learned_entries'] == 0
    assert [e['type'] for e in events[-2:]] == ['verification_incomplete', 'task_finalized']


def test_sticky_completion_does_not_end_recovery_without_a_fresh_completion_call(execute):
    client = Client(['finish', 'inspect', 'inspect'], detector=PREMATURE_COMPLETION)
    result, env, _ = execute(client, environment_class=StickyCompletionEnvironment, max_steps=3)
    assert env.task_completed()  # Official declaration remains set, despite host continuation.
    assert result['error'] is None and result['stop_reason'] == 'max_steps'
    assert result['observed_cycles'] == 3 and env.evaluated and env.closed


def test_completion_check_obeys_model_budget_and_never_reads_evaluation_early(execute):
    result, env, events = execute(Client(['finish']), max_model_calls=1)
    assert result['stop_reason'] == 'model_call_limit'
    assert result['error'] == {'type': 'RunLimitError'}
    assert result['model_calls'] == result['observed_cycles'] == 1
    assert env.evaluated and events[-1]['type'] == 'task_finalized'


@pytest.mark.parametrize('accepted', [False, None])
def test_failed_completion_action_does_not_stop_on_an_existing_complete_flag(execute, accepted):
    class FailedCompletionEnvironment(StickyCompletionEnvironment):
        def task_completed(self):
            return True
        def execute(self, raw_action, *, step_id):
            step = super().execute(raw_action, step_id=step_id)
            if step_id == 0:
                step.action.accepted_by_environment = accepted
            return step
    result, env, _ = execute(Client(['finish', 'finish']), environment_class=FailedCompletionEnvironment)
    assert result['error'] is None and result['stop_reason'] == 'task_completed'
    assert env.executed == ['finish', 'finish'] and result['observed_cycles'] == 2


def test_failed_request_resets_usage_and_redacts_provider_error(execute):
    private = 'SECRET_PROVIDER_BODY_DO_NOT_LOG'
    result, env, events = execute(Client(detector=RuntimeError(private)))
    assert result['error'] == {'type': 'RuntimeError'} and result['metrics'] is None
    assert result['known_cost_usd'] == 0.001 and result['missing_cost_calls'] == 1
    assert result['calls'][-1]['usage'] == result['calls'][-1]['response_metadata'] == {}
    assert private not in json.dumps(result) + json.dumps(events)


def test_http_status_is_retained_without_copying_error_body(execute):
    result, _, _ = execute(Client(detector=RuntimeError('Model endpoint returned HTTP 429')))
    assert result['error'] == {'type': 'RuntimeError', 'http_status': 429}


def test_prompt_limit_stops_before_inference_and_still_finalizes(execute):
    class LargeContext(Environment):
        task_context = TaskContext(objective='x' * 200_001, action_schema='Python code')
    result, env, events = execute(environment_class=LargeContext)
    assert result['stop_reason'] == 'prompt_limit' and result['model_calls'] == 0
    assert not env.executed and env.evaluated
    assert events[-1]['type'] == 'task_finalized'


def test_observation_truncation_is_explicit_and_applies_to_initial_and_subsequent_text(execute):
    class LargeObservation(Environment):
        initial_observation = 'a' * 30_000
        def execute(self, raw_action, *, step_id):
            return run.replace(super().execute(raw_action, step_id=step_id), observation='b' * 30_000)
    client = Client()
    result, _, events = execute(client, environment_class=LargeObservation)
    assert 'Observation truncated by host' in client.calls[0][0]
    for event in events:
        if event['type'] == 'agent_step':
            assert len(event['step']['observation']) <= run.OBSERVATION_LIMIT
            assert 'Observation truncated by host' in event['step']['observation']


@pytest.mark.parametrize('setting', [{'max_steps': 0}, {'max_model_calls': True},
                                     {'max_cost_usd': float('nan')}, {'max_cost_usd': 10**1000}])
def test_invalid_limits_fail_before_client_invocation(tmp_path, setting):
    client = Client()
    with pytest.raises(ValueError):
        run.run_appworld(config(), root=tmp_path, client=client, **setting)
    assert client.calls == []


@pytest.mark.parametrize('section,value', [('playbook', {'persist_path': 'external.json'}),
                                          ('logging', {'log_path': 'external.json'})])
def test_external_artifact_paths_are_rejected_with_output_directory_guidance(tmp_path, section, value):
    with pytest.raises(ValueError, match='--output-dir'):
        run.run_appworld(config(**{section: value}), root=tmp_path, client=Client())


def test_client_identity_and_retries_cannot_silently_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'load_task_ids', lambda *a, **kw: ['dev-task'])
    client = Client()
    client.model = 'wrong-model'
    with pytest.raises(ValueError, match='model'):
        run.run_appworld(config(), root=tmp_path, client=client)
    client.model = 'test-model'
    client.max_retries = 1
    with pytest.raises(ValueError, match='retries'):
        run.run_appworld(config(), root=tmp_path, client=client)
    assert client.calls == []


def test_credential_loader_selects_only_requested_literal_and_environment_wins(tmp_path, monkeypatch):
    name = 'SENTRY_RUN_TEST_CREDENTIAL'
    monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv('IGNORED_TEST_CREDENTIAL', raising=False)
    path = tmp_path / '.env'
    path.write_text(f'IGNORED_TEST_CREDENTIAL=ignored\nexport {name}="literal-$(no-execution)"\n')
    run.load_credential_file(path, name)
    assert run.os.environ[name] == 'literal-$(no-execution)'
    assert 'IGNORED_TEST_CREDENTIAL' not in run.os.environ
    run.load_credential_file(tmp_path / 'absent.env', name)
    monkeypatch.delenv(name)
    path.write_text(f'{name}=first\n{name}=second\n')
    with pytest.raises(ValueError, match='exactly once'):
        run.load_credential_file(path, name)


def test_list_tasks_requires_no_config_or_credentials(monkeypatch, capsys):
    monkeypatch.setattr(run, 'load_task_ids', lambda dataset, *, root: ['dev-a', 'dev-b'])
    monkeypatch.setattr(run, 'create_client', lambda *a: pytest.fail('Listing must not construct a model'))
    assert run.main(['--list-tasks']) == 0
    assert capsys.readouterr().out == 'dev-a\ndev-b\n'


def test_cli_preflight_error_is_actionable_without_copying_private_message(monkeypatch, capsys):
    monkeypatch.setattr(run, 'load_task_ids', lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('private detail')))
    assert run.main(['--list-tasks']) == 1
    output = capsys.readouterr().out
    assert 'appworld install' in output and 'appworld download data' in output
    assert 'private detail' not in output


def test_real_os_clocks_survive_world_time_freezing(execute, monkeypatch):
    import time
    from datetime import datetime as RealDatetime

    real_wall = 1_790_500_000.0
    ticks = iter([40.0, 42.5])
    def clock(clock_id):
        return real_wall if clock_id == time.CLOCK_REALTIME else next(ticks)
    monkeypatch.setattr(run.time, 'clock_gettime', clock, raising=False)
    monkeypatch.setattr(run.time, 'CLOCK_REALTIME', 0, raising=False)
    monkeypatch.setattr(run.time, 'CLOCK_MONOTONIC', 1, raising=False)
    monkeypatch.setattr(run.time, 'monotonic', lambda: 7)

    class FrozenDatetime(RealDatetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2001, 1, 1, tzinfo=tz)

    class FrozenWorld(Environment):
        def __enter__(self):
            monkeypatch.setattr(run, 'datetime', FrozenDatetime)
            return self

    result, _, _ = execute(environment_class=FrozenWorld, max_steps=1)
    assert not result['started_at'].startswith('2001-')  # Captured before entering the world.
    assert result['calls'][0]['elapsed_seconds'] == 2.5
    timestamp = RealDatetime.fromisoformat(result['calls'][0]['started_at'])
    assert timestamp.timestamp() == real_wall


def test_usage_and_metadata_keep_only_safe_finite_allowlisted_fields(execute):
    secret = 'PRIVATE_RESPONSE_BODY'
    class NoisyClient(Client):
        def __call__(self, prompt, **kwargs):
            response = super().__call__(prompt, **kwargs)
            self.last_usage = {'cost': 0.003, 'prompt_tokens': 11, 'completion_tokens': float('nan'),
                'total_tokens': float('inf'), 'input_tokens': -1, 'output_tokens': True,
                'cache_read_input_tokens': 10**1000, 'authorization': secret,
                'provider_payload': {'raw_response': secret}, 'unrecognized_numeric': 12}
            self.last_response_metadata = {'id': 'response-1', 'model': 'test-model',
                'provider': 'Test provider', 'finish_reason': 'stop', 'authorization': secret,
                'response': {'text': secret}, 'api_key': secret}
            return response
    result, _, _ = execute(NoisyClient(), max_steps=1)
    call = result['calls'][0]
    assert call['usage'] == {'cost': 0.003, 'prompt_tokens': 11}
    assert call['response_metadata'] == {'id': 'response-1', 'model': 'test-model',
                                         'provider': 'Test provider', 'finish_reason': 'stop'}
    assert secret not in json.dumps(result)
    assert result['known_cost_usd'] == 0.003


def test_failed_request_metadata_is_filtered_without_leaking_exception_text(execute):
    secret = 'PRIVATE_EXCEPTION_PAYLOAD'
    class FailedClient(Client):
        def __call__(self, prompt, **kwargs):
            self.last_usage = {'cost': 0.002, 'error': secret}
            self.last_response_metadata = {'id': 'failed-response', 'body': secret}
            raise RuntimeError(secret)
    result, _, events = execute(FailedClient())
    assert result['error'] == {'type': 'RuntimeError'}
    assert result['calls'][0]['usage'] == {'cost': 0.002}
    assert result['calls'][0]['response_metadata'] == {'id': 'failed-response'}
    assert secret not in json.dumps(result) + json.dumps(events)


def test_keyboard_interrupt_is_accounted_and_finalizes_without_scoring(execute):
    class InterruptedClient(Client):
        def __call__(self, prompt, **kwargs):
            raise KeyboardInterrupt('PRIVATE_INTERRUPT_DETAIL')
    result, env, events = execute(InterruptedClient())
    assert result['error'] == {'type': 'KeyboardInterrupt'}
    assert result['calls'][0]['error_type'] == 'KeyboardInterrupt'
    assert result['event_time_basis'] == 'appworld_simulated_time'
    assert result['metrics'] is None and not env.evaluated and env.closed
    assert events[-1]['type'] == 'task_finalized'
    assert 'PRIVATE_INTERRUPT_DETAIL' not in json.dumps(result) + json.dumps(events)


def test_limit_error_only_exports_recognized_host_codes():
    assert run._stop_reason(run.RunLimitError('PRIVATE_LIMIT_DETAIL')) == 'error'


def _run_stream(tmp_path, monkeypatch, client, environment_class=Environment, **limits):
    monkeypatch.setattr(run, 'load_task_ids', lambda dataset, *, root: ['dev-a', 'dev-b'])
    environments = []
    def factory(*args, **kwargs):
        environment = environment_class(*args, **kwargs)
        environments.append(environment)
        return environment
    result = run.run_appworld(config(), root=tmp_path, task_ids=['dev-a', 'dev-b'],
        output_dir=tmp_path / 'stream', client=client, environment_factory=factory,
        progress=lambda _: None, **limits)
    events = [json.loads(line) for line in (Path(result['output_dir']) / 'events.jsonl').read_text().splitlines()]
    assert json.loads((Path(result['output_dir']) / 'run.json').read_text()) == result
    return result, environments, events


def test_online_stream_learns_at_h_then_retrieves_in_new_task_without_old_context(tmp_path, monkeypatch):
    failure = json.dumps({'intervention_needed': True, 'failure_type': 'progress_failure',
        'retrieval_labels': ['repetition_or_looping'], 'local_evidence': ['Repeated inspection.'],
        'diagnosis': 'Inspect a different grounded record.'})
    principle = 'Inspect observed records instead of repeating the same search.'
    class LearningClient(Client):
        def __call__(self, prompt, **kwargs):
            if prompt.startswith('[SENTRY ESCAPE VERIFIER]'):
                response = json.dumps({'status': 'resolved', 'justification': 'Useful observations followed.',
                                       'evidence': ['Inspected the records and finished.']})
            elif prompt.startswith('[SENTRY PLAYBOOK UPDATE]'):
                response = json.dumps({'failure_type': 'progress_failure',
                    'retrieval_labels': ['repetition_or_looping'], 'subtype': 'Repetition',
                    'trigger_pattern': 'Repeated inspection of unchanged information.',
                    'repair_principle': principle})
            elif (prompt.startswith('[SENTRY DETECTOR]')
                  and prompt.rsplit('Action: ', 1)[-1].startswith('finish\n')):
                response = NO_FAILURE
            else:
                return super().__call__(prompt, **kwargs)
            assert self.last_usage == self.last_response_metadata == {}
            self.calls.append((prompt, kwargs))
            self.last_usage = {'cost': 0.001}
            return response

    class SequentialEnvironment(Environment):
        def __init__(self, task_id, **kwargs):
            super().__init__(task_id, **kwargs)
            self.task_context = TaskContext(objective=f'Public objective for {task_id}.', action_schema='Python code')
            self.initial_observation = f'Visible initial state for {task_id}.'
        def __enter__(self):
            if self.task_id == 'dev-b':
                checkpoint = json.loads(next((tmp_path / 'stream').glob('*/run.json')).read_text())
                assert checkpoint['tasks'][0]['status'] == 'finished'
                assert checkpoint['tasks'][0]['end_entries'] == 1
            return super().__enter__()
        def evaluate(self):
            events = next((tmp_path / 'stream').glob('*/events.jsonl')).read_text().splitlines()
            assert json.loads(events[-1])['type'] == 'task_finalized'
            assert json.loads(events[-1])['task_id'] == self.task_id
            return super().evaluate()

    client = LearningClient(['inspect'] * 10 + ['finish', 'inspect', 'finish'], detector=failure)
    result, environments, events = _run_stream(tmp_path, monkeypatch, client,
        environment_class=SequentialEnvironment, max_steps=12)
    first, second = result['tasks']
    assert result['error'] is None and result['stop_reason'] == 'task_sequence_finished'
    assert result['task_id'] is None and result['metrics'] is None
    assert first['observed_cycles'] == 11 and second['observed_cycles'] == 2
    assert first['learned_entries'] == 1 and second['learned_entries'] == 0
    assert first['start_entries'] == 0 and first['end_entries'] == second['start_entries'] == 1
    assert first['end_entry_ids'] == second['start_entry_ids'] == second['end_entry_ids']
    assert first['end_model_calls'] == second['start_model_calls']
    assert result['model_calls'] == sum(task['model_calls'] for task in result['tasks']) == 18
    assert all(env.closed and env.evaluated for env in environments)
    assert all(task['metrics']['task_success'] for task in result['tasks'])
    first_verification = next(event for event in events if event['type'] == 'escape_result')
    assert first_verification['observed_cycles'] == 10 and first_verification['task_id'] == 'dev-a'
    later_rescue = next(event for event in events if event['type'] == 'rescue_injected' and event['task_id'] == 'dev-b')
    assert later_rescue['retrieved_entry_ids'] == first['end_entry_ids']
    second_prompts = [prompt for (prompt, _), call in zip(client.calls, result['calls']) if call['task_id'] == 'dev-b']
    assert len(second_prompts) == 3
    assert all('dev-a' not in prompt and '78123' not in prompt for prompt in second_prompts)
    assert principle not in second_prompts[0] and principle not in second_prompts[1]
    assert principle in second_prompts[2]
    assert all(event['task_index'] == (0 if event['task_id'] == 'dev-a' else 1) for event in events)


@pytest.mark.parametrize('limit,reason', [({'max_model_calls': 2}, 'model_call_limit'),
                                         ({'max_cost_usd': 0.002}, 'cost_threshold')])
def test_stream_budget_is_global_and_does_not_open_next_world_when_exhausted(tmp_path, monkeypatch, limit, reason):
    result, environments, events = _run_stream(tmp_path, monkeypatch, Client(['finish']), **limit)
    assert result['stop_reason'] == reason and result['error'] == {'type': 'RunLimitError'}
    assert result['stopped_before_task_id'] == 'dev-b'
    assert len(environments) == len(result['tasks']) == 1 and result['model_calls'] == 2
    assert result['tasks'][0]['error'] is None and result['tasks'][0]['metrics']['task_success']
    assert all(event['task_id'] == 'dev-a' for event in events)


def test_stream_budget_during_second_task_preserves_first_and_scores_partial_second(tmp_path, monkeypatch):
    result, environments, events = _run_stream(tmp_path, monkeypatch,
        Client(['finish', 'inspect']), max_model_calls=4)
    first, second = result['tasks']
    assert result['stop_reason'] == 'model_call_limit' and result['model_calls'] == 4
    assert first['status'] == 'finished' and first['metrics']['task_success']
    assert second['status'] == 'error' and second['metrics']['task_success'] is False
    assert second['observed_cycles'] == 1 and second['model_calls'] == 2
    assert [env.executed for env in environments] == [['finish'], ['inspect']]
    assert events[-1]['type'] == 'task_finalized' and events[-1]['task_id'] == 'dev-b'


def test_action_budget_resets_per_task_and_does_not_claim_task_success(tmp_path, monkeypatch):
    result, environments, _ = _run_stream(tmp_path, monkeypatch, Client(['inspect', 'inspect']), max_steps=1)
    assert result['stop_reason'] == 'task_sequence_finished' and result['error'] is None
    assert result['observed_cycles'] == result['model_calls'] == 2
    assert all(task['stop_reason'] == 'max_steps' and task['metrics']['task_success'] is False
               for task in result['tasks'])
    assert [len(env.executed) for env in environments] == [1, 1]
    assert [call['task_index'] for call in result['calls']] == [0, 1]


def test_interrupted_second_task_preserves_completed_task_checkpoint(tmp_path, monkeypatch):
    class InterruptedSecondClient(Client):
        def __call__(self, prompt, **kwargs):
            if self.calls and not prompt.startswith('[SENTRY DETECTOR]'):
                raise KeyboardInterrupt('PRIVATE_INTERRUPT')
            return super().__call__(prompt, **kwargs)
    result, environments, events = _run_stream(tmp_path, monkeypatch, InterruptedSecondClient(['finish']))
    first, second = result['tasks']
    assert first['status'] == 'finished' and first['metrics']['task_success']
    assert second['status'] == 'error' and second['error'] == {'type': 'KeyboardInterrupt'}
    assert second['metrics'] is None and not environments[1].evaluated
    assert result['calls'][-1]['task_id'] == 'dev-b'
    assert result['calls'][-1]['error_type'] == 'KeyboardInterrupt'
    assert result['calls'][-1]['usage'] == {}
    assert events[-1]['type'] == 'task_finalized'
    assert 'PRIVATE_INTERRUPT' not in json.dumps(result) + json.dumps(events)


@pytest.mark.parametrize('selection', [[], 'dev-a', ['dev-a', 'dev-a'], ['missing'], [True]])
def test_invalid_task_sequence_is_rejected_before_client_or_world(tmp_path, monkeypatch, selection):
    monkeypatch.setattr(run, 'load_task_ids', lambda *a, **kw: ['dev-a', 'dev-b'])
    client = Client()
    with pytest.raises(ValueError):
        run.run_appworld(config(), root=tmp_path, task_ids=selection, client=client,
            environment_factory=lambda *a, **kw: pytest.fail('Invalid selection must not open a world'))
    assert client.calls == []


def test_cli_preserves_repeated_task_id_order(tmp_path, monkeypatch, capsys):
    path = tmp_path / 'config.yaml'
    path.write_text(json.dumps(run.asdict(config())))
    captured = {}
    def fake_run(config, **kwargs):
        captured.update(kwargs)
        return {'output_dir': 'local', 'task_id': None, 'task_ids': kwargs['task_ids'],
                'stop_reason': 'task_sequence_finished', 'model_calls': 0,
                'known_cost_usd': 0, 'metrics': None, 'error': None, 'tasks': []}
    monkeypatch.setattr(run, 'run_appworld', fake_run)
    assert run.main(['--config', str(path), '--task-id', 'dev-b', '--task-id', 'dev-a']) == 0
    assert captured['task_ids'] == ['dev-b', 'dev-a']
    assert json.loads(capsys.readouterr().out)['task_ids'] == ['dev-b', 'dev-a']
