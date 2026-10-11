"""Run AppWorld tasks sequentially with online Sentry memory and explicit limits."""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
import os
import re
from pathlib import Path
import shlex
import time
import uuid

from Sentry.clients import ModelOutputTruncatedError, _PROVIDER_DEFAULTS, create_client
from Sentry.config import LoggingConfig, PlaybookConfig, SentryConfig
from Sentry.integration import SentryRunner, model_to_text_judge
from Sentry.models import agent_step_from_parts

from .adapter import AppWorldEnvironment, load_task_ids

PROMPT_LIMIT = 200_000
OBSERVATION_LIMIT = 20_000
_USAGE_FIELDS = {
    'cost', 'prompt_tokens', 'completion_tokens', 'total_tokens', 'input_tokens', 'output_tokens',
    'cache_creation_input_tokens', 'cache_read_input_tokens', 'promptTokenCount',
    'candidatesTokenCount', 'totalTokenCount', 'cachedContentTokenCount', 'thoughtsTokenCount',
}


class RunLimitError(RuntimeError):
    """A host budget prevented another model invocation."""


def load_credential_file(path: Path, key_name: str) -> None:
    """Read one literal credential only; environment wins, and nothing is sourced."""
    if os.environ.get(key_name):
        return
    matches = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        text = line.strip()
        if text.startswith('export '):
            text = text[7:].lstrip()
        if '=' not in text:
            continue
        name, value = text.split('=', 1)
        if name.strip() != key_name:
            continue
        try:
            values = shlex.split(value, comments=True, posix=True)
        except ValueError:
            raise ValueError('Invalid credential assignment') from None
        if len(values) != 1 or not values[0]:
            raise ValueError('Expected one nonempty credential literal')
        matches.append(values[0])
    if len(matches) != 1:
        raise ValueError('Credential file must define the selected variable exactly once')
    os.environ[key_name] = matches[0]


def _finite_cost(value):
    try:
        return type(value) in {int, float} and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _elapsed_clock():
    # AppWorld/freezegun freezes time.monotonic; the OS clock remains real.
    return time.clock_gettime(time.CLOCK_MONOTONIC) if hasattr(time, 'clock_gettime') else time.monotonic()


def _call_timestamp():
    if hasattr(time, 'clock_gettime'):
        return datetime.fromtimestamp(time.clock_gettime(time.CLOCK_REALTIME), timezone.utc).isoformat()
    return datetime.now(timezone.utc).isoformat()


def _usage(value):
    if not isinstance(value, dict):
        return {}
    return {key: number for key, number in value.items() if key in _USAGE_FIELDS and _finite_cost(number)}


def _response_metadata(value):
    if not isinstance(value, dict):
        return {}
    return {key: text for key, text in value.items()
            if key in {'id', 'model', 'provider', 'finish_reason'}
            and isinstance(text, str) and text.strip() and len(text) <= 512
            and all(character.isprintable() for character in text)}


def _observation(text):
    if len(text) <= OBSERVATION_LIMIT:
        return text
    return text[:OBSERVATION_LIMIT - 100] + '\n[Observation truncated by host at 20,000 characters.]'


class _MeteredModel:
    def __init__(self, client, *, model, max_calls, max_cost_usd, before_call):
        self.client = client
        self.model = getattr(client, 'model', model)
        self.max_calls, self.max_cost_usd = max_calls, max_cost_usd
        self.before_call = before_call
        self.calls = []
        self.cost = 0.0
        self.missing_cost = 0
        self.task_context = {}
        self.after_call = lambda: None

    def check_limits(self):
        if len(self.calls) >= self.max_calls:
            raise RunLimitError('model_call_limit')
        if self.max_cost_usd is not None:
            if self.missing_cost:
                raise RunLimitError('missing_cost')
            if self.cost >= self.max_cost_usd:
                raise RunLimitError('cost_threshold')

    def __call__(self, prompt, **kwargs):
        self.before_call()
        if len(prompt) > PROMPT_LIMIT:
            raise RunLimitError('prompt_limit')
        self.check_limits()
        phase = 'task_agent'
        for header, name in (('[SENTRY DETECTOR]', 'detector'),
                             ('[SENTRY ESCAPE VERIFIER]', 'verifier'),
                             ('[SENTRY PLAYBOOK UPDATE]', 'updater')):
            if prompt.startswith(header):
                phase = name
        call = {'call_id': len(self.calls) + 1, 'phase': phase, 'decoding': kwargs,
                'started_at': _call_timestamp(), **self.task_context}
        self.calls.append(call)
        self.client.last_usage = {}
        self.client.last_response_metadata = {}
        started = _elapsed_clock()
        try:
            return self.client(prompt, **kwargs)
        except (Exception, KeyboardInterrupt) as exc:
            call['error_type'] = type(exc).__name__
            raise
        finally:
            call['elapsed_seconds'] = round(_elapsed_clock() - started, 4)
            call['usage'] = _usage(getattr(self.client, 'last_usage', {}))
            call['response_metadata'] = _response_metadata(getattr(self.client, 'last_response_metadata', {}))
            cost = call['usage'].get('cost')
            if _finite_cost(cost):
                self.cost += cost
            else:
                self.missing_cost += 1
            self.after_call()


def _error_details(exc):
    result = {'type': type(exc).__name__}
    status = re.fullmatch(r'Model endpoint returned HTTP ([0-9]{3})(?:; check model support for the explicit decoding and JSON settings)?', str(exc))
    if isinstance(exc, RuntimeError) and status:
        result['http_status'] = int(status[1])
    return result


def _stop_reason(exc):
    code = str(exc) if isinstance(exc, RunLimitError) else None
    return code if code in {'prompt_limit', 'model_call_limit', 'missing_cost', 'cost_threshold'} else 'error'


def _run_task(task, *, runner, metered, generate, environment_factory, root,
              run_id, max_steps, drain, progress):
    """Use a fresh world/conversation while retaining the runner's online memory."""
    try:
        with environment_factory(task['task_id'], root=root,
                                 experiment_name=f"sentry-{run_id}-{task['task_index']}",
                                 max_interactions=max_steps) as environment:
            task['appworld_version'] = environment.version
            context = environment.task_context
            last_observation = _observation(environment.initial_observation)
            history = [
                'Return one short, incremental action using the task protocol. '
                f'Your output budget is {runner.config.task_agent.max_tokens} tokens per action. '
                'Inspect API documentation before calling unfamiliar APIs. '
                'Use the persistent Python session to continue the task over multiple actions.',
                context.objective, context.action_schema, last_observation,
            ]
            guidance = None
            runner.start_task(objective=context.objective, action_schema=context.action_schema)
            drain()
            try:
                for step_id in range(max_steps):
                    try:
                        prompt = '\n\n'.join(history)
                        if guidance is not None:
                            prompt += '\n\n' + guidance
                        raw = generate(prompt + '\n\nNext action:')
                    except ModelOutputTruncatedError as exc:
                        raw = exc.partial_text
                        step = agent_step_from_parts(
                            step_id=step_id, reasoning='', raw_action=raw,
                            observation='Incomplete model output was rejected without execution.\n' + last_observation,
                            parsed_ok=False, schema_valid=False, accepted_by_environment=None,
                            parser_error=(
                                f'The generation reached its {runner.config.task_agent.max_tokens}-token limit. '
                                'Return one shorter, complete action for the next step only; '
                                'continue the remaining work in later actions.'
                            ),
                        )
                    else:
                        step = environment.execute(raw, step_id=step_id)
                    step = replace(step, observation=_observation(step.observation))
                    task['observed_cycles'] += 1
                    last_observation = step.observation
                    history.extend(['Action: ' + raw, 'Observation: ' + step.observation])
                    completed = environment.task_completed()
                    completion_requested = (completed and step.metadata.get('completion_requested') is True
                                            and step.action.accepted_by_environment is True)
                    budget_exhausted = task['observed_cycles'] == max_steps
                    # A model's completion declaration is not an execution limit.
                    # Give Sentry its ordinary check, retaining the existing pause
                    # on new soft detection while recovery verification is pending.
                    repair = runner.step(step, terminal=budget_exhausted)
                    drain()
                    # Each intervention guides the next action. Its embedded
                    # trajectory must not accumulate again in the agent history.
                    guidance = repair.prompt_text if repair is not None else None
                    if completion_requested and repair is not None:
                        history.append(
                            'Host: Sentry issued repair guidance on the completion action. '
                            'Continue the task in this same Python session. The previous '
                            'completion flag does not end this run; call '
                            'apis.supervisor.complete_task(...) again when ready. '
                            'No official evaluation has been performed.'
                        )
                    progress(f"task={task['task_id']} cycle={task['observed_cycles']} "
                             f"calls={len(metered.calls)} interventions={task['interventions']} "
                             f"learned={len(runner.guard.playbook.entries)}")
                    if (completion_requested and repair is None) or budget_exhausted:
                        task['stop_reason'] = 'task_completed' if completion_requested else 'max_steps'
                        break
            except RunLimitError as exc:
                task['error'] = _error_details(exc)
                task['stop_reason'] = _stop_reason(exc)
            finally:
                runner.finalize()
                drain()
            task['metrics'] = environment.evaluate()
    except (Exception, KeyboardInterrupt) as exc:
        task['error'] = _error_details(exc)
        task['stop_reason'] = _stop_reason(exc)
    finally:
        if runner.guard.task is not None:
            runner.finalize()
        task['status'] = 'error' if task['error'] else 'finished'
        task['finished_at'] = _call_timestamp()
        drain()


def run_appworld(config, *, root, task_id=None, task_ids=None, max_steps=30, max_model_calls=64,
                 output_dir='outputs/appworld', max_cost_usd=None, client=None,
                 environment_factory=AppWorldEnvironment, progress=print):
    """Run selected dev tasks in order; request/cost limits apply to the whole stream."""
    if not isinstance(config, SentryConfig) or config.api is None:
        raise ValueError('An explicit api configuration is required')
    for value in (max_steps, max_model_calls):
        if type(value) is not int or value <= 0:
            raise ValueError('Action and model-call limits must be positive integers')
    if max_cost_usd is not None and (not _finite_cost(max_cost_usd) or max_cost_usd == 0):
        raise ValueError('Cost threshold must be finite and positive')
    if config.playbook.persist_path is not None or config.logging.log_path is not None:
        raise ValueError('Run artifacts use --output-dir; remove configured playbook/log paths')
    if config.api.max_retries != 0:
        raise ValueError('Bounded runs require max_retries=0')
    available = load_task_ids('dev', root=root)
    if task_id is not None and task_ids is not None:
        raise ValueError('Supply task_id or task_ids, not both')
    selected = task_ids if task_ids is not None else [task_id if task_id is not None else next(iter(available), None)]
    if (not isinstance(selected, Sequence) or isinstance(selected, (str, bytes)) or not selected
            or any(not isinstance(item, str) or item not in available for item in selected)):
        raise ValueError('Select tasks from the AppWorld dev dataset')
    selected = list(selected)
    if len(set(selected)) != len(selected):
        raise ValueError('Select distinct task IDs')
    client = create_client(config.api) if client is None else client
    if getattr(client, 'max_retries', 0) != 0:
        raise ValueError('Bounded runs require a client with zero retries')
    if getattr(client, 'model', config.model.model) != config.model.model:
        raise ValueError('Client model differs from configured model')
    started_at = datetime.now(timezone.utc)
    run_id = started_at.strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    output = Path(output_dir).expanduser().resolve() / run_id
    output.mkdir(parents=True, exist_ok=False)
    effective = replace(config, playbook=PlaybookConfig(persist_path=str(output / 'playbook.json')),
                        logging=LoggingConfig(enabled=False))
    runner, current_task = None, None
    event_counts = {'interventions': 0, 'verified_recoveries': 0, 'learned_entries': 0}
    result = {
        'format': 'sentry-appworld-run-v2', 'run_id': run_id,
        'task_id': selected[0] if len(selected) == 1 else None, 'task_ids': selected, 'tasks': [],
        'started_at': started_at.isoformat(), 'status': 'running',
        'event_time_basis': 'appworld_simulated_time',
        'dataset': 'dev', 'appworld_version': None, 'root': str(Path(root).expanduser().resolve()),
        'requested_config': asdict(config), 'effective_config': asdict(effective),
        'limits': {'max_steps': max_steps, 'max_steps_scope': 'per_task',
                   'max_model_calls': max_model_calls, 'model_call_scope': 'run',
                   'max_cost_usd': max_cost_usd, 'cost_scope': 'run',
                   'prompt_characters': PROMPT_LIMIT, 'observation_characters': OBSERVATION_LIMIT,
                   'cost_policy': 'Reported-cost threshold checked between calls; one call can exceed it.'},
        'stop_reason': 'running', 'metrics': None, 'error': None, 'output_dir': str(output),
    }
    with (output / 'events.jsonl').open('w', encoding='utf-8') as event_file:
        def checkpoint():
            if current_task is not None:
                current_task['end_model_calls'] = len(metered.calls)
                current_task['model_calls'] = len(metered.calls) - current_task['start_model_calls']
                current_task['end_entry_ids'] = [entry.insight_id for entry in runner.guard.playbook.entries]
                current_task['end_entries'] = len(current_task['end_entry_ids'])
            result.update(observed_cycles=sum(task['observed_cycles'] for task in result['tasks']),
                          model_calls=len(metered.calls), calls=metered.calls, known_cost_usd=metered.cost,
                          missing_cost_calls=metered.missing_cost, **event_counts)
            pending = output / 'run.json.tmp'
            pending.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            pending.replace(output / 'run.json')

        def drain():
            if runner is None:
                return
            for event in runner.guard.drain_runtime_events():
                event.update(metered.task_context)
                event_file.write(json.dumps(event, ensure_ascii=False) + '\n')
                count = None
                if event['type'] == 'rescue_injected':
                    count = 'interventions'
                elif event['type'] == 'escape_result' and event['resolved']:
                    count = 'verified_recoveries'
                elif event['type'] == 'playbook_update':
                    count = 'learned_entries'
                if count:
                    event_counts[count] += 1
                    current_task[count] += 1
            event_file.flush()
            checkpoint()

        metered = _MeteredModel(client, model=config.model.model, max_calls=max_model_calls,
                                max_cost_usd=max_cost_usd, before_call=drain)
        runner = SentryRunner(metered, replace(effective, api=None), capture_runtime_events=True)
        metered.after_call = checkpoint
        generate = model_to_text_judge(metered, runner.config.task_agent, structured=False)
        runner.guard.playbook.persist()
        checkpoint()
        for index, selected_id in enumerate(selected):
            try:
                metered.check_limits()
            except RunLimitError as exc:
                result.update(error=_error_details(exc), stop_reason=_stop_reason(exc),
                              stopped_before_task_id=selected_id)
                break
            entry_ids = [entry.insight_id for entry in runner.guard.playbook.entries]
            current_task = {
                'task_index': index, 'task_id': selected_id, 'status': 'running',
                'started_at': _call_timestamp(), 'start_model_calls': len(metered.calls),
                'start_entries': len(entry_ids), 'start_entry_ids': entry_ids,
                'observed_cycles': 0, 'metrics': None, 'error': None, 'appworld_version': None,
                'stop_reason': 'running', 'interventions': 0, 'verified_recoveries': 0, 'learned_entries': 0,
            }
            result['tasks'].append(current_task)
            metered.task_context = {'task_id': selected_id, 'task_index': index}
            result['stop_reason'] = 'running'
            checkpoint()
            _run_task(current_task, runner=runner, metered=metered, generate=generate,
                      environment_factory=environment_factory, root=root, run_id=run_id,
                      max_steps=max_steps, drain=drain, progress=progress)
            result.update(error=current_task['error'], stop_reason=current_task['stop_reason'],
                          appworld_version=current_task['appworld_version'])
            if len(selected) == 1:
                result['metrics'] = current_task['metrics']
            checkpoint()
            if result['error']:
                break
        if not result['error'] and len(selected) > 1:
            result['stop_reason'] = 'task_sequence_finished'
        result['status'] = 'error' if result['error'] else 'finished'
        checkpoint()
    return result


def main(argv=None):
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument('--config', type=Path)
    command.add_argument('--env-file', type=Path)
    command.add_argument('--root', type=Path, default=Path(os.environ.get('APPWORLD_ROOT', '.')))
    command.add_argument('--task-id', action='append', help='Repeat for distinct dev tasks in execution order')
    command.add_argument('--max-steps', type=int, default=30)
    command.add_argument('--max-model-calls', type=int, default=64)
    command.add_argument('--max-cost-usd', type=float)
    command.add_argument('--output-dir', type=Path, default=Path('outputs/appworld'))
    command.add_argument('--list-tasks', action='store_true', help='List dev IDs without model calls or credentials')
    args = command.parse_args(argv)
    try:
        if args.list_tasks:
            print('\n'.join(load_task_ids('dev', root=args.root)))
            return 0
        if args.config is None:
            raise ValueError('config_required')
        config = SentryConfig.from_yaml(args.config)
        if config.api is None:
            raise ValueError('api_required')
        if args.env_file:
            key_name = config.api.api_key_env or _PROVIDER_DEFAULTS[config.api.provider][1]
            if key_name is None:
                raise ValueError('credential_variable_required')
            load_credential_file(args.env_file, key_name)
        result = run_appworld(config, root=args.root, task_ids=args.task_id, max_steps=args.max_steps,
                              max_model_calls=args.max_model_calls, output_dir=args.output_dir,
                              max_cost_usd=args.max_cost_usd)
    except Exception as exc:
        guidance = ('Check --config, exported credential, positive limits, and dev task ID. '
                    'Use --output-dir instead of playbook/log file paths. '
                    "For AppWorld setup: pip install -e '.[appworld]', run appworld install, "
                    'and appworld download data --root PATH; set --root to that PATH.')
        print(json.dumps({'status': 'error', **_error_details(exc), 'guidance': guidance}))
        return 1
    summary = {key: result[key] for key in ('output_dir', 'task_id', 'task_ids', 'stop_reason',
                                           'model_calls', 'known_cost_usd', 'metrics', 'error')}
    summary['tasks'] = [{key: task[key] for key in ('task_id', 'stop_reason', 'observed_cycles',
                                                  'learned_entries', 'metrics', 'error')}
                        for task in result['tasks']]
    print(json.dumps(summary, ensure_ascii=False))
    return 1 if result['error'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
