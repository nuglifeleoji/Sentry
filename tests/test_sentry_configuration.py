from __future__ import annotations

import json
from dataclasses import replace

import pytest

from Sentry import ModelConfig, ProviderConfig, SentryConfig, SentryRunner
from Sentry.cli import describe_config, main


@pytest.mark.parametrize('section,patch', [
    ('trace', {'window_size': 5.0}),
    ('trace', {'post_rescue_horizon': True}),
    ('retrieval', {'top_k': 5.0}),
    ('retrieval', {'primary_label_cap': 2.0}),
    ('model', {'temperature': False}),
    ('model', {'temperature': float('nan')}),
    ('model', {'temperature': 10 ** 1000}),
    ('model', {'max_tokens': 4096.0}),
    ('model', {'model': ' '}),
    ('model', {'model': 9}),
    ('model', {'structured_outputs': 'true'}),
    ('task_agent', {'max_tokens': 384.0}),
    ('logging', {'enabled': 'false'}),
    ('logging', {'log_path': True}),
    ('playbook', {'persist_path': True}),
    ('api', {'api_key': 'must-not-appear-in-yaml'}),
])
def test_rejects_settings_that_would_silently_change_a_run(section, patch):
    with pytest.raises(ValueError):
        SentryConfig.from_dict({section: patch})


@pytest.mark.parametrize('field', ['enable_updates', 'load_persisted'])
@pytest.mark.parametrize('value', [True, False, 'true', 'false'])
def test_removed_memory_mode_switches_are_rejected(field, value):
    with pytest.raises(ValueError, match='Unsupported PlaybookConfig fields'):
        SentryConfig.from_dict({'playbook': {field: value}})


def test_yaml_rejects_duplicate_keys_and_nonstring_keys(tmp_path):
    path = tmp_path / 'config.yaml'
    for source in ('model: {}\nmodel: {}\n', 'model:\n  model: first\n  model: second\n', '1: value\n'):
        path.write_text(source)
        with pytest.raises(ValueError):
            SentryConfig.from_yaml(path)


def test_api_model_is_explicit_single_source_and_conflicts_are_errors():
    source = {'api': {'provider': 'openai_compatible', 'model': 'deployment-id', 'base_url': 'http://localhost:8000/v1'}}
    config = SentryConfig.from_dict(source)
    assert config.model.model == config.api.model == 'deployment-id'
    assert 'model' not in source  # no mutation of the caller's configuration
    mirror = SentryConfig.from_dict({'model': {'model': 'deployment-id'}, 'api': {'provider': 'openai'}})
    assert mirror.api.model == 'deployment-id'
    with pytest.raises(ValueError, match='same shared backbone'):
        SentryConfig.from_dict(source | {'model': {'model': 'different'}})


def test_runner_builds_one_configured_client_without_inference(monkeypatch):
    calls = []
    class FakeClient:
        model = 'deployment-id'
        def __call__(self, prompt, **kwargs):
            calls.append(kwargs)
            return '{}'
    created = []
    def factory(config):
        created.append(config)
        return FakeClient()
    monkeypatch.setattr('Sentry.integration.create_client', factory)
    config = SentryConfig.from_dict({'api': {'provider': 'openai', 'model': 'deployment-id'}})
    runner = SentryRunner(config=config)
    assert created == [config.api]
    assert calls == []
    runner._structured('JSON')
    runner._repair('action')
    assert calls == [
        {'temperature': 0.0, 'max_tokens': 4096, 'response_format': {'type': 'json_object'}},
        {'temperature': 0.7, 'max_tokens': 384},
    ]
    with pytest.raises(ValueError, match='either'):
        SentryRunner(FakeClient(), config)


def test_runner_refuses_model_mismatch_and_records_implicit_client_identity():
    class Client:
        model = 'deployment-id'
        def __call__(self, prompt, **kwargs):
            return '{}'
    assert SentryRunner(Client()).config.model.model == 'deployment-id'
    with pytest.raises(ValueError, match='identifier differs'):
        SentryRunner(Client(), SentryConfig())
    runner = SentryRunner(Client(), replace(SentryConfig(), model=ModelConfig(model='deployment-id')))
    assert runner.config.model.model == 'deployment-id'


def test_configuration_manifest_is_offline_and_does_not_read_key(monkeypatch, tmp_path):
    monkeypatch.setenv('SENTRY_API_KEY', 'confidential-test-key')
    path = tmp_path / 'config.yaml'
    path.write_text('api:\n  provider: openai\n  model: deployment-id\n  api_key_env: SENTRY_API_KEY\n')
    report = describe_config(path)
    assert report['api_configured']
    assert report['format'] == 'sentry-config-manifest-v2'
    assert 'confidential-test-key' not in json.dumps(report)
    assert len(report['source_sha256']) == 64
    assert report['config']['playbook'] == {'persist_path': None}
    assert report['declared_model_identity'] == 'deployment-id'


def test_validator_will_not_overwrite_existing_files(monkeypatch, tmp_path):
    path = tmp_path / 'config.yaml'
    source = 'trace:\n  window_size: 5\n'
    path.write_text(source)
    monkeypatch.setattr('sys.argv', ['sentry-validate', str(path), '--manifest', str(path)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert path.read_text() == source


def test_all_shipped_configuration_templates_parse():
    from pathlib import Path
    configs = Path(__file__).resolve().parents[1] / 'configs'
    for path in configs.rglob('*.yaml'):
        config = SentryConfig.from_yaml(path)
        assert config.trace.window_size == 5
        assert config.trace.post_rescue_horizon == 10
