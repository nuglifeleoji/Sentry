from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from benchmarks.appworld.adapter import AppWorldEnvironment, load_task_ids, parse_python_action
from Sentry.detector import detect_action_validity


class PublicTask:
    instruction = "Find the requested item using public APIs."
    supervisor = {"first_name": "Ada", "last_name": "L", "email": "ada@example.com", "secret": "not public"}
    datetime = "2024-01-01T12:00:00"
    allowed_apps = ["api_docs", "supervisor", "spotify"]
    app_descriptions = {"spotify": "Music application"}
    api_docs = {
        "api_docs": {"show_api_doc": {"description": "Read API documentation", "parameters": []}},
        "supervisor": {"complete_task": {"description": "Finish task", "parameters": []}},
    }

    @property
    def ground_truth(self):
        raise AssertionError("Adapter must not inspect ground truth")


@pytest.fixture
def environment(tmp_path, monkeypatch):
    for name in ("datasets", "base_dbs", "tasks", "api_docs"):
        (tmp_path / "data" / name).mkdir(parents=True)
    events = []

    class World:
        def __init__(self, **kwargs):
            self.task = PublicTask()
            self.response = "Execution successful."
            self.completed = False
            self.error = None
            self.calls = []
            self.requester = SimpleNamespace(requests=[])
            self.next_requests = []
            self.evaluations = 0
            events.append(("open", kwargs))

        def execute(self, code):
            self.calls.append(code)
            self.requester.requests.extend(self.next_requests)
            self.next_requests = []
            if self.error:
                raise self.error
            return self.response

        def task_completed(self):
            return self.completed

        def evaluate(self):
            self.evaluations += 1
            return SimpleNamespace(success=False, pass_count=1, fail_count=2, num_tests=3, pass_percentage=33.3)

        def close(self):
            events.append(("close", None))

    module = SimpleNamespace(
        AppWorld=World, __version__="0.1.3", update_root=lambda root: events.append(("root", root)),
        load_task_ids=lambda split: [f"{split}-task"],
    )
    monkeypatch.setitem(sys.modules, "appworld", module)
    with AppWorldEnvironment("example-task", root=tmp_path, experiment_name="unit-test") as env:
        yield env, events, tmp_path


@pytest.mark.parametrize("source, expected", [
    ("x = 1\nprint(x)", "x = 1\nprint(x)"),
    ("```python\nprint(1)\n```", "print(1)"),
    ("```py\r\nprint(1)\r\n```", "print(1)"),
    ("```\nprint(1)\n```", "print(1)"),
    ("print('```')", "print('```')"),
    ("# Reason: Inspect.\n\n```python\nprint(1)\n```", "# Reason: Inspect.\n\nprint(1)"),
    ("# A comment mentioning ```\n```python\nprint(1)\n```", "# A comment mentioning ```\nprint(1)"),
])
def test_parser_accepts_entire_plain_or_fenced_code(source, expected):
    assert parse_python_action(source) == expected


@pytest.mark.parametrize("source", [
    "", "# only a comment", "for", "return 1", "```python\nprint(1)",
    "Thought: inspect\n```python\nprint(1)\n```", "```python\nprint(1)\n```\nDone.",
    "```javascript\nconsole.log(1)\n```", "```python\nprint(1)\n```\n```python\nprint(2)\n```",
    "# Reason: Inspect.\n```python\nprint(1)",
    "# Reason: Inspect.\n```python\nprint(1)\n```\nExtra prose",
    "print('outside code')\n```python\nprint(1)\n```",
])
def test_parser_rejects_incomplete_or_ambiguous_actions(source):
    with pytest.raises(ValueError):
        parse_python_action(source)


def test_parser_does_not_execute_code(tmp_path):
    target = tmp_path / "must-not-exist"
    parse_python_action(f"open({str(target)!r}, 'w').write('unexpected')")
    assert not target.exists()


def test_syntax_rejection_has_no_world_side_effects(environment):
    env, _, _ = environment
    step = env.execute("return 1", step_id=0)
    assert env._world.calls == []
    assert not step.action.parsed_ok and not step.action.schema_valid
    assert step.action.accepted_by_environment is None
    assert not step.metadata["executed"]
    assert detect_action_validity(step) is not None
    assert step.task_progress_score is None
    assert env._world.evaluations == 0


def test_world_execution_keeps_code_and_reasoning_separate(environment):
    env, _, _ = environment
    step = env.execute("```python\nx = 1\nprint(x)\n```", step_id=0, reasoning="Inspect state.")
    assert env._world.calls == ["x = 1\nprint(x)"]
    assert step.reasoning == "Inspect state."
    assert step.action.tool_args == {"code": "x = 1\nprint(x)"}
    assert step.action.accepted_by_environment
    assert step.action.parsed_ok and step.action.schema_valid
    assert step.metadata["environment"] == "appworld"
    assert step.task_progress_score is None
    assert env._world.evaluations == 0


@pytest.mark.parametrize("failure", [
    "KeyError: 'access_token'", "NameError: name 'missing_variable' is not defined",
    "Exception: Response status code is 401:\nInvalid credentials",
    "Exception: Response status code is 404:\nPlaylist not found",
    "Execution timed out after 30 seconds.",
])
def test_runtime_failure_is_not_automatically_a_schema_failure(environment, failure):
    env, _, _ = environment
    env._world.response = "Execution failed. Traceback:\n" + failure
    step = env.execute("print('example')", step_id=0)
    assert step.action.parsed_ok and step.action.schema_valid
    assert step.action.accepted_by_environment is False
    assert detect_action_validity(step) is None


@pytest.mark.parametrize("failure", [
    "Exception: No API named 'invented_api' found in the spotify app.",
    "Exception: No app named 'invented_app' found.",
    "Exception: Response status code is 422:\nMissing required parameter",
    "Usage of the following module is not allowed: os.",
])
def test_known_unavailable_actions_are_schema_failures(environment, failure):
    env, _, _ = environment
    env._world.response = "Execution failed. Traceback:\n" + failure
    step = env.execute("print('example')", step_id=0)
    assert step.action.parsed_ok and not step.action.schema_valid
    assert step.action.accepted_by_environment is False
    assert step.action.schema_error == env._world.response
    assert detect_action_validity(step) is not None


def test_exception_text_in_successful_output_is_not_a_schema_failure(environment):
    env, _, _ = environment
    env._world.response = "Documentation: No API named 'example' found in the spotify app."
    step = env.execute("print('documentation')", step_id=0)
    assert step.action.schema_valid and step.action.accepted_by_environment


def test_infrastructure_errors_propagate(environment):
    env, _, _ = environment
    env._world.error = ConnectionError("Environment unavailable")
    with pytest.raises(ConnectionError, match="unavailable"):
        env.execute("print(1)", step_id=0)


def test_public_prompt_allowlist_and_official_safety_settings(environment):
    env, events, _ = environment
    observation = json.loads(env.initial_observation)
    assert env.task_context.objective == PublicTask.instruction
    assert observation["supervisor"]["first_name"] == "Ada"
    assert "secret" not in observation["supervisor"]
    assert "initial_api_documentation" in observation
    assert "ground_truth" not in observation and "answer" not in observation
    assert env.version == "0.1.3"
    kwargs = next(value for name, value in events if name == "open")
    assert kwargs["ground_truth_mode"] == "minimal"
    assert kwargs["raise_on_failure"]
    assert kwargs["raise_on_unsafe_syntax"]
    assert kwargs["null_patch_unsafe_execution"]
    assert kwargs["max_interactions"] == 1000
    assert kwargs["timeout_seconds"] == 100
    assert kwargs["random_seed"] == 100


def test_completion_is_distinct_from_official_success_and_evaluation_is_terminal(environment):
    env, _, _ = environment
    env._world.completed = True
    env._world.next_requests = [{'method': 'post', 'url': '/supervisor/message', 'data': {'answer': 'wrong'}}]
    step = env.execute("apis.supervisor.complete_task(answer='wrong')", step_id=0)
    assert step.metadata["task_completed"]
    assert step.metadata["completion_requested"]
    assert "success" not in step.metadata
    assert env._world.evaluations == 0
    metrics = env.evaluate()
    assert metrics == {"success": False, "pass_count": 1, "fail_count": 2, "num_tests": 3, "pass_percentage": 33.3}
    metrics["success"] = True
    assert not env.evaluate()["success"]
    assert env._world.evaluations == 1
    with pytest.raises(RuntimeError, match="terminal evaluation"):
        env.execute("print(1)", step_id=1)


@pytest.mark.parametrize('source,expected', [
    ('# Reason: Inspect the visible record.\nprint(1)', 'Inspect the visible record.'),
    ('# Reason: Inspect the visible record.\n```python\nprint(1)\n```', 'Inspect the visible record.'),
    ('```python\n# Reasoning: Read the API schema.\nprint(1)\n```', 'Read the API schema.'),
    ('# Header\n\n# reason: Check state.\nprint(1)', 'Check state.'),
    ('print("# Reason: This is output, not a rationale.")', ''),
    ('print(1)\n# Reason: A later comment', ''),
    ('print(1)', ''),
])
def test_action_rationale_is_explicit_and_keeps_the_python_protocol(environment, source, expected):
    env, _, _ = environment
    step = env.execute(source, step_id=0)
    assert step.reasoning == expected
    assert step.action.raw == source
    assert env._world.calls == [parse_python_action(source)]


def test_explicit_host_reasoning_takes_precedence_over_comment(environment):
    env, _, _ = environment
    step = env.execute('# Reason: Model note.\nprint(1)', step_id=0, reasoning='Host-supplied rationale.')
    assert step.reasoning == 'Host-supplied rationale.'


def test_completion_metadata_tracks_fresh_requests_not_sticky_state_or_source_text(environment):
    env, _, _ = environment
    env._world.completed = True
    env._world.requester.requests.append({'method': 'post', 'url': '/supervisor/message', 'data': {}})
    step = env.execute('print("apis.supervisor.complete_task()")', step_id=0)
    assert step.metadata['task_completed'] and not step.metadata['completion_requested']
    env._world.next_requests = [{'method': 'post', 'url': '/supervisor/message',
                                'data': {'answer': 'PRIVATE_REQUEST_DATA'}}]
    step = env.execute('finish_alias(answer="updated")', step_id=1)
    assert step.metadata['completion_requested']
    assert 'PRIVATE_REQUEST_DATA' not in json.dumps(step.metadata)
    step = env.execute('print(1)', step_id=2)
    assert not step.metadata['completion_requested']


def test_budget_exhaustion_can_be_evaluated_without_completion(environment):
    env, _, _ = environment
    assert not env.task_completed()
    assert not env.evaluate()["success"]


def test_close_is_idempotent_and_context_cannot_be_reentered(environment):
    env, events, _ = environment
    env.close()
    env.close()
    assert events.count(("close", None)) == 1
    with pytest.raises(RuntimeError, match="context manager"):
        env.execute("print(1)", step_id=0)
    with pytest.raises(RuntimeError, match="twice"):
        env.__enter__()


def test_official_split_loading(environment):
    _, _, root = environment
    assert load_task_ids("dev", root=root) == ["dev-task"]
    with pytest.raises(ValueError, match="Unknown AppWorld split"):
        load_task_ids("../../other", root=root)


def test_optional_appworld_dependency_is_not_imported_at_module_import():
    result = subprocess.run([
        sys.executable, "-c",
        "import sys; import Sentry; assert 'benchmarks' not in sys.modules; "
        "import benchmarks.appworld.adapter; assert 'appworld' not in sys.modules",
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("kwargs", [
    {"task_id": "../task"}, {"experiment_name": "../outside"},
    {"max_interactions": 0}, {"timeout_seconds": True}, {"random_seed": "100"},
])
def test_invalid_environment_configuration_rejected(tmp_path, kwargs):
    options = {"task_id": "task", "root": tmp_path, "experiment_name": "test", **kwargs}
    with pytest.raises(ValueError):
        AppWorldEnvironment(**options)


def test_close_restores_upstream_global_exception_and_faulthandler_patches(environment):
    import builtins
    import faulthandler

    env, _, _ = environment
    system_exit, fault_handler_enabled = env._host_state
    builtins.SystemExit = Exception
    if fault_handler_enabled:
        faulthandler.disable()
    else:
        faulthandler.enable()
    env.close()
    assert builtins.SystemExit is system_exit
    assert faulthandler.is_enabled() is fault_handler_enabled
    with pytest.raises(system_exit) as error:
        sys.exit(2)
    assert error.value.code == 2


def test_failed_world_close_still_restores_caller_globals(environment):
    import builtins

    env, _, _ = environment
    system_exit = builtins.SystemExit
    builtins.SystemExit = Exception

    def fail_close():
        raise RuntimeError("upstream close failure")

    env._world.close = fail_close
    with pytest.raises(RuntimeError, match="upstream close failure"):
        env.close()
    assert builtins.SystemExit is system_exit


def test_failed_world_initialization_restores_caller_globals(tmp_path, monkeypatch):
    import builtins

    for name in ("datasets", "base_dbs", "tasks", "api_docs"):
        (tmp_path / "data" / name).mkdir(parents=True)
    system_exit = builtins.SystemExit

    def fail_open(**kwargs):
        builtins.SystemExit = Exception
        raise RuntimeError("upstream initialization failure")

    module = SimpleNamespace(AppWorld=fail_open, update_root=lambda root: None)
    monkeypatch.setitem(sys.modules, "appworld", module)
    with pytest.raises(RuntimeError, match="upstream initialization failure"):
        with AppWorldEnvironment("task", root=tmp_path, experiment_name="test"):
            pass
    assert builtins.SystemExit is system_exit


def test_interrupted_execution_restores_official_safety_hooks_before_propagating(environment):
    env, events, _ = environment
    env._world.safety_guard = SimpleNamespace(disable=lambda: events.append(("disable_safety", None)))
    env._world.error = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        env.execute("print(1)", step_id=0)
    assert ("disable_safety", None) in events


def test_failed_safety_cleanup_does_not_replace_execution_interrupt(environment):
    env, _, _ = environment

    def fail_cleanup():
        raise RuntimeError("cleanup failed")

    env._world.safety_guard = SimpleNamespace(disable=fail_cleanup)
    interrupt = KeyboardInterrupt()
    env._world.error = interrupt
    with pytest.raises(KeyboardInterrupt) as raised:
        env.execute("print(1)", step_id=0)
    assert raised.value is interrupt


def test_missing_data_has_actionable_error_before_loading_optional_sdk(tmp_path, monkeypatch):
    def unexpected_import(name):
        raise AssertionError("Do not import AppWorld before checking its data root")

    monkeypatch.setattr("benchmarks.appworld.adapter.importlib.import_module", unexpected_import)
    with pytest.raises(FileNotFoundError, match="appworld download data --root"):
        load_task_ids("dev", root=tmp_path)


def test_missing_optional_sdk_has_pinned_setup_guidance(environment, monkeypatch):
    _, _, root = environment

    def missing_import(name):
        raise ImportError("AppWorld is not installed")

    monkeypatch.setattr("benchmarks.appworld.adapter.importlib.import_module", missing_import)
    with pytest.raises(RuntimeError, match="appworld==0.1.3.post1"):
        load_task_ids("dev", root=root)


def test_failed_evaluation_still_prevents_further_execution(environment):
    env, _, _ = environment

    def fail_evaluate():
        raise RuntimeError("Evaluator unavailable")

    env._world.evaluate = fail_evaluate
    with pytest.raises(RuntimeError, match="Evaluator unavailable"):
        env.evaluate()
    with pytest.raises(RuntimeError, match="terminal evaluation"):
        env.execute("print(1)", step_id=0)
    assert env._world.calls == []


def test_root_resolution_uses_absolute_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    env = AppWorldEnvironment("task", root="world", experiment_name="test")
    assert env.root == tmp_path / "world"


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_partial_initialization_cleanup_preserves_original_error(tmp_path, monkeypatch, cleanup_fails):
    import builtins

    for name in ("datasets", "base_dbs", "tasks", "api_docs"):
        (tmp_path / "data" / name).mkdir(parents=True)
    original_error = RuntimeError("world initialization failed")
    original_exit = builtins.SystemExit
    cleaned = []

    class FailingWorld:
        def __init__(self, **kwargs):
            builtins.SystemExit = Exception
            raise original_error

        @classmethod
        def close_all(cls):
            cleaned.append(True)
            if cleanup_fails:
                raise RuntimeError("cleanup failed")

    monkeypatch.setitem(sys.modules, "appworld", SimpleNamespace(
        AppWorld=FailingWorld, update_root=lambda root: None,
    ))
    with pytest.raises(RuntimeError) as raised:
        with AppWorldEnvironment("task", root=tmp_path, experiment_name="test"):
            pytest.fail("Failed constructor must not enter the world")
    assert raised.value is original_error
    assert cleaned == [True]
    assert builtins.SystemExit is original_exit
