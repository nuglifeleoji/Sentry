"""Official AppWorld 0.1.3.post1 environment for the reference benchmark.

AppWorld owns execution and safety restrictions. Its process-wide state requires
sequential worlds or separate processes, never concurrent worlds in threads.
Official evaluation is terminal-only and never becomes agent-visible feedback.
"""
from __future__ import annotations

import ast
import builtins
import faulthandler
import importlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from Sentry.models import AgentStep, TaskContext, agent_step_from_parts

ACTION_SCHEMA = """Return executable Python code, plain or in one complete
```python code block with no surrounding prose. Include a brief
`# Reason: ...` Python comment before the executable statements, stating the
purpose of the next action. Put the comment inside the code block when fenced.
The persistent AppWorld session
provides `apis`; call APIs as apis.<app>.<api>(keyword=value). Inspect API names
and parameters through apis.api_docs. AppWorld does not automatically display
expression or API return values. Print results you need to observe, for example
`print(apis.api_docs.show_app_descriptions())`. Successful code with no printed
output yields only "Execution successful." Follow AppWorld's built-in execution
restrictions. Call apis.supervisor.complete_task(...)
when done, with answer=... if an answer is requested. Completion does not imply
benchmark success."""
_FENCE = re.compile(r"```(?:python|py)?[ \t]*\r?\n(?P<code>.*?)\r?\n```", re.DOTALL | re.IGNORECASE)
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
# Only concrete protocol errors, not general execution/semantic failures.
_SCHEMA_ERROR = re.compile(
    r"No API named '[^'\n]+' found in the [A-Za-z0-9_]+ app\."
    r"|No app named '[^'\n]+' found\."
    r"|Unexpected parameter '[^'\n]+' passed to apis\."
    r"|Response status code is 422:"
    r"|Usage of the following (?:module|modules|function|functions) is not allowed:"
)


def parse_python_action(raw_action: str) -> str:
    """Validate the complete action without executing or salvaging fragments."""
    if not isinstance(raw_action, str):
        raise ValueError("AppWorld action must be a string containing Python code")
    code = raw_action.strip()
    lines = code.splitlines(keepends=True)
    prefix_size = 0
    for line in lines:
        if line.strip() and not line.lstrip().startswith("#"):
            break
        prefix_size += len(line)
    body = code[prefix_size:].strip()
    if body.startswith("```"):
        match = _FENCE.fullmatch(body)
        if match is None:
            raise ValueError("Expected one complete Python code block and no surrounding prose")
        # Leading Python comments are harmless protocol annotations, including
        # a rationale placed before the fence. Keep them with the complete code.
        code = (code[:prefix_size] + match.group("code")).strip()
    if not code:
        raise ValueError("No Python code was provided")
    try:
        tree = ast.parse(code, filename="<appworld-action>", mode="exec")
        if not tree.body:
            raise ValueError("The action contains no executable Python statements")
        # Also rejects top-level return and other context-sensitive syntax.
        compile(tree, "<appworld-action>", "exec")
    except (SyntaxError, OverflowError) as exc:
        detail = f"{exc.msg} (line {exc.lineno})" if isinstance(exc, SyntaxError) else str(exc)
        raise ValueError(f"Invalid Python syntax: {detail}") from exc
    return code


def _action_reasoning(code: str) -> str:
    """Read an explicit action rationale from leading comments, never invent it."""
    for line in code.splitlines():
        line = line.strip()
        if not line:
            continue
        if not line.startswith("#"):
            break
        match = re.fullmatch(r"#\s*Reason(?:ing)?:\s*(.+)", line, re.IGNORECASE)
        if match is not None:
            return match.group(1).strip()
    return ""


def _appworld_module(root: str | Path) -> Any:
    root_path = Path(root).expanduser().resolve()
    required = ("datasets", "base_dbs", "tasks", "api_docs")
    missing = [name for name in required if not (root_path / "data" / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"AppWorld data is incomplete under {root_path}: missing {', '.join(missing)}. "
            "Run `appworld download data --root <appworld-root>`."
        )
    try:
        module = importlib.import_module("appworld")
    except ImportError as exc:
        raise RuntimeError(
            "AppWorld is optional. Use Python 3.11, install `appworld==0.1.3.post1`, "
            "then run `appworld install` and `appworld download data --root <appworld-root>`."
        ) from exc
    module.update_root(str(root_path))
    return module


def load_task_ids(dataset: str, *, root: str | Path) -> list[str]:
    """Read an official split; never infer tasks from ground-truth files."""
    if dataset not in {"train", "dev", "test_normal", "test_challenge"}:
        raise ValueError("Unknown AppWorld split; use train, dev, test_normal, or test_challenge")
    return list(_appworld_module(root).load_task_ids(dataset))


def _public_task_observation(task: Any) -> str:
    supervisor = task.supervisor
    identity = {
        name: supervisor.get(name, "") if isinstance(supervisor, Mapping) else getattr(supervisor, name, "")
        for name in ("first_name", "last_name", "email", "phone_number")
    }
    docs = {}
    for app, names in {
        "api_docs": ("show_app_descriptions", "show_api_descriptions", "show_api_doc", "search_api_docs"),
        "supervisor": ("show_active_task", "show_account_passwords", "complete_task"),
    }.items():
        available = task.api_docs.get(app, {})
        docs[app] = {
            name: {key: available[name][key] for key in ("description", "parameters") if key in available[name]}
            for name in names if name in available
        }
    # Explicit allowlist: no task.__dict__, ground_truth, answers or solution metadata.
    return json.dumps({
        "supervisor": identity, "datetime": str(task.datetime),
        "allowed_apps": list(task.allowed_apps), "app_descriptions": dict(task.app_descriptions),
        "initial_api_documentation": docs,
    }, ensure_ascii=False, indent=2)


class AppWorldEnvironment:
    """One world. Use a context manager and a unique experiment name per run.

    Completion is an observable declaration, separate from official success.
    The host may continue after it; only a fresh completion request can signal
    completion again after Sentry has intervened.
    Evaluation ends execution, including when the host exhausted its budget.
    Failed code can have partial effects; the adapter never retries execution.
    """

    def __init__(self, task_id: str, *, root: str | Path, experiment_name: str,
                 max_interactions: int = 1000, timeout_seconds: int = 100,
                 random_seed: int = 100) -> None:
        for name, value in (("task_id", task_id), ("experiment_name", experiment_name)):
            if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
                raise ValueError(f"{name} must be a nonempty path-safe identifier")
        for name, value in (("max_interactions", max_interactions), ("timeout_seconds", timeout_seconds)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(random_seed) is not int:
            raise ValueError("random_seed must be an integer")
        self.task_id, self.root = task_id, Path(root).expanduser().resolve()
        self.experiment_name = experiment_name
        self.max_interactions, self.timeout_seconds = max_interactions, timeout_seconds
        self.random_seed, self.version = random_seed, "unknown"
        self._world: Any = None
        self._entered = self._finished = False
        self._metrics: dict[str, Any] | None = None
        self._host_state: tuple[Any, bool] | None = None

    def __enter__(self) -> AppWorldEnvironment:
        if self._entered:
            raise RuntimeError("An AppWorldEnvironment cannot be entered twice")
        module = _appworld_module(self.root)
        self.version = str(getattr(module, "__version__", "unknown"))
        self._host_state = (builtins.SystemExit, faulthandler.is_enabled())
        try:
            self._world = module.AppWorld(
                task_id=self.task_id, experiment_name=self.experiment_name,
                max_interactions=self.max_interactions, timeout_seconds=self.timeout_seconds,
                random_seed=self.random_seed,
                # Minimal evaluator data stays internal; never serialize it.
                load_ground_truth=True, ground_truth_mode="minimal", raise_on_failure=True,
                raise_on_unsafe_syntax=True, null_patch_unsafe_execution=True,
            )
        except BaseException as error:
            # A failed constructor can leave DB caches or frozen time behind.
            # Only one world may be active in this process (see module contract).
            close_all = getattr(module.AppWorld, "close_all", None)
            if callable(close_all):
                try:
                    close_all()
                except BaseException:
                    if callable(getattr(error, "add_note", None)):
                        error.add_note("AppWorld partial-initialization cleanup also failed.")
            self._restore_host_globals()
            raise
        self._entered = True
        return self

    def _require_world(self) -> Any:
        if self._world is None:
            raise RuntimeError("AppWorldEnvironment must be used inside its context manager")
        return self._world

    @property
    def task_context(self) -> TaskContext:
        return TaskContext(objective=self._require_world().task.instruction, action_schema=ACTION_SCHEMA)

    @property
    def initial_observation(self) -> str:
        return _public_task_observation(self._require_world().task)

    def task_completed(self) -> bool:
        return bool(self._require_world().task_completed())

    def execute(self, raw_action: str, *, step_id: int, reasoning: str = "") -> AgentStep:
        world = self._require_world()
        if self._finished:
            raise RuntimeError("Cannot execute after terminal evaluation")
        if type(step_id) is not int or step_id < 0:
            raise ValueError("step_id must be a nonnegative integer")
        if not isinstance(raw_action, str) or not isinstance(reasoning, str):
            raise ValueError("raw_action and reasoning must be strings")
        code, parsed_ok = "", True
        parser_error = schema_error = accepted = None
        try:
            code = parse_python_action(raw_action)
        except ValueError as exc:
            parsed_ok, parser_error = False, str(exc)
            observation = f"Action rejected before execution: {parser_error}"
        else:
            reasoning = reasoning or _action_reasoning(code)
            request_start = len(world.requester.requests)
            try:
                observation = world.execute(code)
            except BaseException as error:
                # Upstream lacks finally around its process-wide safety hooks.
                disable = getattr(getattr(world, "safety_guard", None), "disable", None)
                if callable(disable):
                    try:
                        disable()
                    except BaseException:
                        if callable(getattr(error, "add_note", None)):
                            error.add_note("AppWorld safety-hook cleanup also failed.")
                raise
            if not isinstance(observation, str):
                raise TypeError("AppWorld.execute must return a text observation")
            # Official 0.1.3.post1 text protocol; exceptions escaping execute
            # denote infrastructure faults, not failed model actions.
            accepted = not observation.startswith("Execution failed. Traceback:")
            if not accepted and _SCHEMA_ERROR.search(observation):
                schema_error = observation
        # Read only routing information from the official request log. Inspecting
        # executed requests handles aliases and conditionals without guessing
        # from source text, and does not expose request data or evaluator state.
        completion_requested = parsed_ok and any(
            request["method"] == "post" and request["url"] == "/supervisor/message"
            for request in world.requester.requests[request_start:]
        )
        return agent_step_from_parts(
            step_id=step_id, reasoning=reasoning, raw_action=raw_action,
            tool_name="execute_python", tool_args={"code": code}, observation=observation,
            parsed_ok=parsed_ok, schema_valid=parsed_ok and schema_error is None,
            accepted_by_environment=accepted, parser_error=parser_error, schema_error=schema_error,
            metadata={"environment": "appworld", "task_id": self.task_id,
                      "executed": parsed_ok, "task_completed": self.task_completed(),
                      "completion_requested": completion_requested},
        )

    def evaluate(self) -> dict[str, Any]:
        """End execution and return only official aggregate test statistics."""
        world = self._require_world()
        self._finished = True
        if self._metrics is None:
            tracker = world.evaluate()
            self._metrics = {
                "success": bool(tracker.success), "pass_count": int(tracker.pass_count),
                "fail_count": int(tracker.fail_count), "num_tests": int(tracker.num_tests),
                "pass_percentage": float(tracker.pass_percentage),
            }
        return dict(self._metrics)

    def _restore_host_globals(self) -> None:
        # Upstream close does not restore these SafetyGuard mutations.
        if self._host_state is not None:
            (system_exit, fault_enabled), self._host_state = self._host_state, None
            builtins.SystemExit = system_exit
            if faulthandler.is_enabled() != fault_enabled:
                faulthandler.enable() if fault_enabled else faulthandler.disable()

    def close(self) -> None:
        try:
            if self._world is not None:
                world, self._world = self._world, None
                world.close()
        finally:
            self._restore_host_globals()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
