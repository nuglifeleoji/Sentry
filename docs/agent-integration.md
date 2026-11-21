# Agent integration

Sentry receives the action, validation results, and observation from each execution cycle. It checks whether intervention is needed and returns guidance for the next model request.

The integration requires an [installed Sentry package](../README.md#installation), a [provider configuration](model-providers.md) at `outputs/sentry.yaml`, and the corresponding credentials.

## Integration Steps

1. **Task initialization.** Before execution, `SentryRunner.from_yaml()` loads the model configuration. `start_task()` receives the task objective and required action format as strings.
2. **Step submission.** After each attempted action, the application's parser, tool checks, and execution supply the fields in `agent_step_from_parts()`. Invalid actions are reported too, with the validation error as feedback, without being executed. `runner.step()` submits this record to Sentry.
3. **Repair feedback.** If `runner.step()` returns a repair, the next model request includes `repair.prompt_text` as `guidance` alongside the task history. The prompt stays outside the stored history to avoid accumulating repeated copies. If `runner.step()` returns `None`, execution continues normally. Subsequent cycles are also submitted so Sentry can verify recovery.
4. **Task finalization.** A cycle that ends execution is submitted with `terminal=True`, which suppresses new interventions while allowing pending verification to observe that cycle. It indicates execution ending, not task success. `finalize()` clears task-local state when execution ends, including on exceptions.

## Integration Example

The following code shows the four integration points in an existing task loop. The numbered comments correspond to the steps above. The task objective, action schema, cycle budget, and per-step variables come from the application's existing model, validation, and execution code.

```python
from Sentry import SentryRunner, agent_step_from_parts

# 1. Task initialization.
runner = SentryRunner.from_yaml("outputs/sentry.yaml")
runner.start_task(
    objective=task_objective,
    action_schema=environment_action_schema,
)
guidance = None

try:
    for step_id in range(max_steps):
        # Existing model and tool calls run here. The model receives guidance
        # with task history; validation and execution produce the fields below.
        ...

        # 2. Step submission: values come from this action's actual results.
        step = agent_step_from_parts(
            step_id=step_id,
            reasoning=reasoning,
            raw_action=raw_action,
            observation=observation,
            parsed_ok=parsed_ok,
            schema_valid=schema_valid,
            accepted_by_environment=accepted_by_environment,
            parser_error=parser_error,
            schema_error=schema_error,
        )
        terminal = execution_ended or step_id == max_steps - 1
        repair = runner.step(step, terminal=terminal)

        # 3. Repair feedback: guidance accompanies the next model request.
        guidance = repair.prompt_text if repair is not None else None
        if terminal:
            break
finally:
    # 4. Task finalization: cleanup also runs on errors.
    runner.finalize()
```

Here, `max_steps` is the application's cycle budget, and `execution_ended` records its decision to stop the task. The application includes `guidance` in the next model request and retains its existing conversation history. The fields passed to Sentry are described below.

## Step Inputs

The table below lists all parameters accepted by `agent_step_from_parts()`. Each record describes one attempted action, including actions rejected before execution. Validation and acceptance fields reflect the application's actual results.

| Parameter | Default | Description |
| --- | --- | --- |
| `step_id` | Required | Nonnegative integer identifying the cycle, such as `0` for the first cycle. |
| `reasoning` | Required | The action rationale as a string; `""` when no separate rationale is available. |
| `raw_action` | Required | The complete raw action text produced by the model. |
| `tool_name` | `None` | The parsed tool name, or `None` if unavailable. |
| `tool_args` | `None` | The parsed tool arguments as a dictionary; `None` becomes an empty dictionary. |
| `observation` | `""` | The tool output, environment feedback, or validation error as a string. |
| `parsed_ok` | `True` | `True` if the complete action parsed successfully; `False` for malformed output. |
| `schema_valid` | `True` | `True` if the tool name and arguments passed validation; `False` if validation or parsing failed. |
| `accepted_by_environment` | `None` | `True` if explicitly accepted, `False` if rejected, or `None` if unexecuted or unknown. |
| `parser_error` | `None` | The parsing error message, or `None` when there is no parsing error. |
| `schema_error` | `None` | The tool or argument validation error message, or `None` when there is no validation error. |
| `task_progress_score` | `None` | Optional score retained for application use; Sentry does not use it for detection, recovery verification, or learning. |
| `metadata` | `None` | Additional cycle information as a dictionary; `None` becomes an empty dictionary. |

For example, malformed JSON sets both validity flags to `False` and is not executed. A valid tool request that times out keeps both validity flags `True`; `observation` describes the timeout, and `accepted_by_environment` is `None` if acceptance is unknown. Sentry uses invalid parsing or schema results to trigger hard repair; execution failure alone is not a format error.

## Multiple Tasks and References

Across multiple tasks, reusing one runner and repeating `start_task()` through `finalize()` retains learned lessons. [Playbook and logging](playbook-and-logging.md) describes persistence and learning conditions. The task loop in [`benchmarks/appworld/run.py`](../benchmarks/appworld/run.py) provides a real benchmark integration.
