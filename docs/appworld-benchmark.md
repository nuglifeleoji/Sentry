# AppWorld benchmark

The [reference runner](../benchmarks/appworld/run.py) uses the official [AppWorld environment and evaluator](https://github.com/StonyBrookNLP/appworld/blob/main/README.md). It currently supports tasks from AppWorld's development split, named `dev`, which is a predefined subset of the benchmark dataset.

Selected tasks run one at a time. Each task starts with a fresh environment and agent conversation, while lessons learned in the playbook carry over to later tasks in the same run.

All commands below run from the repository root.

## Installation and Configuration

The setup below uses Python 3.11 with the pinned AppWorld dependency (`0.1.3.post1`). The commands prepare the environment and data; `API_KEY_PLACEHOLDER` represents the actual OpenRouter key:

```bash
python3.11 -m venv .venv-appworld
source .venv-appworld/bin/activate
python -m pip install -e '.[appworld]'
appworld install
export APPWORLD_ROOT="$PWD/outputs/appworld-root"
appworld download data --root "$APPWORLD_ROOT"
export OPENROUTER_API_KEY="API_KEY_PLACEHOLDER"
```

This example uses the [OpenRouter template](../configs/providers/openrouter.yaml). Its model ID and settings must match the selected deployment. Other provider configurations, prepared as described in [Model providers](model-providers.md), can be supplied with `--config`.

The runner requires an `api` configuration with `max_retries: 0`. The fields `playbook.persist_path` and `logging.log_path` must remain unset; the runner manages those artifacts under `--output-dir`.

## Task Execution

The following command lists development-set task IDs without model requests or credentials:

```bash
python -m benchmarks.appworld.run --list-tasks
```

The following command runs the first available development-set task:

```bash
python -m benchmarks.appworld.run \
  --config configs/providers/openrouter.yaml \
  --max-steps 30 \
  --max-model-calls 64
```

The option `--task-id TASK_ID` selects a specific task, where `TASK_ID` is an ID from the list. Repeated `--task-id` options select distinct tasks in execution order. Without this option, the runner executes one task, not the whole split.

| Option | Default | Meaning |
| --- | --- | --- |
| `--max-steps` | `30` | Maximum cycles per task |
| `--max-model-calls` | `64` | All model calls across the run, including Sentry calls |
| `--max-cost-usd` | Unset | Run-wide reported-cost threshold, checked between calls |
| `--output-dir` | `outputs/appworld` | Parent directory for a unique run directory |
| `--root` | `APPWORLD_ROOT`, or `.` | Downloaded AppWorld data root |

A single call can exceed the cost threshold. With a threshold set, a call without reported cost prevents the next call. The command `python -m benchmarks.appworld.run --help` lists all options.

## Results

The runner prints progress and a final JSON summary. Each run creates these files in a unique directory under `--output-dir`:

| File | Contents |
| --- | --- |
| `run.json` | Settings, task order, evaluator metrics, model calls, usage, errors, and stop reasons |
| `events.jsonl` | Interventions, recovery verification, and learning events |
| `playbook.json` | Lessons learned during the run |

**Task success:** In `run.json`, each entry in the `tasks` list contains a `metrics` object. Its `success` field is `true` when the task passes AppWorld's official evaluation. A stop reason of `task_completed` records the agent's completion declaration and the end of execution; `success` records whether the task actually passed. Evaluation happens after the task ends, and its scores are not passed to Sentry for repair or learning.

**API costs:** `known_cost_usd` adds up the costs returned by the model provider. `missing_cost_calls` counts requests for which the provider returned no cost. If this count is greater than zero, the recorded cost may be lower than the actual total.

## Playbook Across Tasks

Each execution of the runner starts with an empty playbook. When one command selects several tasks, lessons learned from earlier tasks remain available to later tasks. Running the command again starts a new playbook; the runner does not load lessons from previous runs.

The example command above runs one development-set task to check that the model connection, environment, and Sentry work together. The paper reports results across many tasks; this command does not run the full paper experiments.

## Troubleshooting

- **AppWorld cannot be imported:** execution requires the active `.venv-appworld` Python 3.11 environment with `.[appworld]` installed.
- **Data cannot be found:** `APPWORLD_ROOT`, or the explicit `--root` argument, must point to the directory containing the downloaded AppWorld data.
- **A run stops unexpectedly:** `stop_reason` and `error` in `run.json` record whether execution reached a limit or encountered an error. These fields appear at the run level and inside the affected task under `tasks`.
