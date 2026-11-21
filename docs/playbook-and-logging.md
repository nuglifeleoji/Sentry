# Playbook and logging

The playbook stores reusable lessons from verified soft recoveries. Logs record the steps, interventions, verification results, and learning events produced during execution.

## When Sentry Learns

- Hard repair is checked against the next cycle: the corrected action must parse, satisfy the schema, and be explicitly accepted by the environment. It does not create a playbook lesson.
- Soft repair is verified after exactly ten subsequent cycles. Only a verified recovery can produce a new lesson through the updater.
- If the task ends before the soft-recovery window is complete, that recovery is not verified and does not create a lesson.

Reusing the same runner across tasks retains lessons in memory. `start_task()` and `finalize()` reset task-local state while keeping the playbook.

## Lesson Storage and Restoration

The following sections in the provider configuration enable lesson persistence and file logging when the runner is constructed:

```yaml
playbook:
  persist_path: outputs/playbook.json
logging:
  enabled: true
  log_path: outputs/sentry.jsonl
```

Paths are relative to the program's working directory. New lessons are saved automatically when persistence is configured. By default, both paths are unset: memory stays in the process and no log file is written.

A new runner starts with an empty in-memory playbook, even when a snapshot exists. Restoring a saved playbook requires an explicit load before a task starts or new lessons are added:

```python
from Sentry import SentryRunner

runner = SentryRunner.from_yaml("outputs/sentry.yaml")
runner.guard.playbook.load()  # Requires an existing, valid snapshot.
```

An existing runner exposes memory inspection and reset operations:

```python
lessons = runner.guard.playbook.entries
runner.guard.reset_playbook()
```

Resetting also saves the empty playbook when persistence is configured. Preserving an existing snapshot requires loading it before execution; automatic saves do not merge it with new memory.

## Logs and Events

With the YAML settings above, runtime records are written as one JSON object per line to `outputs/sentry.jsonl`. These records show which failures were detected, which repairs were issued, and whether verification and learning completed.

Applications that consume events directly enable capture and drain the buffer regularly, including after finalization:

```python
runner = SentryRunner.from_yaml("outputs/sentry.yaml", capture_runtime_events=True)
# During the application's agent loop:
events = runner.guard.drain_runtime_events()
```

Event capture and file logging are independent. For benchmark runs, the [AppWorld runner's output directory](appworld-benchmark.md#results) contains its managed playbook and events.
