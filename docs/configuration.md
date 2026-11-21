# Configuration

Sentry reads its settings from the YAML file passed to `SentryRunner.from_yaml()`, such as `outputs/sentry.yaml`. This file specifies the model connection and optional storage for lessons and logs.

A copy of a provider template, such as [`configs/providers/openai.yaml`](../configs/providers/openai.yaml), serves as the application's configuration at `outputs/sentry.yaml`. [Model providers](model-providers.md) describes the copying and setup steps.

Sentry automatically fills in default values for omitted settings. [`configs/paper.yaml`](../configs/paper.yaml) shows these defaults for reference and requires no separate copying or loading.

## Configurable Settings

The following fields belong in `outputs/sentry.yaml`. For example, the [OpenAI template](../configs/providers/openai.yaml) groups its connection settings under `api`:

```yaml
api:
  provider: openai
  model: YOUR_DEPLOYMENT_MODEL_ID
  api_key_env: OPENAI_API_KEY
```

Most integrations require a model connection, with optional paths for saving lessons and logs:

| Setting | Purpose |
| --- | --- |
| `provider` under `api` | The API adapter used to reach the model; normally the value from the selected provider template |
| `model` under `api` | The exact model ID served by the provider |
| `base_url` under `api` | The API base address for a custom endpoint; hosted-provider templates supply or use their default address |
| `api_key_env` under `api` | The name of the environment variable containing the API key, such as `OPENAI_API_KEY` |
| `timeout` under `api` | Request timeout in seconds; defaults to `120` |
| `max_retries` under `api` | Retries for transient request failures; defaults to `0`, which the AppWorld runner requires |
| `persist_path` under `playbook` | Optional path for saving lessons; defaults to `null` (memory only) |
| `enabled` under `logging` | Whether file logging is enabled; defaults to `true` |
| `log_path` under `logging` | Optional JSONL log path; defaults to `null` (no file output) |

File paths are relative to the program's working directory. [Playbook and logging](playbook-and-logging.md) covers saving, restoring, and resetting lessons.

Usually, setting `model` under `api` is enough. If `model` is also specified under the top-level `model` section, the two values must match.

## Fixed Algorithm Settings

The current implementation enforces the values below. These fields can be omitted from the provider configuration; changing their values causes validation to fail.

| Setting | Value | Meaning |
| --- | --- | --- |
| `window_size` under `trace` | `5` | Recent cycles available to detection |
| `post_rescue_horizon` under `trace` | `10` | Subsequent cycles required for soft-recovery verification |
| `top_k` under `retrieval` | `5` | Maximum number of retrieved lessons |
| `primary_label_cap` under `retrieval` | `2` | Retrieval diversity cap per primary label when alternatives allow |
| `temperature` under `model` | `0.0` | Temperature for detection, verification, and learning |
| `max_tokens` under `model` | `4096` | Output-token limit for those calls |
| `structured_outputs` under `model` | `true` | Request structured responses for those calls |
| `temperature` under `task_agent` | `0.7` | Decoding temperature for `generate_repair()` |
| `max_tokens` under `task_agent` | `384` | Output-token limit for `generate_repair()` |
| `structured_outputs` under `task_agent` | `false` | Let repair generation follow the environment's action format |

`task_agent` settings control `generate_repair()` and task-model calls in the reference benchmark. An existing agent's model settings remain under the application's control.

## Configuration Validation

```bash
sentry-validate outputs/sentry.yaml
```

This command checks the edited configuration and prints the resolved settings, including defaults, and a hash of the file. Validation is offline: it checks the settings without contacting the provider or testing credentials.
