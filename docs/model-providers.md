# Model providers

Sentry uses an LLM for failure detection, recovery verification, and lesson generation. The **provider** and **model ID** determine the service and model used for these operations. These settings are independent of the task agent's model configuration.

## Provider Selection

The configuration template and API key must match the service receiving the model requests.

| Service / API | Template | Credential variable |
| --- | --- | --- |
| OpenAI Chat Completions | [`openai.yaml`](../configs/providers/openai.yaml) | `OPENAI_API_KEY` |
| OpenAI Responses | [`openai_responses.yaml`](../configs/providers/openai_responses.yaml) | `OPENAI_API_KEY` |
| OpenRouter | [`openrouter.yaml`](../configs/providers/openrouter.yaml) | `OPENROUTER_API_KEY` |
| Anthropic | [`anthropic.yaml`](../configs/providers/anthropic.yaml) | `ANTHROPIC_API_KEY` |
| Gemini | [`gemini.yaml`](../configs/providers/gemini.yaml) | `GEMINI_API_KEY` |
| Together | [`together.yaml`](../configs/providers/together.yaml) | `TOGETHER_API_KEY` |
| Custom OpenAI-compatible endpoint | [`openai_compatible.yaml`](../configs/providers/openai_compatible.yaml) | Configured if authentication is required |

## Configuration and Loading

After [installation](../README.md#installation), the selected template is copied to an editable configuration file. For OpenAI, the following commands run from the repository root:

```bash
mkdir -p outputs
cp configs/providers/openai.yaml outputs/sentry.yaml
```

In `outputs/sentry.yaml`, `api.model` specifies the exact model ID available to the provider account. The other template settings can remain unchanged initially. `api_key_env` names the environment variable containing the API key.

The credential is exported in the same terminal used to run the application. `API_KEY_PLACEHOLDER` represents the actual provider key:

```bash
export OPENAI_API_KEY="API_KEY_PLACEHOLDER"
sentry-validate outputs/sentry.yaml
```

Validation checks configuration settings offline, not connectivity. Sentry does not load `.env` files automatically.

The application loads the configuration with:

```python
from Sentry import SentryRunner

runner = SentryRunner.from_yaml("outputs/sentry.yaml")
```

For a custom endpoint, `api.base_url` specifies the running server's API address. `api_key_env` names the credential variable when authentication is required; otherwise it remains `null`.

[Agent integration](agent-integration.md) explains how Sentry connects to the task loop. [Configuration](configuration.md) lists the decoding and output settings required of the model.
