![Sentry Logo](assets/SENTRY.png)

**An external runtime failure management layer for LLM agents.**

Detect execution failures, guide recovery, and learn reusable lessons from verified recoveries.  
Sentry runs alongside your agent loop without replacing the agent, environment, or evaluator.

**[Quick Start](#quick-start)**  |  **[Paper (arXiv)](https://arxiv.org/abs/2610.02994)**  |  **[License](LICENSE)**

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white) [![MIT License](https://img.shields.io/badge/License-MIT-2ea44f)](LICENSE)

## What is Sentry?

Sentry is an external runtime failure management layer for LLM agents. It helps agents fix invalid actions and recover when they get stuck in loops, drift from the task, or make unsupported assumptions. Unlike typical runtime monitors with fixed advice, Sentry learns and evolves from recoveries that work. By detecting failures, guiding recovery, and checking that the repair worked, Sentry helps agents become more reliable across tasks without cluttering their context.

## Sentry's demo

![Sentry demo](assets/sentry-demo.gif)

## Key Results

- **37% average improvement** over the strongest runtime-intervention baseline across WebShop, AppWorld, SWE-bench Lite, and Mind2Web Replay.
- **39% average improvement** over the strongest context-evolution baseline on held-out WebShop and Mind2Web Replay tasks.
- **1.54× base-agent token usage**, versus up to 2.44× for other runtime methods, while saving 31.7s per task on SWE-bench Lite and 45.6s on AppWorld.
- **81.7% recovery rate** across 939 detected failures.


## Quick Start



### Installation

Sentry requires Python 3.10 or later.

```bash
git clone https://github.com/nuglifeleoji/Sentry.git
cd Sentry
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
sentry-validate configs/paper.yaml
```



### Integrate Sentry into Your Agent Loop

Configure a model provider, then submit each completed agent-environment cycle to Sentry:

```python
from Sentry import SentryRunner, agent_step_from_parts

runner = SentryRunner.from_yaml("configs/providers/openai_compatible.yaml")
runner.start_task(
    objective=task_objective,
    action_schema=environment_action_schema,
)

try:
    step = agent_step_from_parts(
        step_id=0,
        reasoning=reasoning,
        raw_action=raw_action,
        tool_name=tool_name,
        tool_args=tool_args,
        observation=observation,
        parsed_ok=True,
        schema_valid=True,
        accepted_by_environment=True,
    )

    repair = runner.step(step, terminal=False)
    if repair is not None:
        agent_messages.append(repair.prompt_text)
finally:
    runner.finalize()
```

Your application remains responsible for generating, validating, and executing actions. See the integration guide for the full lifecycle and validity rules.

### Run the AppWorld Integration

AppWorld is the included reference integration:

```bash
python3.11 -m venv .venv-appworld
source .venv-appworld/bin/activate
python -m pip install -e '.[appworld]'
appworld install
export APPWORLD_ROOT="$PWD/outputs/appworld-root"
appworld download data
export OPENROUTER_API_KEY="your-api-key"
sentry run appworld --config configs/providers/openrouter.yaml
```

See the AppWorld guide for environment setup, credentials, and run limits.

## How It Works

![Sentry framework: failure detection, repair, verification, and playbook learning](assets/sentry-framework.png)

1. **Failure Detection:** Sentry monitors recent agent steps for invalid actions and behavioral failures.
2. **Hard Repair:** Sentry asks the agent to retry invalid actions in the required format.
3. **Soft Repair:** Progress and reasoning failures receive targeted guidance from relevant playbook lessons.
4. **Recovery Verification:** Sentry checks whether the agent recovers over the following steps.
5. **Online Playbook Learning:** Verified soft recoveries become reusable lessons for similar failures across tasks.



## Further Findings

- **Unconditional exposure can hurt:** Controlled experiments show that keeping all failure-specific knowledge in the agent's context lowers performance. This motivates Sentry to expose recovery lessons only when a matching failure occurs.
- **Retrieval must match the failure:** Unfiltered retrieval and broad failure-type matching both perform worse than retrieval with fine-grained failure labels, motivating Sentry's label-based retrieval.
- **Verified lessons transfer:** Lessons learned from verified recoveries improve performance on unseen tasks, and continued learning adds further gains.
- **Task-level and failure-level learning are complementary:** Combining Sentry with context evolution achieves the best results, showing that the two forms of learning address different needs and add on to each other.



## Release

This repository currently contains the project overview and paper. We are finalizing our code and documentation and plan to release the full version in 2-3 weeks.
