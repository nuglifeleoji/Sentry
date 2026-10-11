from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from numbers import Real
from pathlib import Path
from typing import Any

import yaml

from .clients import ProviderConfig


class _UniqueKeyLoader(yaml.SafeLoader):
    """Never silently overwrite an experiment setting from a duplicate YAML key."""


def _unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ValueError("Configuration mapping keys must be strings")
        if key in result:
            raise ValueError(f"Duplicate configuration key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def _positive_integer(value: Any, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _decoding(temperature: Any, max_tokens: Any, structured: Any) -> None:
    try:
        valid_temperature = not isinstance(temperature, bool) and isinstance(temperature, Real) and math.isfinite(temperature) and temperature >= 0
    except (OverflowError, ValueError):
        valid_temperature = False
    if not valid_temperature:
        raise ValueError("temperature must be a finite nonnegative number")
    _positive_integer(max_tokens, "max_tokens")
    if type(structured) is not bool:
        raise ValueError("structured_outputs must be a boolean")


@dataclass(frozen=True)
class TraceConfig:
    """Trajectory sizes specified in Sections 3.1 and 3.3."""

    window_size: int = 5
    post_rescue_horizon: int = 10

    def __post_init__(self) -> None:
        _positive_integer(self.window_size, "window_size")
        _positive_integer(self.post_rescue_horizon, "post_rescue_horizon")


@dataclass(frozen=True)
class RetrievalConfig:
    """Deterministic retrieval constants from Appendix A.6.2."""

    top_k: int = 5
    primary_label_cap: int = 2

    def __post_init__(self) -> None:
        _positive_integer(self.top_k, "top_k")
        _positive_integer(self.primary_label_cap, "primary_label_cap")


@dataclass(frozen=True)
class ModelConfig:
    """Shared backbone and auxiliary decoding settings from Section 4.3."""

    model: str = "Qwen3.5-9B"
    temperature: float = 0.0
    max_tokens: int = 4096
    structured_outputs: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip() or self.model != self.model.strip():
            raise ValueError("model must be a nonempty model identifier without surrounding whitespace")
        _decoding(self.temperature, self.max_tokens, self.structured_outputs)


@dataclass(frozen=True)
class TaskAgentConfig:
    """Task-agent continuation and regeneration settings from Section 4.3."""

    temperature: float = 0.7
    max_tokens: int = 384
    structured_outputs: bool = False

    def __post_init__(self) -> None:
        _decoding(self.temperature, self.max_tokens, self.structured_outputs)


@dataclass(frozen=True)
class PlaybookConfig:
    """Optional storage for online memory; a new manager starts empty."""

    persist_path: str | None = None

    def __post_init__(self) -> None:
        if self.persist_path is not None and (not isinstance(self.persist_path, str) or not self.persist_path.strip()):
            raise ValueError("persist_path must be a nonempty path string")


@dataclass(frozen=True)
class LoggingConfig:
    enabled: bool = True
    log_path: str | None = None

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("logging.enabled must be a boolean")
        if self.log_path is not None and (not isinstance(self.log_path, str) or not self.log_path.strip()):
            raise ValueError("log_path must be a nonempty path string")


@dataclass(frozen=True)
class SentryConfig:
    trace: TraceConfig = field(default_factory=TraceConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    task_agent: TaskAgentConfig = field(default_factory=TaskAgentConfig)
    playbook: PlaybookConfig = field(default_factory=PlaybookConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    api: ProviderConfig | None = None

    def __post_init__(self) -> None:
        for name, cls in (("trace", TraceConfig), ("retrieval", RetrievalConfig), ("model", ModelConfig),
                          ("task_agent", TaskAgentConfig), ("playbook", PlaybookConfig), ("logging", LoggingConfig)):
            if not isinstance(getattr(self, name), cls):
                raise ValueError(f"{name} must be a {cls.__name__}")
        if self.api is not None:
            if not isinstance(self.api, ProviderConfig):
                raise ValueError("api must be a ProviderConfig")
            if self.api.model != self.model.model:
                raise ValueError("api.model and model.model must identify the same shared backbone")
        if self.trace.window_size != 5:
            raise ValueError("Full Sentry uses W=5")
        if self.trace.post_rescue_horizon != 10:
            raise ValueError("Full Sentry uses H=10")
        if self.retrieval.top_k != 5:
            raise ValueError("Full Sentry uses top-k=5 retrieval")
        if self.retrieval.primary_label_cap != 2:
            raise ValueError("Appendix A.6.2 caps a primary label at two whenever possible")
        if self.model.temperature != 0.0:
            raise ValueError("Full Sentry uses deterministic decoding (temperature=0)")
        if self.model.max_tokens != 4096:
            raise ValueError("Full Sentry uses a maximum generation length of 4096")
        if self.model.structured_outputs is not True:
            raise ValueError("Full Sentry requires structured auxiliary outputs")
        if self.task_agent.temperature != 0.7 or self.task_agent.max_tokens != 384:
            raise ValueError("The paper task agent uses temperature=0.7 and max_tokens=384")
        if self.task_agent.structured_outputs is not False:
            raise ValueError("Task-agent repairs follow the environment action format")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SentryConfig":
        return cls.from_yaml_text(Path(path).read_text(encoding="utf-8"))

    @classmethod
    def from_yaml_text(cls, source: str) -> "SentryConfig":
        """Parse a single snapshot, also used for hashed configuration manifests."""
        if not isinstance(source, str):
            raise ValueError("YAML configuration source must be text")
        try:
            raw = yaml.load(source, Loader=_UniqueKeyLoader)
        except yaml.YAMLError as exc:
            raise ValueError("Invalid Sentry configuration YAML") from exc
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError("Expected mapping YAML for Sentry config")
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "SentryConfig":
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError("Expected a mapping for Sentry configuration")
        allowed = {"trace", "retrieval", "model", "task_agent", "playbook", "logging", "api"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError("Unsupported Sentry configuration sections: " + ", ".join(sorted(map(str, unknown))))
        model_raw = raw.get("model")
        api_raw = raw.get("api")
        # A model identifier may be declared once. Two declarations must agree.
        if isinstance(api_raw, dict) and isinstance(model_raw, (dict, type(None))):
            model_raw = dict(model_raw or {})
            api_raw = dict(api_raw)
            if "model" in api_raw and "model" not in model_raw:
                model_raw["model"] = api_raw["model"]
            elif "model" in model_raw and "model" not in api_raw:
                api_raw["model"] = model_raw["model"]
        return cls(
            trace=_dataclass_from_dict(TraceConfig, raw.get("trace")),
            retrieval=_dataclass_from_dict(RetrievalConfig, raw.get("retrieval")),
            model=_dataclass_from_dict(ModelConfig, model_raw),
            task_agent=_dataclass_from_dict(TaskAgentConfig, raw.get("task_agent")),
            playbook=_dataclass_from_dict(PlaybookConfig, raw.get("playbook")),
            logging=_dataclass_from_dict(LoggingConfig, raw.get("logging")),
            api=None if api_raw is None else _dataclass_from_dict(ProviderConfig, api_raw),
        )


def _dataclass_from_dict(cls, raw: Any):
    if raw is None:
        return cls()
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a mapping for {cls.__name__}")
    allowed = {item.name for item in fields(cls) if item.init}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unsupported {cls.__name__} fields: " + ", ".join(sorted(map(str, unknown))))
    return cls(**raw)
