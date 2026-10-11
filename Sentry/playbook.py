from __future__ import annotations

import copy
import json
import os
import tempfile
import uuid
from pathlib import Path

from ._structured import json_object
from .config import PlaybookConfig, RetrievalConfig
from .models import PlaybookInsight
from .taxonomy import FailureType, normalize_retrieval_label


SOFT_FAILURE_TYPES = {
    FailureType.PROGRESS_FAILURE,
    FailureType.REASONING_GROUNDING_FAILURE,
}


class PlaybookManager:
    """Append-only playbook with Appendix A.6.2 deterministic retrieval."""

    def __init__(
        self,
        config: PlaybookConfig | None = None,
        retrieval: RetrievalConfig | None = None,
    ):
        self.config = config or PlaybookConfig()
        self.retrieval_config = retrieval or RetrievalConfig()
        self._entries: list[PlaybookInsight] = []
        self._insertion_step = 0

    @property
    def entries(self) -> list[PlaybookInsight]:
        """Snapshots of the validated memory; callers cannot mutate stored records."""
        return copy.deepcopy(self._entries)

    def retrieve(
        self,
        failure_type: FailureType,
        labels: list[str] | tuple[str, ...],
    ) -> list[PlaybookInsight]:
        if failure_type not in SOFT_FAILURE_TYPES:
            return []
        query = {
            normalized
            for item in labels
            if (normalized := normalize_retrieval_label(item, failure_type)) is not None
        }
        same_type = [entry for entry in self._entries if entry.failure_type == failure_type]
        matching = [entry for entry in same_type if query.intersection(entry.retrieval_labels)]
        if matching:
            ranked = sorted(
                matching,
                key=lambda entry: (
                    len(query.intersection(entry.retrieval_labels)),
                    entry.insertion_step,
                ),
                reverse=True,
            )
            if len(ranked) <= self.retrieval_config.top_k:
                return copy.deepcopy(ranked)
            return copy.deepcopy(_diverse_top_k(
                ranked,
                self.retrieval_config.top_k,
                self.retrieval_config.primary_label_cap,
            ))

        # Appendix A.6.2: no overlap falls back to recency within the type,
        # including entries bearing different fine-grained labels.
        return copy.deepcopy(sorted(same_type, key=lambda entry: entry.insertion_step, reverse=True)[
            : self.retrieval_config.top_k
        ])

    def add(self, insight: PlaybookInsight) -> PlaybookInsight:
        if insight.failure_type not in SOFT_FAILURE_TYPES:
            raise ValueError("Only verified soft-repair episodes enter the playbook")
        next_step = self._insertion_step + 1
        data = insight.to_json()
        data["insertion_step"] = next_step
        data["insight_id"] = insight.insight_id or f"sentry-{next_step:05d}-{uuid.uuid4().hex[:8]}"
        validated = PlaybookInsight.from_json(data)
        if any(entry.insight_id == validated.insight_id for entry in self._entries):
            raise ValueError(f"Duplicate playbook insight_id: {validated.insight_id}")
        self._insertion_step = next_step
        self._entries.append(validated)
        try:
            self.persist()
        except Exception:
            self._entries.pop()
            self._insertion_step = next_step - 1
            raise
        # Preserve insertion metadata on the submitted object for callers, while
        # retaining an independently validated record in the append-only memory.
        insight.insight_id = validated.insight_id
        insight.insertion_step = next_step
        insight.retrieval_labels = list(validated.retrieval_labels)
        return copy.deepcopy(validated)

    def clear(self, *, persist: bool = True) -> None:
        previous_entries = self._entries
        previous_step = self._insertion_step
        self._entries = []
        self._insertion_step = 0
        if persist:
            try:
                self.persist()
            except Exception:
                self._entries = previous_entries
                self._insertion_step = previous_step
                raise

    def load(self) -> None:
        """Explicitly restore a saved online-memory snapshot from persist_path."""
        path = self._path()
        if path is None:
            raise ValueError("Loading a playbook requires persist_path")
        # An absent snapshot is an error, not an empty initialization.
        raw = json_object(path.read_text(encoding="utf-8"), {
            "version", "format", "insertion_step", "entries",
        })
        if raw is None or type(raw["version"]) is not int or raw["version"] != 1 or raw["format"] != "sentry_appendix_a6_2":
            raise ValueError("Unsupported or malformed Sentry playbook format")
        if not isinstance(raw["entries"], list):
            raise ValueError("Playbook entries must be an array")
        loaded = [PlaybookInsight.from_json(item) for item in raw["entries"]]
        steps = [entry.insertion_step for entry in loaded]
        identifiers = [entry.insight_id for entry in loaded]
        if any(step <= 0 for step in steps) or steps != sorted(set(steps)):
            raise ValueError("Persisted insertion steps must be positive, unique and ordered")
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Persisted playbook identifiers must be unique")
        counter = raw["insertion_step"]
        if type(counter) is not int or counter != max(steps, default=0):
            raise ValueError("Playbook insertion counter does not match its entries")
        # Validate the whole artifact before replacing the current memory.
        self._entries = loaded
        self._insertion_step = counter

    def persist(self) -> None:
        path = self._path()
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "format": "sentry_appendix_a6_2",
            "insertion_step": self._insertion_step,
            "entries": [entry.to_json() for entry in self._entries],
        }
        # Replace atomically so interruption cannot leave a truncated artifact.
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.", suffix=".tmp", delete=False,
            ) as handle:
                temporary = handle.name
                json.dump(payload, handle, indent=2, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)

    def _path(self) -> Path | None:
        return Path(self.config.persist_path) if self.config.persist_path else None


def _diverse_top_k(
    ranked: list[PlaybookInsight],
    top_k: int,
    primary_label_cap: int,
) -> list[PlaybookInsight]:
    """Greedily apply the primary-label cap, then fill if no alternative exists."""

    selected: list[PlaybookInsight] = []
    deferred: list[PlaybookInsight] = []
    counts: dict[str | None, int] = {}
    for entry in ranked:
        primary = entry.primary_label
        if counts.get(primary, 0) >= primary_label_cap:
            deferred.append(entry)
            continue
        selected.append(entry)
        counts[primary] = counts.get(primary, 0) + 1
        if len(selected) == top_k:
            return selected
    # "Whenever possible": exceeding the cap is allowed only to fill otherwise
    # unused slots when no differently labelled entry remains.
    selected.extend(deferred[: max(0, top_k - len(selected))])
    return selected
