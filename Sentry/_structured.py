"""Validation shared by the fixed-field auxiliary response schemas.

Malformed output must never create an intervention or a learned recovery.
"""
from __future__ import annotations

import json


def json_object(raw: str, fields: set[str]) -> dict | None:
    if not isinstance(raw, str):
        return None

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate field: {key}")
            result[key] = value
        return result

    try:
        data = json.loads(raw, object_pairs_hook=unique_object)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict) or set(data) != fields:
        return None
    return data


def nonempty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def string_list(value: object, *, allow_empty: bool = True) -> bool:
    return (
        isinstance(value, list)
        and (allow_empty or bool(value))
        and all(nonempty_text(item) for item in value)
    )
