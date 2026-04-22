"""Small query-intent helpers shared by active retrieval and tests."""

from __future__ import annotations

from typing import Dict


_COMPARE_CUES = (
    "compare",
    "versus",
    "vs",
    "before",
    "after",
    "earlier",
    "later",
    "which came first",
    "between",
    "earliest",
    "latest",
    "more recent",
)

_PREFERENCE_CUES = (
    "prefer",
    "preference",
    "recommend",
    "suggest",
    "resources",
    "resource",
    "how should i",
    "what should i",
    "what resource",
    "what resources",
    "advice",
    "tip",
    "tips",
    "guidance",
    "choose",
    "choice",
)


def extract_query_intent(query: str) -> Dict[str, bool]:
    """Infer coarse intent flags from the query text only.

    This helper is intentionally lightweight and dataset-agnostic. It uses
    generic language cues so tests and graph/tool routing do not depend on any
    dataset-provided question type labels.
    """

    lowered = str(query or "").strip().lower()
    return {
        "asks_where": "where" in lowered or "location" in lowered,
        "asks_when": any(
            token in lowered for token in ("when", "date", "time", "year", "month", "day")
        ),
        "asks_how_many": any(
            token in lowered for token in ("how many", "number of", "count", "total")
        ),
        "asks_current": any(
            token in lowered for token in ("current", "currently", "latest", "now")
        ),
        "asks_compare": any(token in lowered for token in _COMPARE_CUES),
        "asks_preference": any(token in lowered for token in _PREFERENCE_CUES),
    }
