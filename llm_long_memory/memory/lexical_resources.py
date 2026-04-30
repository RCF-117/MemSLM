"""Shared lexical constants for lightweight query and evidence heuristics."""

from __future__ import annotations


BASIC_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "are",
        "did",
        "do",
        "for",
        "i",
        "in",
        "is",
        "me",
        "my",
        "of",
        "on",
        "the",
        "to",
        "with",
        "you",
    }
)

NAMED_TOKEN_STOPWORDS = frozenset(
    {
        "A",
        "An",
        "And",
        "Any",
        "Can",
        "Could",
        "Do",
        "Does",
        "I",
        "If",
        "In",
        "My",
        "Of",
        "Or",
        "The",
        "This",
        "What",
        "When",
        "Where",
        "Which",
        "Who",
        "Would",
        "You",
        "Your",
    }
)

UPDATE_CUES = frozenset(
    {
        "changed",
        "currently",
        "latest",
        "moved",
        "now",
        "recently",
        "set",
        "switched",
        "updated",
    }
)
