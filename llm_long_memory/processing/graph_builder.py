"""Lightweight event extraction for long-memory ingestion."""

from __future__ import annotations

import re
from typing import Any, Dict, List


Event = Dict[str, str]
_SPACY_NLP_CACHE: Dict[str, Any] = {}


def _normalize_token(token: str, stopwords: set[str]) -> str:
    cleaned = "".join(ch for ch in token.lower().strip() if ch.isalnum())
    if not cleaned or cleaned in stopwords:
        return ""
    return cleaned


def _keywords(text: str, stopwords: set[str]) -> List[str]:
    return [tok for tok in (_normalize_token(x, stopwords) for x in text.split()) if tok]


def _extract_first_capitalized_phrase(sentence: str, max_tokens: int) -> str:
    tokens = sentence.strip().split()
    phrase: List[str] = []
    for tok in tokens:
        normalized = tok.strip(".,!?;:()[]{}")
        if normalized[:1].isupper() and normalized.lower() not in {"i", "the", "a", "an"}:
            phrase.append(normalized)
            if len(phrase) >= max_tokens:
                break
        elif phrase:
            break
    return " ".join(phrase).strip()


def _extract_action(sentence: str, actions: List[str], fallback_action: str) -> str:
    lowered = sentence.lower()
    actions_sorted = sorted((a for a in actions if a), key=len, reverse=True)
    for action in actions_sorted:
        if action in lowered:
            return action
    return fallback_action


def _extract_tail_phrase(sentence: str, action: str) -> str:
    lowered = sentence.lower()
    pos = lowered.find(action)
    if pos < 0:
        return ""
    return sentence[pos + len(action) :].strip(" .,:;!?")


def _extract_event_spacy(
    sentence: str,
    role: str,
    session_id: str,
    session_date: str,
    event_cfg: Dict[str, Any],
) -> Event | None:
    model_name = str(event_cfg.get("spacy_model", "en_core_web_sm")).strip()
    if not model_name:
        return None

    nlp = _SPACY_NLP_CACHE.get(model_name)
    if nlp is None:
        try:
            import spacy  # type: ignore
        except (ImportError, ModuleNotFoundError):
            _SPACY_NLP_CACHE[model_name] = False
            return None
        try:
            nlp = spacy.load(model_name)
            _SPACY_NLP_CACHE[model_name] = nlp
        except OSError:
            _SPACY_NLP_CACHE[model_name] = False
            return None

    if nlp is False:
        return None

    doc = nlp(sentence)
    subject = ""
    action = ""
    obj = ""

    for token in doc:
        if token.dep_ in {"nsubj", "nsubjpass"} and not subject:
            subject = str(token.text).strip()
        if token.dep_ == "ROOT" and not action:
            action = str(token.lemma_ or token.text).strip().lower()
        if token.dep_ in {"dobj", "attr", "pobj", "obj"} and not obj:
            obj = str(token.text).strip()

    if not action:
        return None
    if not subject:
        subject = role

    # If both subject and object are missing semantic content, skip early.
    if not subject and not obj:
        return None

    return {
        "subject": subject,
        "action": action,
        "object": obj,
        "location": "",
        "time": session_date,
        "sentence": sentence,
        "role": role,
        "session_id": session_id,
        "confidence": "0.8000",
    }


def _is_sentence_noise(sentence: str, noise_prefixes: List[str]) -> bool:
    lowered = sentence.strip().lower()
    if not lowered:
        return True
    for p in noise_prefixes:
        prefix = str(p).strip().lower()
        if prefix and lowered.startswith(prefix):
            return True
    return False


def extract_events_from_message(message: Dict[str, Any], event_cfg: Dict[str, Any]) -> List[Event]:
    """Extract event tuples from one message with conservative filtering."""
    content = str(message.get("content", "")).strip()
    if not content:
        return []

    sentence_split_regex = str(event_cfg.get("sentence_split_regex", r"(?<=[.!?])\\s+"))
    min_chars = int(event_cfg.get("min_sentence_chars", 12))
    max_chars = int(event_cfg.get("max_sentence_chars", 400))
    fallback_action = str(event_cfg.get("fallback_action", "states")).strip().lower() or "states"
    actions = [str(x).strip().lower() for x in list(event_cfg.get("action_verbs", []))]
    stopwords = {str(x).strip().lower() for x in list(event_cfg.get("stopwords", []))}
    noise_prefixes = [str(x) for x in list(event_cfg.get("noise_prefixes", []))]

    role = str(message.get("role", "user")).strip().lower() or "user"
    session_id = str(message.get("session_id", "")).strip()
    session_date = str(message.get("session_date", "")).strip()
    backend = str(event_cfg.get("extractor_backend", "heuristic")).strip().lower()

    sentences = [x.strip() for x in re.split(sentence_split_regex, content) if x.strip()]
    events: List[Event] = []

    for sentence in sentences:
        if len(sentence) < min_chars:
            continue
        clipped = sentence[:max_chars]
        if _is_sentence_noise(clipped, noise_prefixes):
            continue

        if backend in {"spacy", "spacy_then_heuristic"}:
            spacy_event = _extract_event_spacy(
                sentence=clipped,
                role=role,
                session_id=session_id,
                session_date=session_date,
                event_cfg=event_cfg,
            )
            if spacy_event is not None:
                events.append(spacy_event)
                continue
            if backend == "spacy":
                continue

        subject = _extract_first_capitalized_phrase(clipped, max_tokens=4)
        if not subject:
            subject = role

        action = _extract_action(clipped, actions, fallback_action)
        obj = _extract_tail_phrase(clipped, action)
        if not obj:
            keys = _keywords(clipped, stopwords)
            obj = " ".join(keys[:6])
        if not obj:
            continue

        content_keywords = _keywords(clipped, stopwords)
        lexical_density = float(len(content_keywords)) / float(max(1, len(clipped.split())))
        role_boost = 1.0 if role == "user" else 0.85
        confidence = min(
            1.0,
            max(
                0.0,
                0.20
                + (0.25 if subject and subject != role else 0.0)
                + (0.25 if action != fallback_action else 0.0)
                + (0.20 if obj else 0.0)
                + (0.10 if lexical_density >= 0.45 else 0.0),
            )
            * role_boost,
        )

        events.append(
            {
                "subject": subject,
                "action": action,
                "object": obj,
                "location": "",
                "time": session_date,
                "sentence": clipped,
                "role": role,
                "session_id": session_id,
                "confidence": f"{confidence:.4f}",
            }
        )

    return events
