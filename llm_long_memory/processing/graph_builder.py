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
        pattern = re.compile(r"\b" + re.escape(action) + r"\b", flags=re.IGNORECASE)
        if pattern.search(lowered):
            return action
    return fallback_action


def _extract_tail_phrase(sentence: str, action: str) -> str:
    match = re.search(r"\b" + re.escape(action) + r"\b", sentence, flags=re.IGNORECASE)
    if not match:
        return ""
    return sentence[match.end() :].strip(" .,:;!?")


def _normalize_sentence(sentence: str) -> str:
    cleaned = str(sentence).strip()
    # Remove common role tags that pollute extracted objects.
    cleaned = re.sub(r"^\((user|assistant|system)\)\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned


def _truncate_phrase(text: str, max_tokens: int) -> str:
    tokens = [t for t in str(text).split() if t]
    if not tokens:
        return ""
    return " ".join(tokens[:max_tokens]).strip(" .,:;!?")


def _is_instruction_sentence(sentence: str) -> bool:
    lowered = sentence.lower().strip()
    if not lowered:
        return True
    instruction_prefixes = (
        "answer with only",
        "no explanation",
        "respond with only",
        "just answer",
    )
    if lowered.startswith(instruction_prefixes):
        return True
    # Bullet-heavy recommendation lists usually add noise to fact graph.
    if lowered.startswith("* ") or lowered.startswith("- "):
        return True
    return False


def _has_time_marker(text: str) -> bool:
    lowered = str(text).lower()
    if re.search(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", lowered):
        return True
    if re.search(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b", lowered):
        return True
    if re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b", lowered):
        return True
    if re.search(r"\b\d+\s*(?:day|days|week|weeks|month|months|year|years|hour|hours|minute|minutes)\b", lowered):
        return True
    return False


def _has_location_marker(text: str) -> bool:
    lowered = str(text).lower()
    if re.search(r"\b(?:in|at|near|from|to)\s+[A-Z][a-zA-Z]+", str(text)):
        return True
    return any(k in lowered for k in ("city", "country", "town", "village", "office", "school"))


def _infer_fact_type(action: str, has_time: bool, has_location: bool) -> str:
    norm_action = str(action).strip().lower()
    if has_time:
        return "time"
    if has_location:
        return "location"
    if norm_action in {"is", "are", "was", "were", "be", "has", "have"}:
        return "attribute"
    return "event"


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
    root = None

    for token in doc:
        if token.dep_ == "ROOT" and root is None:
            root = token
        if token.dep_ in {"nsubj", "nsubjpass"} and not subject:
            subject = str(token.text).strip()
        if token.dep_ == "ROOT" and not action:
            lemma = str(token.lemma_ or token.text).strip().lower()
            action = lemma if lemma else str(token.text).strip().lower()
        if token.dep_ in {"dobj", "attr", "pobj", "obj", "oprd"} and not obj:
            obj = str(token.text).strip()

    if root is not None:
        if not subject:
            for child in root.children:
                if child.dep_ in {"nsubj", "nsubjpass"}:
                    subject = str(child.text).strip()
                    break
        if not obj:
            for child in root.children:
                if child.dep_ in {"dobj", "attr", "pobj", "obj", "oprd"}:
                    obj = str(child.subtree).strip()
                    break

    if not obj:
        for chunk in doc.noun_chunks:
            chunk_text = str(chunk.text).strip()
            if not chunk_text:
                continue
            if subject and chunk_text.lower() == subject.lower():
                continue
            obj = chunk_text
            break

    if not action:
        return None
    if not subject:
        subject = role

    # If both subject and object are missing semantic content, skip early.
    if not subject and not obj:
        return None

    obj = _truncate_phrase(obj, max_tokens=14)
    subject = _truncate_phrase(subject, max_tokens=6)
    if not obj:
        return None

    confidence = 0.55
    if subject and subject.lower() not in {"user", "assistant", "system"}:
        confidence += 0.15
    if obj and len(obj.split()) >= 2:
        confidence += 0.10
    if any(ent.label_ in {"DATE", "TIME", "GPE", "LOC", "ORG", "PERSON"} for ent in doc.ents):
        confidence += 0.10
    confidence = min(0.95, max(0.40, confidence))
    has_time = _has_time_marker(sentence) or bool(session_date)
    has_location = _has_location_marker(sentence)
    entity_count = sum(
        1 for ent in doc.ents if ent.label_ in {"DATE", "TIME", "GPE", "LOC", "ORG", "PERSON"}
    )
    fact_type = _infer_fact_type(action=action, has_time=has_time, has_location=has_location)

    return {
        "subject": subject,
        "action": action,
        "object": obj,
        "location": "",
        "time": session_date,
        "sentence": sentence,
        "role": role,
        "session_id": session_id,
        "confidence": f"{confidence:.4f}",
        "fact_type": fact_type,
        "entity_count": str(entity_count),
        "has_time": "1" if has_time else "0",
        "has_location": "1" if has_location else "0",
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
        normalized = _normalize_sentence(sentence)
        clipped = normalized[:max_chars]
        if _is_instruction_sentence(clipped):
            continue
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
        obj = _truncate_phrase(_extract_tail_phrase(clipped, action), max_tokens=14)
        if not obj:
            keys = _keywords(clipped, stopwords)
            obj = " ".join(keys[:8])
        if not obj:
            continue

        content_keywords = _keywords(clipped, stopwords)
        lexical_density = float(len(content_keywords)) / float(max(1, len(clipped.split())))
        role_boost = 1.0 if role == "user" else 0.85
        has_time = _has_time_marker(clipped) or bool(session_date)
        has_location = _has_location_marker(clipped)
        fact_type = _infer_fact_type(action=action, has_time=has_time, has_location=has_location)
        entity_count = 0
        if subject and subject != role:
            entity_count += 1
        if has_time:
            entity_count += 1
        if has_location:
            entity_count += 1
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
                "fact_type": fact_type,
                "entity_count": str(entity_count),
                "has_time": "1" if has_time else "0",
                "has_location": "1" if has_location else "0",
            }
        )

    return events
