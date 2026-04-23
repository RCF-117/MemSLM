"""LLM-backed fixed-schema claim extraction from filtered evidence packs."""

from __future__ import annotations

import json
import re
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple

from llm_long_memory.llm.ollama_client import ollama_generate_with_retry
from llm_long_memory.memory.model_output_json_utils import (
    extract_first_json_block,
    safe_json_loads_relaxed,
)


class EvidenceGraphExtractor:
    """Extract fixed-schema claims from filtered evidence with an 8B model."""

    _CONTENT_TOKEN_RE = re.compile(r"[a-z0-9]+")
    _NAMED_TOKEN_RE = re.compile(r"\b[A-Z][A-Za-z0-9'/-]+\b")
    _CONTENT_STOPWORDS = {
        "a",
        "an",
        "and",
        "any",
        "about",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "for",
        "from",
        "get",
        "good",
        "have",
        "help",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "learn",
        "me",
        "more",
        "my",
        "of",
        "on",
        "or",
        "our",
        "should",
        "some",
        "stay",
        "suggest",
        "suggestions",
        "than",
        "that",
        "the",
        "their",
        "them",
        "these",
        "this",
        "to",
        "use",
        "using",
        "video",
        "ways",
        "what",
        "where",
        "with",
        "work",
        "working",
        "would",
        "you",
        "your",
    }
    _NAMED_TOKEN_STOPWORDS = {
        "A",
        "An",
        "And",
        "Any",
        "Can",
        "Could",
        "Do",
        "Does",
        "For",
        "I",
        "I'd",
        "I've",
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
    _PREFERENCE_QUERY_RE = re.compile(
        r"\b("
        r"recommend|suggest(?:ion|ions)?|advice|tips?|"
        r"what should i|which .* should i|"
        r"ways to|resources? where i can learn|learn more about|"
        r"any suggestions|ideas? for"
        r")\b"
    )
    _PREFERENCE_SIGNAL_RE = re.compile(
        r"\b("
        r"recommend|suggest|resource|course|tutorial|guide|recipe|activity|"
        r"option|idea|learn|serve|schedule|discussion|feedback|break"
        r")\b"
    )
    _PREFERENCE_DIRECTION_PREDICATES = {
        "recommend",
        "recommendation",
        "suggestion",
        "name",
        "serve",
        "resource",
        "resources",
        "schedule",
    }
    _GENERIC_VALUE_HINTS = {
        "",
        "resource",
        "resources",
        "recipe",
        "recipes",
        "option",
        "options",
        "activity",
        "activities",
        "suggestion",
        "suggestions",
        "idea",
        "ideas",
    }
    _META_ADVICE_RE = re.compile(
        r"\b("
        r"i(?:'m| am)? happy to help|i(?:'d| would) be happy to help|"
        r"once i have a better understanding|"
        r"let me ask you|let me know|"
        r"before i give you|"
        r"better understanding of your needs|"
        r"narrow down the options|"
        r"what's your budget|"
        r"what type of|"
        r"a few questions"
        r")\b"
    )
    _GENERIC_PREFERENCE_PHRASES = {
        "advice",
        "generic advice",
        "personalized recommendation",
        "personalized recommendations",
        "recommendation",
        "recommendations",
        "suggestion",
        "suggestions",
        "tips",
    }
    _META_PREDICATES = {
        "asking",
        "can give",
    }

    def __init__(self, manager: Any, cfg: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(cfg or {})
        self.manager = manager
        self.enabled = bool(cfg.get("claims_enabled", cfg.get("enabled", False)))
        self.model = str(cfg.get("model", manager.config["llm"]["default_model"])).strip()
        self.temperature = float(cfg.get("temperature", 0.0))
        self.timeout_sec = int(cfg.get("timeout_sec", 120))
        self.max_output_tokens = int(cfg.get("max_output_tokens", 384))
        self.batch_size = max(1, int(cfg.get("extractor_batch_size", 8)))
        self.max_claims_per_batch = max(1, int(cfg.get("extractor_max_claims_per_batch", 12)))
        self.max_support_units_per_batch = max(
            1, int(cfg.get("extractor_max_support_units_per_batch", 12))
        )
        self.force_json_output = bool(cfg.get("force_json_output", True))
        self.think = bool(cfg.get("think", False))
        self.max_batches = max(1, int(cfg.get("extractor_max_batches", 3)))
        self.fallback_on_empty = bool(cfg.get("extractor_fallback_on_empty", True))

    @staticmethod
    def _answer_type_guidance(answer_type: str) -> str:
        low = str(answer_type or "").strip().lower()
        if low == "count":
            return (
                "For count questions, preserve item-level evidence and any explicit count statements. "
                "Do not collapse multiple supported items into a bare number without context."
            )
        if low in {"temporal", "temporal_comparison"}:
            return (
                "For temporal questions, preserve separate events or states with their own time anchors. "
                "Do not merge before/after evidence into one vague claim."
            )
        if low == "update":
            return (
                "For update questions, preserve old and new values as separate claims when both are supported. "
                "Use state_snapshot when the evidence describes a current location, setting, status, or configuration."
            )
        if low == "preference":
            return (
                "For preference questions, extract supported preference direction and supported reason separately. "
                "Do not invent recommendations that are not explicitly grounded in the evidence."
            )
        return (
            "Prefer claims that preserve directly usable facts, states, rows, or list entries from the evidence. "
            "Do not summarize away important details."
        )

    @classmethod
    def _extract_content_tokens(cls, *texts: str) -> List[str]:
        seen = set()
        out: List[str] = []
        for text in texts:
            for token in cls._CONTENT_TOKEN_RE.findall(str(text or "").lower()):
                if len(token) <= 2 or token in cls._CONTENT_STOPWORDS or token in seen:
                    continue
                seen.add(token)
                out.append(token)
        return out

    @classmethod
    def _extract_named_tokens(cls, *texts: str) -> List[str]:
        seen = set()
        out: List[str] = []
        for text in texts:
            for token in cls._NAMED_TOKEN_RE.findall(str(text or "")):
                if token in cls._NAMED_TOKEN_STOPWORDS or token in seen:
                    continue
                seen.add(token)
                out.append(token)
        return out

    @classmethod
    def _infer_answer_type_from_query(cls, query: str) -> str:
        low = str(query or "").strip().lower()
        if not low:
            return ""
        if re.search(r"\b(how many|number of|count)\b", low):
            return "count"
        if re.search(r"\b(before|after|first|second|earlier|later|how long|weeks? after|days? after)\b", low):
            return "temporal" if " or " not in low else "temporal_comparison"
        if re.search(r"\b(currently|now|updated|change(?:d)?|switched|moved|where is|where are)\b", low):
            return "update"
        if cls._PREFERENCE_QUERY_RE.search(low):
            return "preference"
        return ""

    def _effective_answer_type(self, filtered_pack: Dict[str, Any]) -> str:
        answer_type = self._normalize_space(str(filtered_pack.get("answer_type", ""))).lower()
        query = str(filtered_pack.get("query", ""))
        inferred = self._infer_answer_type_from_query(query)
        if inferred == "preference" and self._PREFERENCE_QUERY_RE.search(query.lower()):
            if answer_type not in {"count", "temporal", "temporal_comparison"}:
                return "preference"
        if answer_type in {"count", "temporal", "temporal_comparison", "update", "preference"}:
            return answer_type
        if answer_type in {"", "factoid", "lookup"} and inferred:
            return inferred
        return answer_type or inferred or "factoid"

    @staticmethod
    def _fallback_guidance() -> str:
        return (
            "If the evidence is tabular, listed, or highly structured, convert the most relevant row, entry, "
            "or factual sentence into a supported claim. If you cannot extract many claims, extract at least one "
            "well-supported claim instead of returning an empty list."
        )

    @staticmethod
    def _infer_status_from_text(*parts: str) -> str:
        text = " ".join(str(p or "") for p in parts).lower()
        if re.search(r"\b(will|plan(?:ning)? to|planned to|going to|hope to|hoping to|intend to)\b", text):
            return "planned"
        if re.search(r"\b(currently|current|now|latest|recently|today)\b", text):
            return "current"
        if re.search(r"\b(was|were|had|previously|before|earlier|used to|ago)\b", text):
            return "past"
        return "unknown"

    @staticmethod
    def _infer_claim_type_from_text(*parts: str) -> str:
        text = " ".join(str(p or "") for p in parts).lower()
        if re.search(r"\b(moved|located|hanging|kept|stored|set to|switched to|changed to|currently in|currently on)\b", text):
            return "state_snapshot"
        if re.search(r"\b(met|graduated|completed|accepted|serviced|booked|scheduled|set|won|tried|led)\b", text):
            return "event_record"
        return "fact_statement"

    @staticmethod
    def _infer_state_key(*parts: str) -> str:
        text = " ".join(str(p or "") for p in parts).lower()
        if re.search(r"\b(where|located|location|hanging|hung|above|below|in my|in the|on my|on the|moved to)\b", text):
            return "location"
        if re.search(r"\b(personal best|time|date|when)\b", text):
            return "time"
        if re.search(r"\b(degree|major|certification|certificate)\b", text):
            return "credential"
        if re.search(r"\b(schedule|shift|rotation|sunday|monday|tuesday|wednesday|thursday|friday|saturday)\b", text):
            return "schedule"
        if re.search(r"\b(prefer|preference|like|want|interested in)\b", text):
            return "preference"
        return ""

    @staticmethod
    def _normalize_space(text: str) -> str:
        return " ".join(str(text or "").split())

    @staticmethod
    def _text_key(text: str) -> str:
        return " ".join(str(text or "").strip().lower().split())

    @staticmethod
    def _valid_claim_type(value: str) -> str:
        low = str(value or "").strip().lower()
        if low in {"fact_statement", "state_snapshot", "event_record"}:
            return low
        return "fact_statement"

    @staticmethod
    def _valid_status(value: str) -> str:
        low = str(value or "").strip().lower()
        if low in {"current", "past", "planned", "unknown"}:
            return low
        return "unknown"

    @staticmethod
    def _valid_modality(value: str) -> str:
        low = str(value or "").strip().lower()
        if low in {"observed", "historical", "planned", "reported", "unknown"}:
            return low
        return "unknown"

    @staticmethod
    def _valid_compare_role(value: str) -> str:
        low = str(value or "").strip().lower()
        if low in {"option_a", "option_b", "shared", ""}:
            return low
        return ""

    @staticmethod
    def _valid_unit_type(value: str) -> str:
        low = str(value or "").strip().lower()
        if low in {"fact_span", "state_span", "event_span", "table_row", "list_item"}:
            return low
        return "fact_span"

    @staticmethod
    def _number_from_text(text: str) -> Optional[int]:
        low = str(text or "").lower()
        m = re.search(r"\b(\d+)\b", low)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return None
        word_map = {
            "zero": 0,
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "eleven": 11,
            "twelve": 12,
        }
        for word, value in word_map.items():
            if re.search(rf"\b{word}\b", low):
                return value
        return None

    def _item_prompt_text(self, item: Dict[str, Any]) -> str:
        prompt_text = self._normalize_space(str(item.get("prompt_text", "")))
        if prompt_text:
            return prompt_text
        return self._normalize_space(str(item.get("text", "")))

    def _claim_key(self, claim: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
        return (
            self._text_key(str(claim.get("subject", ""))),
            self._text_key(str(claim.get("predicate", ""))),
            self._text_key(str(claim.get("value", ""))),
            self._text_key(str(claim.get("time_anchor", ""))),
            self._text_key(str(claim.get("state_key", ""))),
        )

    def _support_unit_key(self, unit: Dict[str, Any]) -> Tuple[str, str, str]:
        return (
            self._text_key(str(unit.get("text", ""))),
            self._text_key(str(unit.get("time_anchor", ""))),
            self._text_key(str(unit.get("state_key", ""))),
        )

    def _batch_anchor_key(self, item: Dict[str, Any], filtered_pack: Dict[str, Any]) -> str:
        target_object = self._text_key(str(filtered_pack.get("target_object", "")))
        if bool(item.get("target_match")) and target_object:
            return f"target:{target_object}"
        subject_hint = self._text_key(str(item.get("subject_hint", "")))
        if subject_hint:
            return f"subject:{subject_hint}"
        for field in ("matched_focus", "matched_entities", "matched_state", "matched_compare"):
            values = list(item.get(field, []))
            if values:
                candidate = self._text_key(str(values[0]))
                if candidate:
                    return f"{field}:{candidate}"
        backup_group = self._text_key(str(item.get("backup_group", "")))
        if backup_group:
            return f"backup:{backup_group}"
        return f"channel:{self._text_key(str(item.get('channel', 'unknown')))}"

    def _build_batches(
        self,
        *,
        selected: Sequence[Dict[str, Any]],
        filtered_pack: Dict[str, Any],
    ) -> List[List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        group_scores: Dict[str, float] = {}
        group_order: List[str] = []
        for item in list(selected):
            key = self._batch_anchor_key(item, filtered_pack)
            if key not in grouped:
                grouped[key] = []
                group_scores[key] = 0.0
                group_order.append(key)
            grouped[key].append(item)
            group_scores[key] = max(group_scores[key], float(item.get("score", 0.0)))
        ordered_keys = sorted(
            group_order,
            key=lambda k: (group_scores.get(k, 0.0), len(grouped.get(k, []))),
            reverse=True,
        )
        group_queues: Dict[str, List[Dict[str, Any]]] = {
            key: sorted(
                grouped.get(key, []),
                key=lambda x: (float(x.get("score", 0.0)), -int(x.get("raw_rank", 0))),
                reverse=True,
            )
            for key in ordered_keys
        }
        flattened: List[Dict[str, Any]] = []
        max_items = self.max_batches * self.batch_size
        while len(flattened) < max_items:
            progressed = False
            for key in ordered_keys:
                queue = group_queues.get(key, [])
                if not queue:
                    continue
                flattened.append(queue.pop(0))
                progressed = True
                if len(flattened) >= max_items:
                    break
            if not progressed:
                break
        return [
            flattened[idx : idx + self.batch_size]
            for idx in range(0, len(flattened), self.batch_size)
            if flattened[idx : idx + self.batch_size]
        ]

    def _prompt(
        self,
        *,
        filtered_pack: Dict[str, Any],
        batch: Sequence[Dict[str, Any]],
        fallback_mode: bool = False,
    ) -> str:
        effective_answer_type = self._effective_answer_type(filtered_pack)
        lines: List[str] = []
        for item in batch:
            evidence_id = str(item.get("evidence_id", "")).strip()
            channel = str(item.get("channel", "")).strip()
            session_date = str(item.get("session_date", "")).strip()
            bucket = str(item.get("bucket", "")).strip()
            prefix = f"{evidence_id} [{channel}"
            if bucket:
                prefix += f", {bucket}"
            if session_date:
                prefix += f", {session_date}"
            prefix += "]"
            signal_tags = []
            if bool(item.get("window_backup", False)):
                signal_tags.append("window_backup")
            if bool(item.get("structured_format", False)):
                signal_tags.append("structured")
            for tag in list(item.get("signals", []))[:4]:
                if str(tag).strip():
                    signal_tags.append(str(tag).strip())
            if signal_tags:
                prefix += " {" + ",".join(signal_tags) + "}"
            lines.append(f"{prefix}: {self._item_prompt_text(item)}")
        schema = {
            "support_units": [
                {
                    "unit_type": "fact_span|state_span|event_span|table_row|list_item",
                    "text": "compact supported unit text",
                    "subject_hint": "string",
                    "predicate_hint": "string",
                    "value_hint": "string",
                    "time_anchor": "string",
                    "state_key": "string",
                    "status": "current|past|planned|unknown",
                    "confidence": 0.0,
                    "evidence_ids": ["ev_001"],
                    "verbatim_span": "exact supporting phrase from one evidence line"
                }
            ],
            "claims": [
                {
                    "claim_type": "fact_statement|state_snapshot|event_record",
                    "subject": "string",
                    "predicate": "string",
                    "value": "string",
                    "time_anchor": "string",
                    "state_key": "string",
                    "status": "current|past|planned|unknown",
                    "modality": "observed|historical|planned|reported|unknown",
                    "compare_role": "option_a|option_b|shared|",
                    "numeric_value": "string",
                    "unit": "string",
                    "confidence": 0.0,
                    "evidence_ids": ["ev_001"],
                    "verbatim_span": "exact supporting phrase from one evidence line"
                }
            ]
        }
        prompt_lines = [
            "You are extracting question-scoped memory claims from retrieved evidence.",
            "Only output claims explicitly supported by the evidence lines.",
            "Do not infer missing values. Preserve conflicting claims separately.",
            "Prefer complete factual phrases over single words.",
            "When evidence is tabular, listed, or structured, convert the most relevant row or entry into a claim.",
            "First preserve support units tied to evidence lines; then produce higher-level claims only when they remain directly grounded.",
            "Leave compare_role empty unless the question explicitly compares two options.",
            "Use state_snapshot for changeable attributes, event_record for dated actions, fact_statement for stable facts.",
            self._answer_type_guidance(effective_answer_type),
        ]
        if fallback_mode:
            prompt_lines.append(self._fallback_guidance())
        prompt_lines.extend(
            [
                "Return strict JSON only with this shape:",
                json.dumps(schema, ensure_ascii=True),
                f"Question: {self._normalize_space(str(filtered_pack.get('query', '')))}",
                f"Intent: {self._normalize_space(str(filtered_pack.get('intent', 'lookup')))}",
                f"Answer type: {self._normalize_space(effective_answer_type)}",
                f"Target object: {self._normalize_space(str(filtered_pack.get('target_object', '')))}",
                f"Focus phrases: {json.dumps(list(filtered_pack.get('focus_phrases', [])), ensure_ascii=True)}",
                f"Max support units: {self.max_support_units_per_batch}",
                f"Max claims: {self.max_claims_per_batch}",
                "Evidence lines:",
                *lines,
            ]
        )
        return "\n".join(prompt_lines)

    def _call_model(self, prompt: str) -> str:
        llm = getattr(self.manager, "llm", None)
        if llm is None or (not hasattr(llm, "host")):
            raise RuntimeError("EvidenceGraphExtractor requires a configured Ollama-backed llm.")
        host = str(getattr(llm, "host", self.manager.config["llm"]["host"])).rstrip("/")
        opener = getattr(llm, "_opener", None)
        if opener is None:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        return ollama_generate_with_retry(
            host=host,
            model=self.model,
            prompt=prompt,
            temperature=self.temperature,
            timeout_sec=self.timeout_sec,
            opener=opener,
            max_attempts=max(1, int(getattr(llm, "retry_max_attempts", 1))),
            backoff_sec=float(getattr(llm, "retry_backoff_sec", 0.0)),
            retry_on_timeout=bool(getattr(llm, "retry_on_timeout", True)),
            retry_on_http_502=bool(getattr(llm, "retry_on_http_502", True)),
            retry_on_url_error=bool(getattr(llm, "retry_on_url_error", False)),
            max_output_tokens=self.max_output_tokens,
            think=self.think,
            response_format="json" if self.force_json_output else None,
        )

    def _normalize_claim(
        self,
        *,
        raw_claim: Dict[str, Any],
        valid_evidence_ids: Sequence[str],
        fallback_claim_id: str,
    ) -> Optional[Dict[str, Any]]:
        subject = self._normalize_space(str(raw_claim.get("subject", ""))).strip(" ,.;:!?")
        predicate = self._normalize_space(str(raw_claim.get("predicate", ""))).strip(" ,.;:!?")
        value = self._normalize_space(str(raw_claim.get("value", ""))).strip(" ,.;:!?")
        if not subject or not predicate or not value:
            return None
        evidence_ids = [
            str(x).strip()
            for x in list(raw_claim.get("evidence_ids", []))
            if str(x).strip() in valid_evidence_ids
        ]
        if not evidence_ids:
            return None
        confidence = raw_claim.get("confidence", 0.0)
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            confidence_value = 0.0
        confidence_value = min(1.0, max(0.0, confidence_value))
        time_anchor = self._normalize_space(str(raw_claim.get("time_anchor", "")))
        state_key = self._normalize_space(str(raw_claim.get("state_key", "")))
        verbatim_span = self._normalize_space(str(raw_claim.get("verbatim_span", "")))
        status = self._valid_status(raw_claim.get("status", ""))
        if status == "unknown":
            status = self._infer_status_from_text(predicate, value, time_anchor, verbatim_span)
        claim_type = self._valid_claim_type(raw_claim.get("claim_type", ""))
        if claim_type == "fact_statement":
            claim_type = self._infer_claim_type_from_text(predicate, value, time_anchor, verbatim_span)
        if not state_key:
            state_key = self._infer_state_key(predicate, value, verbatim_span)
        return {
            "claim_id": fallback_claim_id,
            "claim_type": claim_type,
            "subject": subject,
            "predicate": predicate,
            "value": value,
            "time_anchor": time_anchor,
            "state_key": state_key,
            "status": status,
            "modality": self._valid_modality(raw_claim.get("modality", "")),
            "compare_role": self._valid_compare_role(raw_claim.get("compare_role", "")),
            "numeric_value": self._normalize_space(str(raw_claim.get("numeric_value", ""))),
            "unit": self._normalize_space(str(raw_claim.get("unit", ""))),
            "confidence": round(confidence_value, 4),
            "evidence_ids": evidence_ids,
            "verbatim_span": verbatim_span,
        }

    def _normalize_support_unit(
        self,
        *,
        raw_unit: Dict[str, Any],
        valid_evidence_ids: Sequence[str],
        fallback_unit_id: str,
    ) -> Optional[Dict[str, Any]]:
        evidence_ids = [
            str(x).strip()
            for x in list(raw_unit.get("evidence_ids", []))
            if str(x).strip() in valid_evidence_ids
        ]
        verbatim_span = self._normalize_space(str(raw_unit.get("verbatim_span", "")))
        text = self._normalize_space(str(raw_unit.get("text", "")))
        if not text:
            text = verbatim_span
        if not evidence_ids or not (text or verbatim_span):
            return None
        subject_hint = self._normalize_space(str(raw_unit.get("subject_hint", "")))
        predicate_hint = self._normalize_space(str(raw_unit.get("predicate_hint", "")))
        value_hint = self._normalize_space(str(raw_unit.get("value_hint", "")))
        time_anchor = self._normalize_space(str(raw_unit.get("time_anchor", "")))
        state_key = self._normalize_space(str(raw_unit.get("state_key", "")))
        status = self._valid_status(raw_unit.get("status", ""))
        if status == "unknown":
            status = self._infer_status_from_text(text, verbatim_span, predicate_hint, value_hint, time_anchor)
        if not state_key:
            state_key = self._infer_state_key(text, verbatim_span, predicate_hint, value_hint)
        confidence = raw_unit.get("confidence", 0.0)
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            confidence_value = 0.0
        confidence_value = min(1.0, max(0.0, confidence_value))
        return {
            "unit_id": fallback_unit_id,
            "unit_type": self._valid_unit_type(raw_unit.get("unit_type", "")),
            "text": text,
            "subject_hint": subject_hint,
            "predicate_hint": predicate_hint,
            "value_hint": value_hint,
            "time_anchor": time_anchor,
            "state_key": state_key,
            "status": status,
            "confidence": round(confidence_value, 4),
            "evidence_ids": evidence_ids,
            "verbatim_span": verbatim_span,
        }

    @staticmethod
    def _extract_named_array(raw: str, key: str) -> str:
        match = re.search(rf'"{re.escape(key)}"\s*:\s*\[', str(raw or ""))
        if not match:
            return ""
        start = match.end() - 1
        depth = 0
        in_string = False
        escaped = False
        text = str(raw or "")
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
        return text[start:]

    def _recover_object_list(self, raw: str, key: str) -> List[Dict[str, Any]]:
        segment = self._extract_named_array(raw, key)
        if not segment:
            return []
        recovered: List[Dict[str, Any]] = []
        idx = 0
        while idx < len(segment):
            start = segment.find("{", idx)
            if start < 0:
                break
            depth = 0
            in_string = False
            escaped = False
            end = -1
            for cursor in range(start, len(segment)):
                ch = segment[cursor]
                if in_string:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == '"':
                        in_string = False
                    continue
                if ch == '"':
                    in_string = True
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = cursor + 1
                        break
            if end < 0:
                break
            obj_text = segment[start:end]
            parsed = safe_json_loads_relaxed(obj_text)
            if isinstance(parsed, dict):
                recovered.append(parsed)
            idx = end
        return recovered

    def _parse_model_output(self, raw: str) -> Dict[str, Any]:
        parsed = safe_json_loads_relaxed(extract_first_json_block(raw))
        support_units = parsed.get("support_units", []) if isinstance(parsed, dict) else []
        claims = parsed.get("claims", []) if isinstance(parsed, dict) else []
        if not isinstance(support_units, list):
            support_units = []
        if not isinstance(claims, list):
            claims = []
        if not support_units:
            support_units = self._recover_object_list(raw, "support_units")
        if not claims:
            claims = self._recover_object_list(raw, "claims")
        return {
            "support_units": support_units if isinstance(support_units, list) else [],
            "claims": claims if isinstance(claims, list) else [],
        }

    def _synthesize_claim_from_unit(
        self,
        *,
        unit: Dict[str, Any],
        fallback_claim_id: str,
    ) -> Optional[Dict[str, Any]]:
        subject = self._normalize_space(str(unit.get("subject_hint", "")))
        predicate = self._normalize_space(str(unit.get("predicate_hint", "")))
        value = self._normalize_space(str(unit.get("value_hint", "")))
        text = self._normalize_space(str(unit.get("text", "")))
        verbatim_span = self._normalize_space(str(unit.get("verbatim_span", "")))

        if not subject:
            return None
        if not predicate:
            if str(unit.get("state_key", "")).strip():
                predicate = self._normalize_space(str(unit.get("state_key", "")))
            else:
                predicate = "evidence"
        if not value:
            value = verbatim_span or text
        if not value:
            return None

        unit_type = str(unit.get("unit_type", "")).strip().lower()
        claim_type = "fact_statement"
        if unit_type in {"state_span"}:
            claim_type = "state_snapshot"
        elif unit_type in {"event_span"}:
            claim_type = "event_record"
        elif str(unit.get("state_key", "")).strip():
            claim_type = "state_snapshot"

        return {
            "claim_id": fallback_claim_id,
            "claim_type": claim_type,
            "subject": subject,
            "predicate": predicate,
            "value": value,
            "time_anchor": self._normalize_space(str(unit.get("time_anchor", ""))),
            "state_key": self._normalize_space(str(unit.get("state_key", ""))),
            "status": self._valid_status(unit.get("status", "")),
            "modality": "reported",
            "compare_role": "",
            "numeric_value": "",
            "unit": "",
            "confidence": round(float(unit.get("confidence", 0.0) or 0.0), 4),
            "evidence_ids": list(unit.get("evidence_ids", [])),
            "verbatim_span": verbatim_span or text,
        }

    def _synthesize_count_summary_claim(
        self,
        *,
        filtered_pack: Dict[str, Any],
        support_units: Sequence[Dict[str, Any]],
        fallback_claim_id: str,
    ) -> Optional[Dict[str, Any]]:
        answer_type = self._effective_answer_type(filtered_pack)
        if answer_type != "count" or not support_units:
            return None
        target_object = self._normalize_space(str(filtered_pack.get("target_object", "")))
        subject = target_object or self._normalize_space(str(support_units[0].get("subject_hint", "")))
        if not subject:
            return None

        explicit_counts: List[Tuple[int, Dict[str, Any]]] = []
        distinct_units: Dict[str, Dict[str, Any]] = {}
        for unit in list(support_units):
            text = self._normalize_space(
                str(unit.get("verbatim_span", "")) or str(unit.get("text", ""))
            )
            number = self._number_from_text(
                " ".join(
                    [
                        text,
                        str(unit.get("value_hint", "")),
                        str(unit.get("predicate_hint", "")),
                    ]
                )
            )
            if number is not None:
                explicit_counts.append((number, unit))
            key = self._text_key(
                " | ".join(
                    [
                        str(unit.get("subject_hint", "")),
                        str(unit.get("predicate_hint", "")),
                        str(unit.get("value_hint", "")),
                        str(unit.get("text", "")),
                    ]
                )
            )
            if key:
                distinct_units.setdefault(key, unit)

        chosen_value: Optional[int] = None
        source_unit: Optional[Dict[str, Any]] = None
        if explicit_counts:
            counts = [value for value, _unit in explicit_counts]
            if len(set(counts)) == 1:
                chosen_value = counts[0]
                source_unit = explicit_counts[0][1]
        if chosen_value is None and distinct_units:
            chosen_value = len(distinct_units)
            source_unit = next(iter(distinct_units.values()))
        if chosen_value is None:
            return None

        evidence_ids: List[str] = []
        for unit in list(support_units):
            for evidence_id in list(unit.get("evidence_ids", [])):
                if evidence_id not in evidence_ids:
                    evidence_ids.append(str(evidence_id))
        return {
            "claim_id": fallback_claim_id,
            "claim_type": "fact_statement",
            "subject": subject,
            "predicate": "count",
            "value": str(chosen_value),
            "time_anchor": "",
            "state_key": "",
            "status": "unknown",
            "modality": "reported",
            "compare_role": "",
            "numeric_value": str(chosen_value),
            "unit": "",
            "confidence": round(float((source_unit or {}).get("confidence", 0.0) or 0.0), 4),
            "evidence_ids": evidence_ids or list((source_unit or {}).get("evidence_ids", [])),
            "verbatim_span": self._normalize_space(
                str((source_unit or {}).get("verbatim_span", "")) or str((source_unit or {}).get("text", ""))
            ),
        }

    def _trim_clause(self, text: str, *, max_words: int = 20) -> str:
        clean = self._normalize_space(str(text or ""))
        if not clean:
            return ""
        clean = re.sub(r"^\d+\.\s*", "", clean)
        clean = clean.strip("-* ")
        parts = re.split(r"(?<=[.!?])\s+|[:;]\s+", clean, maxsplit=1)
        head = self._normalize_space(parts[0])
        words = head.split()
        if len(words) > max_words:
            head = " ".join(words[:max_words]).strip()
        return head.strip(" ,.;:")

    def _preference_subject(self, filtered_pack: Dict[str, Any]) -> str:
        target_object = self._normalize_space(str(filtered_pack.get("target_object", "")))
        if target_object:
            return target_object
        for phrase in list(filtered_pack.get("focus_phrases", [])):
            norm = self._normalize_space(str(phrase))
            if norm:
                return norm
        query = self._normalize_space(str(filtered_pack.get("query", "")))
        if query:
            return query.rstrip(" ?.!")[:120]
        return "user preference"

    def _preference_unit_phrase(self, unit: Dict[str, Any]) -> str:
        text = self._normalize_space(str(unit.get("text", "")))
        subject = self._normalize_space(str(unit.get("subject_hint", "")))
        predicate = self._normalize_space(str(unit.get("predicate_hint", ""))).lower()
        value = self._normalize_space(str(unit.get("value_hint", "")))
        unit_type = self._normalize_space(str(unit.get("unit_type", ""))).lower()
        verbatim = self._normalize_space(str(unit.get("verbatim_span", "")))

        if unit_type == "list_item" and text:
            return self._trim_clause(text, max_words=18)
        if value and value.lower() not in self._GENERIC_VALUE_HINTS:
            if predicate in self._META_PREDICATES:
                return self._trim_clause(value, max_words=18)
            if predicate in self._PREFERENCE_DIRECTION_PREDICATES:
                return self._trim_clause(value, max_words=18)
            if subject and predicate and predicate not in {"is", "are", "use"}:
                return self._trim_clause(f"{subject} {predicate} {value}", max_words=18)
            return self._trim_clause(value, max_words=18)
        if text:
            return self._trim_clause(text, max_words=18)
        if verbatim:
            return self._trim_clause(verbatim, max_words=18)
        return ""

    def _preference_reason_phrase(self, text: str) -> str:
        clean = self._normalize_space(str(text or ""))
        if not clean:
            return ""
        if self._META_ADVICE_RE.search(clean.lower()):
            return ""
        clean = clean.replace("**", " ").replace("*", " ")
        segments = [self._normalize_space(seg) for seg in re.split(r"[:;]\s+|(?<=[.!?])\s+", clean) if self._normalize_space(seg)]
        for seg in segments:
            content = self._extract_content_tokens(seg)
            if len(content) >= 4:
                return self._trim_clause(seg, max_words=24)
        return self._trim_clause(clean, max_words=24)

    def _is_meta_advice_text(self, text: str) -> bool:
        return bool(self._META_ADVICE_RE.search(self._normalize_space(str(text or "")).lower()))

    def _is_generic_preference_phrase(self, phrase: str) -> bool:
        norm = self._text_key(phrase)
        return norm in self._GENERIC_PREFERENCE_PHRASES

    def _named_alignment_score(
        self,
        *,
        query_named_tokens: Sequence[str],
        text: str,
    ) -> float:
        query_named = {str(x).lower() for x in list(query_named_tokens) if str(x).strip()}
        if not query_named:
            return 0.0
        candidate_named = {str(x).lower() for x in self._extract_named_tokens(text)}
        if not candidate_named:
            return 0.0
        if candidate_named.intersection(query_named):
            return 2.0
        return -4.0

    def _preference_unit_score(
        self,
        *,
        filtered_pack: Dict[str, Any],
        unit: Dict[str, Any],
    ) -> float:
        query_tokens = self._extract_content_tokens(
            str(filtered_pack.get("query", "")),
            " ".join(str(x) for x in list(filtered_pack.get("focus_phrases", []))),
            str(filtered_pack.get("target_object", "")),
        )
        blob = " ".join(
            [
                str(unit.get("text", "")),
                str(unit.get("subject_hint", "")),
                str(unit.get("predicate_hint", "")),
                str(unit.get("value_hint", "")),
                str(unit.get("verbatim_span", "")),
            ]
        ).lower()
        score = float(sum(1 for token in query_tokens if token in blob))
        if str(unit.get("unit_type", "")).strip().lower() == "list_item":
            score += 1.0
        predicate = self._normalize_space(str(unit.get("predicate_hint", ""))).lower()
        if predicate in self._PREFERENCE_DIRECTION_PREDICATES:
            score += 1.5
        if str(unit.get("state_key", "")).strip().lower() == "preference":
            score += 0.75
        if self._PREFERENCE_SIGNAL_RE.search(blob):
            score += 0.5
        return score

    def _preference_evidence_score(
        self,
        *,
        filtered_pack: Dict[str, Any],
        item: Dict[str, Any],
    ) -> float:
        query_tokens = self._extract_content_tokens(
            str(filtered_pack.get("query", "")),
            " ".join(str(x) for x in list(filtered_pack.get("focus_phrases", []))),
            str(filtered_pack.get("target_object", "")),
        )
        blob = self._normalize_space(str(item.get("text", ""))).lower()
        score = float(sum(1 for token in query_tokens if token in blob))
        if bool(item.get("structured_format", False)):
            score += 0.5
        if self._PREFERENCE_SIGNAL_RE.search(blob):
            score += 1.0
        return score

    def _preference_claim_score(
        self,
        *,
        filtered_pack: Dict[str, Any],
        claim: Dict[str, Any],
    ) -> float:
        query_tokens = self._extract_content_tokens(
            str(filtered_pack.get("query", "")),
            " ".join(str(x) for x in list(filtered_pack.get("focus_phrases", []))),
            str(filtered_pack.get("target_object", "")),
        )
        blob = " ".join(
            [
                str(claim.get("subject", "")),
                str(claim.get("predicate", "")),
                str(claim.get("value", "")),
                str(claim.get("verbatim_span", "")),
            ]
        ).lower()
        score = float(sum(1 for token in query_tokens if token in blob))
        predicate = self._normalize_space(str(claim.get("predicate", ""))).lower()
        state_key = self._normalize_space(str(claim.get("state_key", ""))).lower()
        if predicate in {"preferred_direction", "supported_reason", "recommend", "recommendation", "suggestion"}:
            score += 1.0
        if state_key == "preference":
            score += 0.75
        if self._PREFERENCE_SIGNAL_RE.search(blob):
            score += 0.5
        return score

    def _filter_preference_support_units(
        self,
        *,
        filtered_pack: Dict[str, Any],
        support_units: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if self._effective_answer_type(filtered_pack) != "preference":
            return list(support_units)
        query_named_tokens = self._extract_named_tokens(
            str(filtered_pack.get("query", "")),
            " ".join(str(x) for x in list(filtered_pack.get("focus_phrases", []))),
            str(filtered_pack.get("target_object", "")),
        )
        kept: List[Dict[str, Any]] = []
        for unit in list(support_units):
            source_text = self._normalize_space(
                str(unit.get("verbatim_span", "")) or str(unit.get("text", ""))
            )
            phrase = self._preference_unit_phrase(unit)
            if self._is_meta_advice_text(source_text):
                continue
            score = self._preference_unit_score(filtered_pack=filtered_pack, unit=unit)
            alignment = self._named_alignment_score(
                query_named_tokens=query_named_tokens,
                text=" ".join([phrase, source_text]),
            )
            predicate = self._normalize_space(str(unit.get("predicate_hint", ""))).lower()
            if predicate in self._META_PREDICATES and (not phrase or self._is_generic_preference_phrase(phrase)):
                continue
            if alignment < 0.0 and score < 2.5:
                continue
            if phrase and self._is_generic_preference_phrase(phrase) and score < 2.0:
                continue
            kept.append(unit)
        return kept

    def _filter_preference_claims(
        self,
        *,
        filtered_pack: Dict[str, Any],
        claims: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if self._effective_answer_type(filtered_pack) != "preference":
            return list(claims)
        query_named_tokens = self._extract_named_tokens(
            str(filtered_pack.get("query", "")),
            " ".join(str(x) for x in list(filtered_pack.get("focus_phrases", []))),
            str(filtered_pack.get("target_object", "")),
        )
        kept: List[Dict[str, Any]] = []
        for claim in list(claims):
            source_text = self._normalize_space(
                str(claim.get("verbatim_span", ""))
                or " ".join(
                    str(x) for x in [claim.get("subject", ""), claim.get("predicate", ""), claim.get("value", "")] if str(x).strip()
                )
            )
            phrase = self._trim_clause(
                " ".join(
                    str(x) for x in [claim.get("subject", ""), claim.get("predicate", ""), claim.get("value", "")] if str(x).strip()
                ),
                max_words=18,
            )
            if self._is_meta_advice_text(source_text):
                continue
            score = self._preference_claim_score(filtered_pack=filtered_pack, claim=claim)
            alignment = self._named_alignment_score(
                query_named_tokens=query_named_tokens,
                text=" ".join([phrase, source_text]),
            )
            predicate = self._normalize_space(str(claim.get("predicate", ""))).lower()
            if predicate in self._META_PREDICATES:
                continue
            if alignment < 0.0 and score < 2.5:
                continue
            if self._is_generic_preference_phrase(self._normalize_space(str(claim.get("value", "")))) and score < 2.0:
                continue
            kept.append(claim)
        return kept

    def _synthesize_preference_summary_claims(
        self,
        *,
        filtered_pack: Dict[str, Any],
        selected_evidence: Sequence[Dict[str, Any]],
        support_units: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if self._effective_answer_type(filtered_pack) != "preference":
            return []

        option_candidates: List[Tuple[float, float, str, List[str], str]] = []
        reason_candidates: List[Tuple[float, float, str, List[str], str]] = []
        seen_direction = set()
        seen_reason = set()
        query_named_tokens = self._extract_named_tokens(
            str(filtered_pack.get("query", "")),
            " ".join(str(x) for x in list(filtered_pack.get("focus_phrases", []))),
            str(filtered_pack.get("target_object", "")),
        )

        for unit in list(support_units):
            source_text = self._normalize_space(
                str(unit.get("verbatim_span", "")) or str(unit.get("text", ""))
            )
            phrase = self._preference_unit_phrase(unit)
            if phrase and not self._is_generic_preference_phrase(phrase) and not self._is_meta_advice_text(source_text):
                score = self._preference_unit_score(filtered_pack=filtered_pack, unit=unit)
                alignment = self._named_alignment_score(
                    query_named_tokens=query_named_tokens,
                    text=" ".join([phrase, source_text]),
                )
                score += alignment
                key = self._text_key(phrase)
                if key and key not in seen_direction and score >= 0.0:
                    option_candidates.append(
                        (
                            score,
                            alignment,
                            phrase,
                            [str(x) for x in list(unit.get("evidence_ids", [])) if str(x).strip()],
                            source_text,
                        )
                    )
                    seen_direction.add(key)
            reason = self._preference_reason_phrase(
                source_text
            )
            if reason:
                score = self._preference_unit_score(filtered_pack=filtered_pack, unit=unit)
                alignment = self._named_alignment_score(
                    query_named_tokens=query_named_tokens,
                    text=" ".join([reason, source_text]),
                )
                score += alignment
                key = self._text_key(reason)
                if key and key not in seen_reason and score >= 0.0:
                    reason_candidates.append(
                        (
                            score,
                            alignment,
                            reason,
                            [str(x) for x in list(unit.get("evidence_ids", [])) if str(x).strip()],
                            source_text,
                        )
                    )
                    seen_reason.add(key)

        for item in list(selected_evidence):
            if str(item.get("bucket", "")) == "conflict":
                continue
            source_text = self._normalize_space(str(item.get("text", "")))
            phrase = self._preference_reason_phrase(source_text)
            if not phrase:
                continue
            score = self._preference_evidence_score(filtered_pack=filtered_pack, item=item)
            alignment = self._named_alignment_score(
                query_named_tokens=query_named_tokens,
                text=" ".join([phrase, source_text]),
            )
            score += alignment
            evidence_ids = [str(item.get("evidence_id", "")).strip()] if str(item.get("evidence_id", "")).strip() else []
            key = self._text_key(phrase)
            if key and key not in seen_reason and score >= 0.0:
                reason_candidates.append((score, alignment, phrase, evidence_ids, source_text))
                seen_reason.add(key)

        option_candidates.sort(key=lambda x: (x[0], len(x[2].split())), reverse=True)
        reason_candidates.sort(key=lambda x: (x[0], len(x[2].split())), reverse=True)

        direction_parts: List[str] = []
        direction_evidence_ids: List[str] = []
        direction_spans: List[str] = []
        positive_named_alignment = False
        for _score, alignment, phrase, evidence_ids, span in option_candidates:
            if alignment > 0.0:
                positive_named_alignment = True
            if not phrase or self._text_key(phrase) in {self._text_key(x) for x in direction_parts}:
                continue
            direction_parts.append(phrase)
            direction_spans.append(span or phrase)
            for evidence_id in evidence_ids:
                if evidence_id and evidence_id not in direction_evidence_ids:
                    direction_evidence_ids.append(evidence_id)
            if len(direction_parts) >= 3:
                break

        reason_parts: List[str] = []
        reason_evidence_ids: List[str] = []
        reason_spans: List[str] = []
        seen_reason_parts = {self._text_key(x) for x in direction_parts}
        for _score, alignment, phrase, evidence_ids, span in reason_candidates:
            if alignment > 0.0:
                positive_named_alignment = True
            key = self._text_key(phrase)
            if not phrase or key in seen_reason_parts:
                continue
            reason_parts.append(phrase)
            reason_spans.append(span or phrase)
            for evidence_id in evidence_ids:
                if evidence_id and evidence_id not in reason_evidence_ids:
                    reason_evidence_ids.append(evidence_id)
            seen_reason_parts.add(key)
            if len(reason_parts) >= 2:
                break

        if not direction_parts and not reason_parts:
            return []
        if query_named_tokens and not positive_named_alignment:
            return []

        subject = self._preference_subject(filtered_pack)
        synthesized: List[Dict[str, Any]] = []
        if direction_parts:
            value = "; ".join(direction_parts)
            if reason_parts:
                value += ". Grounding: " + "; ".join(reason_parts)
            synthesized.append(
                {
                    "claim_type": "state_snapshot",
                    "subject": subject,
                    "predicate": "preferred_direction",
                    "value": value,
                    "time_anchor": "",
                    "state_key": "preference",
                    "status": "current",
                    "modality": "reported",
                    "compare_role": "",
                    "numeric_value": "",
                    "unit": "",
                    "confidence": 0.0,
                    "evidence_ids": direction_evidence_ids + [
                        x for x in reason_evidence_ids if x not in direction_evidence_ids
                    ],
                    "verbatim_span": " | ".join(direction_spans + reason_spans[:1]),
                }
            )
        if reason_parts:
            synthesized.append(
                {
                    "claim_type": "fact_statement",
                    "subject": subject,
                    "predicate": "supported_reason",
                    "value": "; ".join(reason_parts),
                    "time_anchor": "",
                    "state_key": "preference",
                    "status": "current",
                    "modality": "reported",
                    "compare_role": "",
                    "numeric_value": "",
                    "unit": "",
                    "confidence": 0.0,
                    "evidence_ids": reason_evidence_ids,
                    "verbatim_span": " | ".join(reason_spans),
                }
            )
        return synthesized

    def extract_claims(self, filtered_pack: Dict[str, Any]) -> Dict[str, Any]:
        selected: List[Dict[str, Any]] = []
        for bucket_name in ("core_evidence", "supporting_evidence", "conflict_evidence"):
            for item in list(filtered_pack.get(bucket_name, [])):
                enriched = dict(item)
                enriched["bucket"] = bucket_name.replace("_evidence", "")
                selected.append(enriched)
        if not self.enabled:
            return {
                "enabled": False,
                "model": self.model,
                "support_units": [],
                "claims": [],
                "raw_batches": [],
                "stats": {"selected_evidence": len(selected), "batches": 0, "support_units": 0, "claims": 0},
            }
        if not selected:
            return {
                "enabled": True,
                "model": self.model,
                "support_units": [],
                "claims": [],
                "raw_batches": [],
                "stats": {"selected_evidence": 0, "batches": 0, "support_units": 0, "claims": 0},
            }

        valid_ids = [str(x.get("evidence_id", "")).strip() for x in selected if str(x.get("evidence_id", "")).strip()]
        effective_answer_type = self._effective_answer_type(filtered_pack)
        allow_compare_role = effective_answer_type == "temporal_comparison"
        batches = self._build_batches(selected=selected, filtered_pack=filtered_pack)

        raw_batches: List[Dict[str, Any]] = []
        support_units: List[Dict[str, Any]] = []
        claims: List[Dict[str, Any]] = []
        seen_unit_keys = set()
        seen_keys = set()
        unit_counter = 0
        claim_counter = 0

        for batch_idx, batch in enumerate(batches, start=1):
            attempts: List[Dict[str, Any]] = []
            normalized_units: List[Dict[str, Any]] = []
            normalized_batch: List[Dict[str, Any]] = []
            for attempt_idx, fallback_mode in enumerate(
                [False, True] if self.fallback_on_empty else [False],
                start=1,
            ):
                prompt = self._prompt(
                    filtered_pack=filtered_pack,
                    batch=batch,
                    fallback_mode=fallback_mode,
                )
                raw = self._call_model(prompt)
                parsed = self._parse_model_output(raw)
                batch_units = list(parsed.get("support_units", []))
                batch_claims = list(parsed.get("claims", []))
                attempt_units: List[Dict[str, Any]] = []
                for raw_unit in batch_units[: self.max_support_units_per_batch]:
                    if not isinstance(raw_unit, dict):
                        continue
                    unit_counter += 1
                    unit = self._normalize_support_unit(
                        raw_unit=raw_unit,
                        valid_evidence_ids=valid_ids,
                        fallback_unit_id=f"su_{unit_counter:03d}",
                    )
                    if unit is None:
                        continue
                    key = self._support_unit_key(unit)
                    if key in seen_unit_keys:
                        continue
                    attempt_units.append(unit)
                attempt_claims: List[Dict[str, Any]] = []
                for raw_claim in batch_claims[: self.max_claims_per_batch]:
                    if not isinstance(raw_claim, dict):
                        continue
                    claim_counter += 1
                    claim = self._normalize_claim(
                        raw_claim=raw_claim,
                        valid_evidence_ids=valid_ids,
                        fallback_claim_id=f"cl_{claim_counter:03d}",
                    )
                    if claim is None:
                        continue
                    if not allow_compare_role:
                        claim["compare_role"] = ""
                    key = self._claim_key(claim)
                    if key in seen_keys:
                        continue
                    attempt_claims.append(claim)
                attempts.append(
                    {
                        "attempt_index": attempt_idx,
                        "fallback_mode": fallback_mode,
                        "raw_response": raw,
                        "support_unit_count": len(attempt_units),
                        "claim_count": len(attempt_claims),
                    }
                )
                if attempt_units and not normalized_units:
                    for unit in attempt_units:
                        key = self._support_unit_key(unit)
                        if key in seen_unit_keys:
                            continue
                        seen_unit_keys.add(key)
                        normalized_units.append(unit)
                        support_units.append(unit)
                if attempt_claims:
                    for claim in attempt_claims:
                        key = self._claim_key(claim)
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)
                        normalized_batch.append(claim)
                        claims.append(claim)
                    break
                if attempt_units:
                    break
            raw_batches.append(
                {
                    "batch_index": batch_idx,
                    "evidence_ids": [str(x.get("evidence_id", "")) for x in batch],
                    "attempts": attempts,
                    "support_unit_count": len(normalized_units),
                    "claim_count": len(normalized_batch),
                }
            )
            synthesized_batch: List[Dict[str, Any]] = []
            batch_claim_keys = {self._claim_key(claim) for claim in normalized_batch}
            if normalized_units:
                for unit in normalized_units:
                    claim_counter += 1
                    claim = self._synthesize_claim_from_unit(
                        unit=unit,
                        fallback_claim_id=f"cl_{claim_counter:03d}",
                    )
                    if claim is None:
                        continue
                    key = self._claim_key(claim)
                    if key in seen_keys or key in batch_claim_keys:
                        continue
                    seen_keys.add(key)
                    batch_claim_keys.add(key)
                    synthesized_batch.append(claim)
                    claims.append(claim)
                count_claim = self._synthesize_count_summary_claim(
                    filtered_pack=filtered_pack,
                    support_units=normalized_units,
                    fallback_claim_id=f"cl_{claim_counter + 1:03d}",
                )
                if count_claim is not None:
                    key = self._claim_key(count_claim)
                    if key not in seen_keys and key not in batch_claim_keys:
                        claim_counter += 1
                        count_claim["claim_id"] = f"cl_{claim_counter:03d}"
                        seen_keys.add(key)
                        batch_claim_keys.add(key)
                        synthesized_batch.append(count_claim)
                        claims.append(count_claim)
                if synthesized_batch:
                    raw_batches[-1]["claim_count"] = len(normalized_batch) + len(synthesized_batch)

        support_units = self._filter_preference_support_units(
            filtered_pack=filtered_pack,
            support_units=support_units,
        )
        claims = self._filter_preference_claims(
            filtered_pack=filtered_pack,
            claims=claims,
        )
        seen_keys = {self._claim_key(claim) for claim in claims}
        for raw_claim in self._synthesize_preference_summary_claims(
            filtered_pack=filtered_pack,
            selected_evidence=selected,
            support_units=support_units,
        ):
            claim_counter += 1
            claim = dict(raw_claim)
            claim["claim_id"] = f"cl_{claim_counter:03d}"
            key = (
                self._text_key(claim["subject"]),
                self._text_key(claim["predicate"]),
                self._text_key(claim["value"]),
                self._text_key(claim["time_anchor"]),
                self._text_key(claim["state_key"]),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            claims.append(claim)

        claims = self._filter_preference_claims(
            filtered_pack=filtered_pack,
            claims=claims,
        )

        return {
            "enabled": True,
            "model": self.model,
            "support_units": support_units,
            "claims": claims,
            "raw_batches": raw_batches,
            "stats": {
                "selected_evidence": len(selected),
                "batches": len(batches),
                "support_units": len(support_units),
                "claims": len(claims),
            },
        }
