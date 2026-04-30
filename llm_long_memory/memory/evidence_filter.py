"""Generic evidence filtering for question-scoped evidence graph construction."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Set

from llm_long_memory.memory.lexical_resources import BASIC_STOPWORDS, UPDATE_CUES


_TIME_PATTERNS = [
    re.compile(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", flags=re.IGNORECASE),
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", flags=re.IGNORECASE),
    re.compile(r"\b\d{1,2}:\d{2}\s?(?:am|pm)?\b", flags=re.IGNORECASE),
    re.compile(
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?\b",
        flags=re.IGNORECASE,
    ),
]
_NUMBER_PATTERNS = [
    re.compile(r"\b\d+(?:\.\d+)?\b", flags=re.IGNORECASE),
    re.compile(
        r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|once|twice)\b",
        flags=re.IGNORECASE,
    ),
]
_UPDATE_CUES = set(UPDATE_CUES).union(
    {
        "completed",
        "finished",
        "became",
    }
)
_COMPARATIVE_CUES = {
    "first",
    "before",
    "after",
    "earlier",
    "later",
    "more",
    "less",
    "than",
    "between",
}
_REASON_CUES = {
    "because",
    "since",
    "so",
    "helps",
    "helped",
    "for",
    "lets",
    "allows",
    "useful",
    "better",
}
_NOISE_PATTERNS = [
    re.compile(r"\bas an ai\b", flags=re.IGNORECASE),
    re.compile(r"\blarge language model\b", flags=re.IGNORECASE),
    re.compile(r"\bdon't have personal experiences\b", flags=re.IGNORECASE),
    re.compile(r"\bdo not have personal experiences\b", flags=re.IGNORECASE),
    re.compile(r"\bdon't have access\b", flags=re.IGNORECASE),
    re.compile(r"\bdo not have access\b", flags=re.IGNORECASE),
    re.compile(r"\b(?:tips?|how to|guide|tutorial|best practices?)\b", flags=re.IGNORECASE),
]
_CONTENT_STOPWORDS = set(BASIC_STOPWORDS).union(
    {
        "and",
        "am",
        "as",
        "at",
        "be",
        "been",
        "being",
        "by",
        "does",
        "from",
        "had",
        "has",
        "have",
        "how",
        "if",
        "into",
        "it",
        "its",
        "or",
        "our",
        "should",
        "so",
        "that",
        "their",
        "them",
        "there",
        "these",
        "they",
        "this",
        "those",
        "up",
        "us",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "would",
        "your",
    }
)


class EvidenceFilter:
    """Convert noisy mixed retrieval output into a compact evidence pack."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(cfg or {})
        self.max_core = max(1, int(cfg.get("filter_max_core", 8)))
        self.max_supporting = max(0, int(cfg.get("filter_max_supporting", 8)))
        self.max_conflict = max(0, int(cfg.get("filter_max_conflict", 6)))
        self.max_backup = max(0, int(cfg.get("filter_max_backup", 3)))
        self.max_selected = max(1, int(cfg.get("filter_max_selected", 18)))
        self.split_long_chars = max(160, int(cfg.get("filter_split_long_chars", 380)))
        self.min_sentence_chars = max(8, int(cfg.get("filter_min_sentence_chars", 18)))
        self.core_min_score = float(cfg.get("filter_core_min_score", 0.34))
        self.supporting_min_score = float(cfg.get("filter_supporting_min_score", 0.22))
        self.channel_caps = {
            "rag_evidence": max(1, int(cfg.get("filter_core_rag_cap", 5))),
            "evidence_pack": max(1, int(cfg.get("filter_core_pack_cap", 3))),
            "plan_combined_evidence": max(1, int(cfg.get("filter_core_plan_cap", 3))),
        }
        self.prompt_text_max_chars = max(80, int(cfg.get("filter_prompt_text_max_chars", 280)))
        self.prompt_text_max_sentences = max(1, int(cfg.get("filter_prompt_text_max_sentences", 2)))
        self.prompt_text_max_structured_units = max(
            1, int(cfg.get("filter_prompt_text_max_structured_units", 3))
        )

    @staticmethod
    def _normalize_space(text: str) -> str:
        return " ".join(str(text or "").split())

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", str(text or "").lower())

    def _content_tokens(self, text: str) -> List[str]:
        out: List[str] = []
        seen: Set[str] = set()
        for tok in self._tokenize(text):
            if len(tok) < 3 or tok in _CONTENT_STOPWORDS:
                continue
            if tok in seen:
                continue
            seen.add(tok)
            out.append(tok)
        return out

    @staticmethod
    def _text_key(text: str) -> str:
        return " ".join(str(text or "").strip().lower().split())

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        pieces = re.split(r"(?<=[.!?。！？])\s+|\n+", str(text or ""))
        return [" ".join(x.split()) for x in pieces if " ".join(x.split())]

    def _split_structured_units(self, text: str) -> List[str]:
        raw = str(text or "")
        normalized_lines = [
            self._normalize_space(x) for x in raw.splitlines() if self._normalize_space(x)
        ]
        pieces: List[str] = []

        if len(normalized_lines) >= 2:
            pieces.extend(normalized_lines)
        elif re.search(r"(?:^|\n)\s*(?:[-*]|\d+[.)])\s+", raw):
            for part in re.split(r"(?:^|\n)\s*(?=(?:[-*]|\d+[.)])\s+)", raw):
                normalized = self._normalize_space(part)
                if normalized:
                    pieces.append(normalized)
        elif raw.count(";") >= 2:
            for part in raw.split(";"):
                normalized = self._normalize_space(part)
                if normalized:
                    pieces.append(normalized)
        elif raw.count("|") >= 4:
            for part in raw.splitlines():
                normalized = self._normalize_space(part)
                if normalized:
                    pieces.append(normalized)

        deduped: List[str] = []
        seen: Set[str] = set()
        for piece in pieces:
            key = self._text_key(piece)
            if not key or key in seen or len(piece) < self.min_sentence_chars:
                continue
            seen.add(key)
            deduped.append(piece)
        return deduped

    def _compose_prompt_text(
        self,
        text: str,
        *,
        structured_units: Optional[Sequence[str]] = None,
    ) -> str:
        normalized = self._normalize_space(text)
        if not normalized:
            return ""
        pieces = [
            self._normalize_space(x)
            for x in list(structured_units or [])
            if self._normalize_space(x)
        ]
        separator = " ; "
        max_segments = self.prompt_text_max_structured_units
        if not pieces:
            pieces = self._split_sentences(normalized) or [normalized]
            separator = " "
            max_segments = self.prompt_text_max_sentences
        selected: List[str] = []
        total_chars = 0
        for piece in pieces:
            if len(selected) >= max_segments:
                break
            remaining = self.prompt_text_max_chars - total_chars
            if remaining <= 0:
                break
            clipped = piece
            if len(clipped) > remaining:
                clipped = clipped[: max(remaining - 3, 0)].rstrip(" ,.;:!?")
                if clipped:
                    clipped += "..."
            clipped = self._normalize_space(clipped)
            if not clipped:
                continue
            selected.append(clipped)
            total_chars += len(clipped)
            if total_chars >= self.prompt_text_max_chars:
                break
        prompt_text = separator.join(selected).strip()
        if not prompt_text:
            prompt_text = normalized[: self.prompt_text_max_chars].rstrip(" ,.;:!?")
            if len(normalized) > len(prompt_text):
                prompt_text = f"{prompt_text}..." if prompt_text else ""
        return prompt_text

    @staticmethod
    def _looks_structured(text: str) -> bool:
        normalized = str(text or "")
        low = normalized.lower()
        if normalized.count("|") >= 2:
            return True
        if normalized.count(":") >= 2 and len(normalized) >= 80:
            return True
        if normalized.count(";") >= 2 and len(normalized) >= 80:
            return True
        if re.search(r"(?:^|\s)(?:\d+[.)]|[-*])\s+\w", normalized):
            return True
        if len(re.findall(r"\b\d{1,2}:\d{2}\s?(?:am|pm)?\b", low, flags=re.IGNORECASE)) >= 2:
            return True
        return False

    @staticmethod
    def _contains_phrase(text_low: str, phrase: str) -> bool:
        phrase_low = " ".join(str(phrase or "").strip().lower().split())
        if not phrase_low:
            return False
        phrase_tokens = re.findall(r"[a-z0-9]+", phrase_low)
        text_tokens = set(re.findall(r"[a-z0-9]+", text_low))
        if not phrase_tokens or not text_tokens:
            return False
        if len(phrase_tokens) == 1:
            return phrase_tokens[0] in text_tokens
        return all(tok in text_tokens for tok in phrase_tokens)

    def _hard_drop(self, text: str, channel: str) -> Optional[str]:
        normalized = self._normalize_space(text)
        low = normalized.lower()
        if not normalized:
            return "empty"
        if channel == "plan_keywords":
            return "plan_keywords_control"
        if re.match(r"^(?:target_object|option_a|option_b|compare_rule)\s*:", low):
            return "control_line"
        if len(self._tokenize(normalized)) <= 1 and len(normalized) <= 10:
            return "short_fragment"
        return None

    def _source_prior(self, channel: str) -> float:
        if channel == "rag_evidence":
            return 0.22
        if channel == "evidence_pack":
            return 0.16
        if channel == "plan_combined_evidence":
            return 0.10
        return 0.04

    def _noise_penalty(self, text: str) -> float:
        low = self._normalize_space(text).lower()
        if not low:
            return 0.0
        penalty = 0.0
        structured = self._looks_structured(low)
        for pattern in _NOISE_PATTERNS:
            if pattern.search(low):
                penalty += 0.18
        if low.endswith("?"):
            penalty += 0.05
        if re.search(r"\b(you can|consider|try|should|recommended?)\b", low):
            penalty += 0.08
        if len(low) > 420 and not structured:
            penalty += 0.06
        return min(0.42, penalty)

    def _factness_score(self, text: str) -> float:
        low = text.lower()
        score = 0.0
        if re.search(
            r"\b(i|my|we|our|he|she|they|it)\b.{0,40}\b(was|were|is|are|have|had|did|own|owned|bought|moved|met|set|completed|graduated|tried|led|serviced|scheduled|booked)\b",
            low,
        ):
            score += 0.24
        if re.search(r"\b(?:in|on|at|to|from)\b\s+[A-Za-z0-9]", text):
            score += 0.06
        if any(pattern.search(low) for pattern in _TIME_PATTERNS):
            score += 0.10
        if any(pattern.search(low) for pattern in _NUMBER_PATTERNS):
            score += 0.08
        return min(0.34, score)

    def _guidance(self, query_plan: Dict[str, Any]) -> Dict[str, Any]:
        focus_phrases = [
            str(x).strip() for x in list(query_plan.get("focus_phrases", [])) if str(x).strip()
        ][:8]
        entities = [str(x).strip() for x in list(query_plan.get("entities", [])) if str(x).strip()][
            :8
        ]
        compare_options = [
            str(x).strip() for x in list(query_plan.get("compare_options", [])) if str(x).strip()
        ][:4]
        state_keys = [
            str(x).strip() for x in list(query_plan.get("state_keys", [])) if str(x).strip()
        ][:4]
        subject_focus_phrases: List[str] = []
        action_focus_phrases: List[str] = []
        target_tokens = set(
            self._content_tokens(" ".join([str(query_plan.get("target_object", "")).strip()]))
        )
        for phrase in focus_phrases:
            phrase_tokens = set(self._content_tokens(phrase))
            if (
                target_tokens
                and phrase_tokens
                and (phrase_tokens.issubset(target_tokens) or target_tokens.issubset(phrase_tokens))
            ):
                subject_focus_phrases.append(phrase)
            else:
                action_focus_phrases.append(phrase)
        content_tokens: List[str] = []
        seen_content: Set[str] = set()
        for text in (
            focus_phrases
            + entities
            + compare_options
            + state_keys
            + [str(x).strip() for x in list(query_plan.get("sub_queries", [])) if str(x).strip()][
                :4
            ]
        ):
            for tok in self._content_tokens(text):
                if tok in seen_content:
                    continue
                seen_content.add(tok)
                content_tokens.append(tok)
                if len(content_tokens) >= 16:
                    break
            if len(content_tokens) >= 16:
                break
        subject_tokens: List[str] = []
        seen_subject: Set[str] = set()
        for text in (
            [str(query_plan.get("target_object", "")).strip()]
            + subject_focus_phrases
            + entities[:2]
        ):
            for tok in self._content_tokens(text):
                if tok in seen_subject:
                    continue
                seen_subject.add(tok)
                subject_tokens.append(tok)
                if len(subject_tokens) >= 10:
                    break
            if len(subject_tokens) >= 10:
                break
        action_tokens: List[str] = []
        seen_action: Set[str] = set()
        for text in action_focus_phrases + state_keys + compare_options:
            for tok in self._content_tokens(text):
                if tok in seen_action or tok in set(subject_tokens):
                    continue
                seen_action.add(tok)
                action_tokens.append(tok)
                if len(action_tokens) >= 12:
                    break
            if len(action_tokens) >= 12:
                break
        return {
            "intent": str(query_plan.get("intent", "")).strip() or "lookup",
            "answer_type": str(query_plan.get("answer_type", "")).strip() or "factoid",
            "focus_phrases": focus_phrases,
            "subject_focus_phrases": subject_focus_phrases[:4],
            "action_focus_phrases": action_focus_phrases[:6],
            "entities": entities,
            "compare_options": compare_options,
            "state_keys": state_keys,
            "content_tokens": content_tokens[:16],
            "subject_tokens": subject_tokens[:10],
            "action_tokens": action_tokens[:12],
            "target_object": str(query_plan.get("target_object", "")).strip(),
            "sub_queries": [
                str(x).strip() for x in list(query_plan.get("sub_queries", [])) if str(x).strip()
            ][:4],
        }

    def _extract_time_anchors(self, text: str) -> List[str]:
        out: List[str] = []
        low = text.lower()
        for pattern in _TIME_PATTERNS:
            for m in pattern.findall(text):
                value = self._normalize_space(str(m)).strip(" ,.;:!?")
                if value:
                    out.append(value)
        for rel in re.findall(
            r"\b(?:today|yesterday|tomorrow|last month|last week|last year|this month|this week|a month ago|a few months ago|recently|currently|now)\b",
            low,
            flags=re.IGNORECASE,
        ):
            value = self._normalize_space(str(rel))
            if value:
                out.append(value)
        dedup: List[str] = []
        seen: Set[str] = set()
        for item in out:
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            dedup.append(item)
        return dedup[:4]

    def _extract_numeric_values(self, text: str) -> List[str]:
        out: List[str] = []
        for pattern in _NUMBER_PATTERNS:
            for m in pattern.findall(text):
                value = self._normalize_space(str(m)).strip(" ,.;:!?")
                if value:
                    out.append(value)
        dedup: List[str] = []
        seen: Set[str] = set()
        for item in out:
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            dedup.append(item)
        return dedup[:4]

    def _extract_value_signature(self, text: str) -> str:
        normalized = self._normalize_space(text)
        time_anchors = self._extract_time_anchors(normalized)
        if time_anchors:
            return f"time:{time_anchors[0].lower()}"
        numbers = self._extract_numeric_values(normalized)
        if numbers:
            return f"value:{numbers[0].lower()}"
        quoted = re.findall(r'"([^"]{2,80})"', normalized)
        if quoted:
            return f"value:{quoted[0].strip().lower()}"
        m = re.search(
            r"\b(?:is|are|was|were|became|set to|moved to|switched to)\s+(.+?)(?:[,.!?;:]|$)",
            normalized,
            flags=re.IGNORECASE,
        )
        if m:
            phrase = self._normalize_space(str(m.group(1))).strip(" ,.;:!?\"'")
            if phrase:
                return f"value:{phrase.lower()}"
        return ""

    def _sentence_signals(
        self,
        *,
        query: str,
        text: str,
        item: Dict[str, Any],
        guidance: Dict[str, Any],
    ) -> Dict[str, Any]:
        tokens = set(self._tokenize(text))
        query_tokens = set(self._tokenize(query))
        focus_phrases = list(guidance.get("focus_phrases", []))
        subject_focus_phrases = list(guidance.get("subject_focus_phrases", []))
        action_focus_phrases = list(guidance.get("action_focus_phrases", []))
        entities = list(guidance.get("entities", []))
        compare_options = list(guidance.get("compare_options", []))
        state_keys = list(guidance.get("state_keys", []))
        content_tokens = set(
            str(x).strip().lower()
            for x in list(guidance.get("content_tokens", []))
            if str(x).strip()
        )
        action_tokens = set(
            str(x).strip().lower()
            for x in list(guidance.get("action_tokens", []))
            if str(x).strip()
        )
        target_object = str(guidance.get("target_object", "")).strip()
        answer_type = str(guidance.get("answer_type", "")).strip().lower()
        low = text.lower()

        matched_focus = [p for p in focus_phrases if self._contains_phrase(low, p)]
        matched_subject_focus = [p for p in subject_focus_phrases if self._contains_phrase(low, p)]
        matched_action_focus = [p for p in action_focus_phrases if self._contains_phrase(low, p)]
        matched_entities = [p for p in entities if self._contains_phrase(low, p)]
        matched_compare = [p for p in compare_options if self._contains_phrase(low, p)]
        matched_state = [p for p in state_keys if self._contains_phrase(low, p)]
        target_match = bool(target_object and self._contains_phrase(low, target_object))
        time_anchors = self._extract_time_anchors(text)
        numeric_values = self._extract_numeric_values(text)
        update_hit = bool(tokens.intersection(_UPDATE_CUES))
        comparative_hit = bool(tokens.intersection(_COMPARATIVE_CUES))
        reason_hit = bool(tokens.intersection(_REASON_CUES))
        first_person_fact = bool(
            re.search(
                r"\b(i|my|we|our)\b.{0,48}\b(have|had|own|owned|bought|got|kept|led|managed|tried|completed|serviced|scheduled|planned|plan|met|moved|set)\b",
                low,
            )
        )
        structured_hit = self._looks_structured(text)
        query_overlap = (
            float(len(query_tokens.intersection(tokens))) / float(len(query_tokens))
            if query_tokens and tokens
            else 0.0
        )
        content_overlap = (
            float(len(content_tokens.intersection(tokens))) / float(len(content_tokens))
            if content_tokens and tokens
            else 0.0
        )
        action_overlap = (
            float(len(action_tokens.intersection(tokens))) / float(len(action_tokens))
            if action_tokens and tokens
            else 0.0
        )
        focus_overlap = 0.0
        if matched_focus:
            focus_overlap += 0.08 * len(matched_focus[:2])
        if matched_subject_focus:
            focus_overlap += 0.10 * len(matched_subject_focus[:2])
        if matched_action_focus:
            focus_overlap += 0.22 * len(matched_action_focus[:2])
        if matched_entities:
            focus_overlap += 0.12 * len(matched_entities[:2])
        if matched_compare:
            focus_overlap += 0.12 * len(matched_compare[:2])
        if matched_state:
            focus_overlap += 0.10 * len(matched_state[:2])
        if target_match:
            focus_overlap += 0.16
        if structured_hit:
            focus_overlap += 0.08

        score = (
            (0.34 * float(item.get("score", 0.0)))
            + (0.18 * query_overlap)
            + (0.12 * content_overlap)
            + (0.12 * action_overlap)
            + focus_overlap
            + self._source_prior(str(item.get("channel", "")))
            + self._factness_score(text)
        )
        signals: List[str] = []
        if matched_focus:
            signals.append("focus_match")
        if matched_subject_focus:
            signals.append("subject_focus_match")
        if matched_action_focus:
            signals.append("action_focus_match")
        if matched_entities:
            signals.append("entity_match")
        if matched_compare:
            signals.append("compare_match")
        if matched_state:
            signals.append("state_match")
        if target_match:
            signals.append("target_match")
        if time_anchors:
            signals.append("time_match")
            score += 0.10
        if numeric_values:
            signals.append("value_match")
            score += 0.08
        if update_hit:
            signals.append("update_signal")
            score += 0.08
        if comparative_hit:
            signals.append("comparative_signal")
            score += 0.08
        if reason_hit:
            signals.append("reason_signal")
            score += 0.06
        if structured_hit:
            signals.append("structured_format")
            score += 0.06
        if first_person_fact:
            signals.append("first_person_fact")
            if answer_type in {"count", "update"}:
                score += 0.08

        if content_overlap >= 0.25:
            signals.append("content_match")
        if action_overlap >= 0.20:
            signals.append("action_match")
        if answer_type in {"count", "update"} and time_anchors:
            score += 0.04
        if answer_type in {"count", "update"} and numeric_values:
            score += 0.04

        penalty = self._noise_penalty(text)
        channel = str(item.get("channel", "")).strip() or "unknown"
        needs_action_alignment = answer_type in {
            "count",
            "update",
            "preference",
            "temporal",
            "temporal_comparison",
        }
        if (
            needs_action_alignment
            and action_tokens
            and action_overlap < 0.20
            and not matched_action_focus
        ):
            strong_subject_alignment = bool(
                target_match
                or matched_subject_focus
                or matched_entities
                or matched_state
                or matched_compare
            )
            if not (
                first_person_fact
                or update_hit
                or time_anchors
                or numeric_values
                or strong_subject_alignment
                or structured_hit
            ):
                if channel == "plan_combined_evidence":
                    penalty += 0.12
                elif channel == "evidence_pack":
                    penalty += 0.08
                else:
                    penalty += 0.03
        score -= penalty

        subject_hint = ""
        for candidate in (
            matched_action_focus
            + matched_subject_focus
            + matched_focus
            + matched_entities
            + matched_compare
            + matched_state
        ):
            subject_hint = str(candidate).strip()
            if subject_hint:
                break
        if not subject_hint and target_match:
            subject_hint = target_object

        slot_keys: List[str] = []
        for item_key in time_anchors[:2]:
            slot_keys.append(f"time:{item_key.lower()}")
        for item_key in numeric_values[:2]:
            slot_keys.append(f"value:{item_key.lower()}")
        for item_key in matched_compare[:2]:
            slot_keys.append(f"compare:{item_key.lower()}")
        if subject_hint:
            slot_keys.append(f"subject:{self._text_key(subject_hint)}")

        return {
            "score": round(max(0.0, score), 4),
            "signals": signals,
            "matched_focus": matched_focus[:4],
            "matched_entities": matched_entities[:4],
            "matched_compare": matched_compare[:4],
            "matched_state": matched_state[:4],
            "target_match": target_match,
            "time_anchors": time_anchors,
            "numeric_values": numeric_values,
            "update_signal": update_hit,
            "comparative_signal": comparative_hit,
            "reason_signal": reason_hit,
            "subject_hint": subject_hint,
            "value_signature": self._extract_value_signature(text),
            "slot_keys": slot_keys,
            "structured_format": structured_hit,
            "noise_penalty": round(penalty, 4),
            "query_overlap": round(query_overlap, 4),
            "content_overlap": round(content_overlap, 4),
            "action_overlap": round(action_overlap, 4),
        }

    def _prepare_items(
        self,
        *,
        query: str,
        guidance: Dict[str, Any],
        unified_source: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        prepared: List[Dict[str, Any]] = []
        discarded: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        evidence_counter = 0

        for raw_rank, raw_item in enumerate(list(unified_source), start=1):
            text = self._normalize_space(str(raw_item.get("text", "")))
            channel = str(raw_item.get("channel", "")).strip() or "unknown"
            base_reason = self._hard_drop(text, channel)
            if base_reason is not None:
                discarded.append(
                    {"text": text, "channel": channel, "reason": base_reason, "raw_rank": raw_rank}
                )
                continue

            sentences = [text]
            structured_units = self._split_structured_units(str(raw_item.get("text", "")))
            if structured_units:
                sentences = structured_units
            elif (
                channel == "plan_combined_evidence"
                or len(text) > self.split_long_chars
                or "\n" in text
            ):
                sentences = self._split_sentences(text) or [text]
            backup_group = f"{channel}:{int(raw_item.get('chunk_id', 0) or 0)}:{raw_rank}"
            add_window_backup = len(sentences) > 1 and (
                self._looks_structured(text)
                or len(sentences) >= 3
                or "\n" in str(raw_item.get("text", ""))
            )

            if add_window_backup:
                key = self._text_key(text)
                if key and key not in seen:
                    seen.add(key)
                    evidence_counter += 1
                    backup_item = {
                        "evidence_id": f"ev_{evidence_counter:03d}",
                        "text": text,
                        "prompt_text": self._compose_prompt_text(
                            text,
                            structured_units=structured_units,
                        ),
                        "channel": channel,
                        "score": float(raw_item.get("score", 0.0)) * 0.92,
                        "chunk_id": int(raw_item.get("chunk_id", 0) or 0),
                        "session_date": str(raw_item.get("session_date", "")),
                        "raw_rank": raw_rank,
                        "sentence_index": 0,
                        "window_backup": True,
                        "backup_group": backup_group,
                    }
                    backup_item.update(
                        self._sentence_signals(
                            query=query,
                            text=text,
                            item=backup_item,
                            guidance=guidance,
                        )
                    )
                    prepared.append(backup_item)

            for sent_idx, sentence in enumerate(sentences, start=1):
                sent = self._normalize_space(sentence)
                if not sent or len(sent) < self.min_sentence_chars:
                    continue
                reason = self._hard_drop(sent, channel)
                if reason is not None:
                    discarded.append(
                        {
                            "text": sent,
                            "channel": channel,
                            "reason": reason,
                            "raw_rank": raw_rank,
                            "sentence_index": sent_idx,
                        }
                    )
                    continue
                key = self._text_key(sent)
                if not key or key in seen:
                    continue
                seen.add(key)
                evidence_counter += 1
                item = {
                    "evidence_id": f"ev_{evidence_counter:03d}",
                    "text": sent,
                    "prompt_text": self._compose_prompt_text(sent),
                    "channel": channel,
                    "score": float(raw_item.get("score", 0.0)),
                    "chunk_id": int(raw_item.get("chunk_id", 0) or 0),
                    "session_date": str(raw_item.get("session_date", "")),
                    "raw_rank": raw_rank,
                    "sentence_index": sent_idx,
                    "window_backup": False,
                    "backup_group": backup_group,
                }
                item.update(
                    self._sentence_signals(query=query, text=sent, item=item, guidance=guidance)
                )
                prepared.append(item)

        prepared.sort(
            key=lambda x: (float(x.get("score", 0.0)), -int(x.get("raw_rank", 0))),
            reverse=True,
        )
        return {"prepared": prepared, "discarded": discarded}

    @staticmethod
    def _slot_set(items: Sequence[Dict[str, Any]]) -> Set[str]:
        slots: Set[str] = set()
        for item in items:
            for slot in list(item.get("slot_keys", [])):
                if str(slot).strip():
                    slots.add(str(slot).strip().lower())
        return slots

    def _adds_new_slot(self, item: Dict[str, Any], seen_slots: Set[str]) -> bool:
        for slot in list(item.get("slot_keys", [])):
            if str(slot).strip().lower() not in seen_slots:
                return True
        return False

    def _type_coverage_needed(
        self,
        answer_type: str,
        selected: Sequence[Dict[str, Any]],
    ) -> Dict[str, int]:
        counts = {
            "time": 0,
            "value": 0,
            "compare": 0,
            "reason": 0,
            "subject": 0,
        }
        for item in selected:
            if list(item.get("time_anchors", [])):
                counts["time"] += 1
            if list(item.get("numeric_values", [])) or str(item.get("value_signature", "")).strip():
                counts["value"] += 1
            if list(item.get("matched_compare", [])):
                counts["compare"] += 1
            if bool(item.get("reason_signal", False)):
                counts["reason"] += 1
            if str(item.get("subject_hint", "")).strip():
                counts["subject"] += 1

        if answer_type == "count":
            return {"subject": max(0, 1 - counts["subject"]), "value": max(0, 1 - counts["value"])}
        if answer_type == "temporal_comparison":
            return {"compare": max(0, 2 - counts["compare"]), "time": max(0, 2 - counts["time"])}
        if answer_type == "temporal":
            return {"time": max(0, 2 - counts["time"])}
        if answer_type == "update":
            return {"subject": max(0, 1 - counts["subject"]), "value": max(0, 2 - counts["value"])}
        if answer_type == "preference":
            return {
                "subject": max(0, 1 - counts["subject"]),
                "reason": max(0, 1 - counts["reason"]),
            }
        return {"subject": max(0, 1 - counts["subject"])}

    def _coverage_match(self, item: Dict[str, Any], need: Dict[str, int]) -> bool:
        if need.get("time", 0) > 0 and list(item.get("time_anchors", [])):
            return True
        if need.get("value", 0) > 0 and (
            list(item.get("numeric_values", [])) or str(item.get("value_signature", "")).strip()
        ):
            return True
        if need.get("compare", 0) > 0 and list(item.get("matched_compare", [])):
            return True
        if need.get("reason", 0) > 0 and bool(item.get("reason_signal", False)):
            return True
        if need.get("subject", 0) > 0 and str(item.get("subject_hint", "")).strip():
            return True
        return False

    @staticmethod
    def _is_anchor_item(item: Dict[str, Any]) -> bool:
        return bool(
            item.get("target_match")
            or list(item.get("matched_focus", []))
            or list(item.get("matched_entities", []))
            or list(item.get("matched_state", []))
            or list(item.get("matched_compare", []))
            or bool(item.get("structured_format", False))
        )

    def _select_core(
        self, items: Sequence[Dict[str, Any]], answer_type: str
    ) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        seen_slots: Set[str] = set()
        channel_use: Dict[str, int] = {}

        for item in items:
            if len(selected) >= self.max_core:
                break
            channel = str(item.get("channel", "")).strip() or "unknown"
            used = int(channel_use.get(channel, 0))
            cap = int(self.channel_caps.get(channel, 2))
            want = self._adds_new_slot(item, seen_slots)
            if float(item.get("score", 0.0)) < self.core_min_score and not want:
                continue
            if used >= cap and not want:
                continue
            selected.append(item)
            channel_use[channel] = used + 1
            seen_slots.update(self._slot_set([item]))

        need = self._type_coverage_needed(answer_type, selected)
        if any(v > 0 for v in need.values()):
            for item in items:
                if len(selected) >= self.max_core:
                    break
                if item in selected:
                    continue
                if self._coverage_match(item, need):
                    selected.append(item)
                    seen_slots.update(self._slot_set([item]))
                    need = self._type_coverage_needed(answer_type, selected)
                    if not any(v > 0 for v in need.values()):
                        break
        return selected[: self.max_core]

    def _select_supporting(
        self,
        items: Sequence[Dict[str, Any]],
        selected_core: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen_slots = self._slot_set(selected_core)
        selected_ids = {str(x.get("evidence_id", "")) for x in selected_core}
        selected_groups = {
            str(x.get("backup_group", "")).strip()
            for x in selected_core
            if str(x.get("backup_group", "")).strip()
        }
        for item in items:
            if len(out) >= self.max_supporting:
                break
            evidence_id = str(item.get("evidence_id", ""))
            if evidence_id in selected_ids:
                continue
            adds_slot = self._adds_new_slot(item, seen_slots)
            anchor_item = self._is_anchor_item(item)
            backup_item = bool(item.get("window_backup", False))
            if float(item.get("score", 0.0)) < self.supporting_min_score and not (
                adds_slot or anchor_item or backup_item
            ):
                continue
            if (
                not adds_slot
                and not anchor_item
                and not backup_item
                and float(item.get("score", 0.0)) < (self.supporting_min_score + 0.08)
            ):
                continue
            out.append(item)
            seen_slots.update(self._slot_set([item]))
            selected_ids.add(evidence_id)
            if len(selected_core) + len(out) >= self.max_selected:
                break

        backup_added = 0
        for item in items:
            if (
                len(out) >= self.max_supporting
                or len(selected_core) + len(out) >= self.max_selected
            ):
                break
            if backup_added >= self.max_backup:
                break
            evidence_id = str(item.get("evidence_id", ""))
            if evidence_id in selected_ids or not bool(item.get("window_backup", False)):
                continue
            backup_group = str(item.get("backup_group", "")).strip()
            if not backup_group or backup_group not in selected_groups:
                continue
            if not self._is_anchor_item(item):
                continue
            out.append(item)
            selected_ids.add(evidence_id)
            backup_added += 1
        return out[: self.max_supporting]

    def _select_conflicts(
        self,
        items: Sequence[Dict[str, Any]],
        selected: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        selected_ids = {str(x.get("evidence_id", "")) for x in selected}
        subject_values: Dict[str, Set[str]] = {}
        for item in selected:
            subject = self._text_key(str(item.get("subject_hint", "")))
            value = str(item.get("value_signature", "")).strip().lower()
            if subject and value:
                subject_values.setdefault(subject, set()).add(value)

        for item in items:
            if len(out) >= self.max_conflict:
                break
            evidence_id = str(item.get("evidence_id", ""))
            if evidence_id in selected_ids:
                continue
            subject = self._text_key(str(item.get("subject_hint", "")))
            value = str(item.get("value_signature", "")).strip().lower()
            if not subject or not value:
                continue
            seen_values = subject_values.get(subject, set())
            if seen_values and value not in seen_values:
                out.append(item)
                subject_values.setdefault(subject, set()).add(value)
                selected_ids.add(evidence_id)
        return out[: self.max_conflict]

    def build_filtered_pack(
        self,
        *,
        query: str,
        query_plan: Dict[str, Any],
        unified_source: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        guidance = self._guidance(query_plan)
        prepared_bundle = self._prepare_items(
            query=query,
            guidance=guidance,
            unified_source=unified_source,
        )
        prepared = list(prepared_bundle["prepared"])
        discarded = list(prepared_bundle["discarded"])
        answer_type = str(guidance.get("answer_type", "factoid")).strip().lower()

        core = self._select_core(prepared, answer_type)
        supporting = self._select_supporting(prepared, core)
        conflict = self._select_conflicts(prepared, core + supporting)
        selected_ids = {
            str(x.get("evidence_id", "")) for x in list(core) + list(supporting) + list(conflict)
        }
        overflow = [
            item for item in prepared if str(item.get("evidence_id", "")) not in selected_ids
        ]

        insufficient = len(core) <= 0
        if answer_type in {"temporal", "temporal_comparison"}:
            total_time = sum(1 for x in core + supporting if list(x.get("time_anchors", [])))
            insufficient = insufficient or total_time < 2
        elif answer_type == "count":
            total_value = sum(1 for x in core + supporting if list(x.get("numeric_values", [])))
            insufficient = insufficient or total_value < 1
        elif answer_type == "update":
            total_values = {
                str(x.get("value_signature", "")).strip().lower()
                for x in core + supporting + conflict
                if str(x.get("value_signature", "")).strip()
            }
            insufficient = insufficient or len(total_values) < 1

        return {
            "query": str(query or ""),
            "intent": str(guidance.get("intent", "")),
            "answer_type": str(guidance.get("answer_type", "")),
            "focus_phrases": list(guidance.get("focus_phrases", [])),
            "sub_queries": list(guidance.get("sub_queries", [])),
            "target_object": str(guidance.get("target_object", "")),
            "core_evidence": core,
            "supporting_evidence": supporting,
            "conflict_evidence": conflict,
            "overflow_evidence": overflow,
            "discarded": discarded,
            "stats": {
                "input_items": len(list(unified_source)),
                "prepared_items": len(prepared),
                "core_count": len(core),
                "supporting_count": len(supporting),
                "conflict_count": len(conflict),
                "discarded_count": len(discarded),
                "insufficient": bool(insufficient),
            },
        }
