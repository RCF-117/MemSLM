"""Structured fact extraction engine for LongMemory."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from llm_long_memory.llm.ollama_client import ollama_generate_with_retry
from llm_long_memory.utils.logger import logger


class LongMemoryExtractor:
    """Run structured fact extraction using LongMemory config/runtime state."""

    def __init__(self, memory: Any) -> None:
        self.m = memory

    def extract_events_structured(
        self,
        message: Dict[str, Any],
        force: bool = False,
    ) -> List[Dict[str, Any]]:
        """Extract atomic factual memories using the configured local model."""
        if not self.m.extractor_enabled:
            return []
        role = str(message.get("role", "user")).strip().lower() or "user"
        content = str(message.get("content", "")).strip()
        if role not in self.m.extractor_process_roles:
            return []
        if len(content) < self.m.extractor_min_chars or len(content) > self.m.extractor_max_chars:
            return []

        self.m._extractor_seen_messages += 1
        if (not force) and ((self.m._extractor_seen_messages % self.m.extractor_every_n_messages) != 0):
            return []

        self.m._warmup_extractor()
        self.m._extractor_calls += 1
        trimmed = content[: self.m.extractor_input_max_chars]

        out = self._extract_once(role=role, trimmed=trimmed)
        if out:
            return out
        if (not self.m.extractor_compact_retry_enabled) or (
            len(trimmed) <= self.m.extractor_compact_window_chars
        ):
            self.m._extractor_empty_payload += 1
            return []
        compact = self._compact_window(trimmed, self.m.extractor_compact_window_chars)
        if compact == trimmed:
            self.m._extractor_empty_payload += 1
            return []
        self.m._extractor_retry_compact += 1
        out = self._extract_once(role=role, trimmed=compact)
        if out:
            return out
        self.m._extractor_empty_payload += 1
        return []

    @staticmethod
    def _compact_window(text: str, max_chars: int) -> str:
        normalized = " ".join(str(text).split()).strip()
        return normalized[: max(64, int(max_chars))]

    def _build_prompt(self, trimmed: str) -> str:
        return (
            "Task: extract atomic factual memories from one message.\n"
            "Output: JSON only, valid UTF-8, no markdown.\n"
            "Schema (strict):\n"
            '{"events":[{"subject":"","action":"","value":"","time":"","location":"","evidence_span":"","fact_type":"state_fact|episodic_fact","confidence":0.0}]}\n'
            "Hard rules:\n"
            "- Extract stable answer-bearing facts, not generic topics.\n"
            "- A single message may yield multiple events; extract each distinct answer-bearing fact as a separate event, even if they are independent or only loosely related.\n"
            "- Prefer names, counts, owned items, locations, certifications, dates, preferences, updates, and completed actions.\n"
            "- Use fact_type=state_fact for stable facts that may update over time, such as counts, ownership, locations, certifications, current status, preferences, scores, ratios, and latest values.\n"
            "- Use fact_type=episodic_fact for one-off happenings, meetings, visits, actions, and comparisons that should coexist rather than overwrite each other.\n"
            "- Exclude pure chitchat, instructions, meta statements, or vague summaries.\n"
            "- Prefer the most answer-bearing fact in each sentence window.\n"
            "- value should contain the answer-bearing value when possible.\n"
            "- evidence_span must be a short exact quote from the message.\n"
            "- confidence is optional but should be in [0,1] when present.\n"
            "- If no event, return {\"events\":[]}.\n"
            f"Max events: {self.m.extractor_max_events_per_message}\n"
            f"Message:\n{trimmed}\n"
        )

    def _extract_once(self, *, role: str, trimmed: str) -> List[Dict[str, Any]]:
        prompt = self._build_prompt(trimmed)
        try:
            raw = ollama_generate_with_retry(
                host=self.m.ollama_host,
                model=self.m.extractor_model,
                prompt=prompt,
                temperature=self.m.extractor_temperature,
                timeout_sec=self.m.extractor_timeout_sec,
                opener=self.m._extract_opener,
                max_attempts=self.m.extractor_retry_max_attempts,
                backoff_sec=self.m.extractor_retry_backoff_sec,
                retry_on_timeout=self.m.extractor_retry_on_timeout,
                retry_on_http_502=self.m.extractor_retry_on_http_502,
                retry_on_url_error=self.m.extractor_retry_on_url_error,
                max_output_tokens=self.m.extractor_max_output_tokens,
                think=self.m.extractor_think,
                response_format="json" if self.m.extractor_force_json_output else None,
            )
            payload_any = self.m._safe_json_loads_relaxed(self.m._extract_first_json_block(raw))
            if isinstance(payload_any, list):
                payload = {"events": payload_any}
            elif isinstance(payload_any, dict):
                payload = payload_any
            else:
                payload = {}
            self.m._extractor_json_success += 1
            rows = payload.get("events", [])
            if isinstance(rows, dict):
                rows = [rows]
            elif not isinstance(rows, list):
                maybe_single = payload.get("event", None)
                rows = [maybe_single] if isinstance(maybe_single, dict) else []
            if not isinstance(rows, list):
                self.m._extractor_failures += 1
                return []

            out: List[Dict[str, Any]] = []
            for item in rows[: self.m.extractor_max_events_per_message]:
                if not isinstance(item, dict):
                    continue
                if not self._schema_valid(item):
                    continue
                self.m._extractor_schema_pass += 1

                subject = self.m._entity_norm.normalize_subject(
                    str(item.get("subject", "")).strip(),
                    role,
                )
                action = str(item.get("action", item.get("predicate", ""))).strip()
                obj = self.m._entity_norm.normalize_object(
                    str(item.get("object", "")).strip(),
                    role,
                )
                value_text = self.m._entity_norm.normalize_object(
                    str(item.get("value", "")).strip(),
                    role,
                ) or obj
                time_text = str(item.get("time", "")).strip()
                location_text = str(item.get("location", "")).strip()
                value_type = str(item.get("value_type", "")).strip().lower()
                fact_slot = str(item.get("fact_slot", "")).strip().lower()
                fact_type = self._normalize_fact_type(str(item.get("fact_type", "")).strip().lower())
                canonical_fact = str(item.get("canonical_fact", "")).strip()
                event_text = str(item.get("event_text", "")).strip()
                raw_span = str(item.get("evidence_span", "")).strip()
                if not raw_span:
                    raw_span = str(item.get("raw_span", "")).strip()
                if (not value_text) and raw_span:
                    value_text = self.m._entity_norm.normalize_object(raw_span, role)

                if not value_type:
                    value_type = self._infer_value_type(
                        value_text=value_text,
                        time_text=time_text,
                        location_text=location_text,
                    )
                if not fact_slot:
                    fact_slot = self._infer_fact_slot(
                        action=action,
                        value_type=value_type,
                        time_text=time_text,
                        location_text=location_text,
                    )
                value_text = self.m._text.normalize_value_text(
                    value_text,
                    fact_slot=fact_slot,
                    value_type=value_type,
                )
                if not canonical_fact:
                    canonical_fact = self._build_canonical_fact(
                        subject=subject,
                        action=action,
                        value_text=value_text,
                        time_text=time_text,
                        location_text=location_text,
                    )
                if canonical_fact:
                    event_text = canonical_fact
                elif subject and action and value_text:
                    event_text = f"{subject} | {action} | {value_text}"
                if not event_text:
                    continue

                grounding_ratio = 0.0
                if raw_span:
                    grounding_ratio = self.m._span_overlap_ratio(raw_span, trimmed)

                confidence_default = max(
                    float(self.m.gating_hard_min_confidence),
                    float(self.m.extractor_min_confidence),
                )
                confidence = float(item.get("confidence", confidence_default) or confidence_default)
                confidence = max(0.0, min(1.0, confidence))
                if raw_span and grounding_ratio < float(self.m.extractor_span_grounding_min_overlap):
                    confidence *= max(0.5, grounding_ratio + 0.2)

                raw_keywords = item.get("keywords", [])
                if isinstance(raw_keywords, list):
                    model_keywords = [str(x).strip().lower() for x in raw_keywords if str(x).strip()]
                else:
                    model_keywords = self.m._tokenize(str(raw_keywords))
                keywords = self.m._build_keywords(
                    model_keywords=model_keywords,
                    subject=subject,
                    action=action,
                    obj=value_text,
                    event_text=event_text,
                    raw_span=raw_span,
                    source_content=trimmed,
                    time_text=time_text,
                    location_text=location_text,
                )
                if not fact_type:
                    fact_type = self._infer_fact_type(
                        subject=subject,
                        action=action,
                        value_text=value_text,
                        value_type=value_type,
                        fact_slot=fact_slot,
                        time_text=time_text,
                        location_text=location_text,
                        keywords=keywords,
                        event_text=event_text,
                    )
                if not fact_type:
                    fact_type = "episodic_fact"

                if self.m.gating_enabled:
                    accepted, reason, _score = self.m._gate.evaluate(
                        subject=subject,
                        action=action,
                        obj=value_text,
                        confidence=confidence,
                        time_text=time_text,
                        location_text=location_text,
                        evidence_span=raw_span,
                        source_content=trimmed,
                        keywords=keywords,
                    )
                    if not accepted:
                        self.m._record_reject(reason)
                        self.m._stage_rejected_event(
                            reason=reason,
                            subject=subject,
                            action=action,
                            obj=value_text,
                            event_text=event_text,
                            keywords=keywords,
                            role=role,
                            confidence=confidence,
                            source_model=self.m.extractor_model,
                            raw_span=raw_span or event_text,
                            source_content=trimmed,
                        )
                        continue
                else:
                    if not self.m._event_is_valid(
                        subject=subject,
                        action=action,
                        obj=value_text,
                        confidence=confidence,
                        evidence_span=raw_span,
                        source_content=trimmed,
                    ):
                        self.m._record_reject("legacy_gate_reject")
                        self.m._stage_rejected_event(
                            reason="legacy_gate_reject",
                            subject=subject,
                            action=action,
                            obj=value_text,
                            event_text=event_text,
                            keywords=keywords,
                            role=role,
                            confidence=confidence,
                            source_model=self.m.extractor_model,
                            raw_span=raw_span or event_text,
                            source_content=trimmed,
                        )
                        continue

                out.append(
                    {
                        "subject": subject,
                        "action": action,
                        "object": value_text,
                        "value": value_text,
                        "value_type": value_type,
                        "fact_slot": fact_slot,
                        "canonical_fact": canonical_fact or event_text,
                        "event_text": event_text,
                        "keywords": keywords,
                        "time": time_text,
                        "location": location_text,
                        "confidence": confidence,
                        "role": role,
                        "source_model": self.m.extractor_model,
                        "raw_span": raw_span or event_text,
                        "source_content": trimmed,
                        "fact_type": fact_type,
                    }
                )
            if out:
                self.m._extractor_success += 1
            else:
                self.m._extractor_failures += 1
            return out
        except (RuntimeError, ValueError, TypeError, OSError, json.JSONDecodeError) as exc:
            self.m._extractor_failures += 1
            logger.warn(f"LongMemory structured extraction failed: {exc}")
            return []

    @staticmethod
    def _schema_valid(item: Dict[str, Any]) -> bool:
        action = str(item.get("action", item.get("predicate", "")) or "").strip()
        subject = str(item.get("subject", "") or "").strip()
        value = str(item.get("value", item.get("object", "")) or "").strip()
        evidence = str(item.get("evidence_span", item.get("raw_span", "")) or "").strip()
        if not action:
            return False
        if not any([subject, value, evidence]):
            return False
        keywords = item.get("keywords", [])
        if (not isinstance(keywords, list)) and (not isinstance(keywords, str)):
            return False
        try:
            if "confidence" in item and str(item.get("confidence", "")).strip() != "":
                float(item.get("confidence", 0.0) or 0.0)
        except (ValueError, TypeError):
            return False
        return True

    @staticmethod
    def _normalize_fact_type(fact_type: str) -> str:
        value = str(fact_type or "").strip().lower()
        if value in {"state", "state_fact", "state-fact", "latest", "current"}:
            return "state_fact"
        if value in {"episodic", "episodic_fact", "episodic-fact", "event", "fact"}:
            return "episodic_fact"
        return ""

    def _infer_fact_type(
        self,
        *,
        subject: str,
        action: str,
        value_text: str,
        value_type: str,
        fact_slot: str,
        time_text: str,
        location_text: str,
        keywords: List[str],
        event_text: str,
    ) -> str:
        bag = set(self.m._tokenize(action))
        bag.update(self.m._tokenize(value_text))
        bag.update(self.m._tokenize(event_text))
        bag.update({str(x).strip().lower() for x in keywords if str(x).strip()})
        slot = str(fact_slot or "").strip().lower()
        vtype = str(value_type or "").strip().lower()
        state_terms = {
            "current",
            "currently",
            "latest",
            "now",
            "own",
            "owned",
            "have",
            "has",
            "is",
            "are",
            "was",
            "were",
            "live",
            "lives",
            "reside",
            "resides",
            "located",
            "location",
            "degree",
            "graduated",
            "graduate",
            "graduation",
            "certification",
            "certificate",
            "completed",
            "finish",
            "finished",
            "preference",
            "prefer",
            "favorite",
            "favourite",
            "best",
            "score",
            "ratio",
            "count",
            "total",
            "number",
            "age",
            "price",
            "cost",
            "weight",
            "height",
            "status",
        }
        episodic_terms = {
            "met",
            "meet",
            "visited",
            "visit",
            "bought",
            "buy",
            "traveled",
            "travel",
            "went",
            "go",
            "called",
            "sent",
            "received",
            "booked",
            "scheduled",
            "started",
            "finished",
            "planned",
            "moved",
            "attended",
            "walked",
            "played",
            "watched",
            "ate",
            "had",
        }
        if slot in {"count", "location", "time"}:
            return "state_fact"
        if vtype in {"number", "location", "time"}:
            return "state_fact"
        if bag.intersection(state_terms):
            return "state_fact"
        if bag.intersection(episodic_terms):
            return "episodic_fact"
        if time_text and location_text:
            return "episodic_fact"
        if subject and value_text:
            return "episodic_fact"
        return "episodic_fact"

    def _span_grounded(self, span: str, content: str) -> bool:
        ratio = self.m._gate.span_overlap_ratio(span, content)
        return ratio >= float(self.m.extractor_span_grounding_min_overlap)

    @staticmethod
    def _infer_value_type(*, value_text: str, time_text: str, location_text: str) -> str:
        value = str(value_text).strip()
        if str(location_text).strip():
            return "location"
        if str(time_text).strip():
            return "time"
        if not value:
            return "text"
        if any(ch.isdigit() for ch in value):
            return "number"
        number_words = {
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
            "twenty",
            "thirty",
            "forty",
            "fifty",
            "sixty",
            "seventy",
            "eighty",
            "ninety",
            "hundred",
            "thousand",
        }
        for tok in re.findall(r"[a-z]+", value.lower()):
            if tok in number_words:
                return "number"
        if len(value.split()) <= 4 and any(ch.isupper() for ch in value):
            return "entity"
        return "text"

    @staticmethod
    def _infer_fact_slot(
        *,
        action: str,
        value_type: str,
        time_text: str,
        location_text: str,
    ) -> str:
        if str(location_text).strip():
            return "location"
        if str(time_text).strip():
            return "time"
        if value_type == "number":
            return "count"
        cleaned = "_".join([tok for tok in str(action).strip().lower().split() if tok])
        return cleaned or "fact"

    @staticmethod
    def _build_canonical_fact(
        *,
        subject: str,
        action: str,
        value_text: str,
        time_text: str,
        location_text: str,
    ) -> str:
        core = " ".join([x for x in [subject, action, value_text] if str(x).strip()]).strip()
        extras: List[str] = []
        if location_text:
            extras.append(f"at {location_text}")
        if time_text:
            extras.append(f"on {time_text}")
        return " ".join([core] + extras).strip()
