"""Structured event extraction engine for LongMemory."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from llm_long_memory.llm.ollama_client import ollama_generate_with_retry
from llm_long_memory.utils.logger import logger


class LongMemoryExtractor:
    """Run structured extraction using LongMemory config/runtime state."""

    def __init__(self, memory: Any) -> None:
        self.m = memory

    def extract_events_structured(self, message: Dict[str, Any], force: bool = False) -> List[Dict[str, Any]]:
        """Extract minimal event skeletons using local model."""
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
        if (not self.m.extractor_compact_retry_enabled) or (len(trimmed) <= self.m.extractor_compact_window_chars):
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
            "Task: extract factual memory events from one message.\n"
            "Output: JSON only, valid UTF-8, no markdown.\n"
            "Schema (strict):\n"
            '{"events":[{"subject":"","action":"","object":"","time":"","location":"","evidence_span":"","event_text":"","keywords":[],"confidence":0.0}]}\n'
            "Hard rules:\n"
            "- Keep only completed or observed facts.\n"
            "- Keep personal preferences and stated choices if they are user-specific facts.\n"
            "- Exclude pure instruction/chitchat/meta text without stable facts.\n"
            "- evidence_span must be an exact short quote from the message.\n"
            "- event_text must be concise and factual.\n"
            "- confidence range [0,1].\n"
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
                subject = str(item.get("subject", "")).strip()
                action = str(item.get("action", "")).strip()
                obj = str(item.get("object", "")).strip()
                subject = self.m._entity_norm.normalize_subject(subject, role)
                obj = self.m._entity_norm.normalize_object(obj, role)
                event_text = str(item.get("event_text", "")).strip()
                raw_span = str(item.get("evidence_span", "")).strip()
                if not raw_span:
                    raw_span = str(item.get("raw_span", "")).strip()
                if subject and action:
                    event_text = f"{subject} | {action} | {obj}".strip()
                elif (not event_text) and subject and action and obj:
                    event_text = f"{subject} | {action} | {obj}"
                if not event_text:
                    continue
                if raw_span and (not self._span_grounded(raw_span, trimmed)):
                    continue
                confidence = float(item.get("confidence", 0.0) or 0.0)
                confidence = max(0.0, min(1.0, confidence))
                raw_keywords = item.get("keywords", [])
                if isinstance(raw_keywords, list):
                    model_keywords = [str(x).strip().lower() for x in raw_keywords if str(x).strip()]
                else:
                    model_keywords = self.m._tokenize(str(raw_keywords))
                time_text = str(item.get("time", "")).strip()
                location_text = str(item.get("location", "")).strip()
                keywords = self.m._build_keywords(
                    model_keywords=model_keywords,
                    subject=subject,
                    action=action,
                    obj=obj,
                    event_text=event_text,
                    raw_span=raw_span,
                    source_content=trimmed,
                    time_text=time_text,
                    location_text=location_text,
                )
                if self.m.gating_enabled:
                    accepted, reason, _score = self.m._gate.evaluate(
                        subject=subject,
                        action=action,
                        obj=obj,
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
                            obj=obj,
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
                        obj=obj,
                        confidence=confidence,
                        evidence_span=raw_span,
                        source_content=trimmed,
                    ):
                        self.m._record_reject("legacy_gate_reject")
                        self.m._stage_rejected_event(
                            reason="legacy_gate_reject",
                            subject=subject,
                            action=action,
                            obj=obj,
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
                        "object": obj,
                        "event_text": event_text,
                        "keywords": keywords,
                        "time": time_text,
                        "location": location_text,
                        "confidence": confidence,
                        "role": role,
                        "source_model": self.m.extractor_model,
                        "raw_span": raw_span or event_text,
                        "source_content": trimmed,
                        "fact_type": self.m._classify_fact_type(
                            action=action,
                            event_text=event_text,
                            keywords=keywords,
                        ),
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
        required = ["subject", "action", "object", "time", "location", "event_text", "confidence"]
        for key in required:
            if key not in item:
                return False
        keywords = item.get("keywords", [])
        if (not isinstance(keywords, list)) and (not isinstance(keywords, str)):
            return False
        try:
            float(item.get("confidence", 0.0) or 0.0)
        except (ValueError, TypeError):
            return False
        return True

    def _span_grounded(self, span: str, content: str) -> bool:
        ratio = self.m._gate.span_overlap_ratio(span, content)
        return ratio >= float(self.m.extractor_span_grounding_min_overlap)
