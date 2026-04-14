"""Fact persistence and minimal edge-linking engine for LongMemory."""

from __future__ import annotations

import re
from typing import Any, Dict, List


class LongMemoryPersistEngine:
    """Persist extracted atomic facts and maintain minimal update links."""

    def __init__(self, memory: Any) -> None:
        self.m = memory

    @staticmethod
    def _time_tokens(value: str) -> List[str]:
        text = str(value or "").strip()
        if not text:
            return []
        patterns = (
            r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
            r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b",
            r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?\b",
            r"\b\d{4}\b",
        )
        out: List[str] = []
        low = text.lower()
        for pat in patterns:
            for match in re.findall(pat, low):
                token = str(match).strip()
                if token and token not in out:
                    out.append(token)
        return out

    @staticmethod
    def _infer_time_granularity(time_text: str) -> str:
        low = str(time_text or "").strip().lower()
        if not low:
            return ""
        if re.search(r"\b\d{1,2}:\d{2}\b", low):
            return "time"
        if re.search(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", low):
            return "date"
        if re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\b", low):
            return "month"
        if re.search(r"\b\d{4}\b", low):
            return "year"
        if re.search(r"\b\d+\s*(day|days|week|weeks|month|months|year|years)\b", low):
            return "duration"
        return "unknown"

    def _normalize_slot(self, fact_slot: str, value_type: str, action: str) -> str:
        slot = self.m._normalize_fact_component(fact_slot)
        if slot:
            return slot
        if value_type == "location":
            return "location"
        if value_type == "time":
            return "time"
        if value_type == "number":
            return "count"
        action_norm = self.m._normalize_fact_component(action)
        return action_norm or "fact"

    def _build_atomic_fact_key(
        self,
        *,
        subject: str,
        action: str,
        fact_slot: str,
    ) -> str:
        subject_norm = self.m._normalize_fact_component(subject)
        action_norm = self.m._normalize_fact_component(action)
        slot_norm = self._normalize_slot(fact_slot, "", action)
        return f"{subject_norm}|{action_norm}|{slot_norm}".strip("|")

    def _build_episodic_fact_key(
        self,
        *,
        subject: str,
        action: str,
        value_text: str,
        time_text: str,
        location_text: str,
        raw_span: str,
    ) -> str:
        subject_norm = self.m._normalize_fact_component(subject)
        action_norm = self.m._normalize_fact_component(action)
        value_norm = self.m._normalize_fact_component(value_text)
        time_norm = self.m._normalize_fact_component(time_text)
        location_norm = self.m._normalize_fact_component(location_text)
        parts = [x for x in [subject_norm, action_norm, value_norm, time_norm, location_norm] if x]
        if parts:
            return "|".join(parts)
        return self.m._stable_id("fact", raw_span or f"{subject}|{action}|{value_text}")

    def _build_skeleton_text(
        self,
        *,
        subject: str,
        action: str,
        value_text: str,
        canonical_fact: str,
    ) -> str:
        if str(canonical_fact).strip():
            return str(canonical_fact).strip()
        return " ".join([x for x in [subject, action, value_text] if str(x).strip()]).strip()

    def persist_event(self, event: Dict[str, Any]) -> None:
        subject = str(event.get("subject", "")).strip()
        action = str(event.get("action", "")).strip()
        value_text = str(event.get("value", event.get("object", ""))).strip()
        canonical_fact = str(event.get("canonical_fact", event.get("event_text", ""))).strip()
        event_text = self._build_skeleton_text(
            subject=subject,
            action=action,
            value_text=value_text,
            canonical_fact=canonical_fact,
        )
        keywords = [str(x).strip().lower() for x in list(event.get("keywords", [])) if str(x).strip()]
        time_text = str(event.get("time", "")).strip()
        location_text = str(event.get("location", "")).strip()
        value_type = str(event.get("value_type", "")).strip().lower() or "text"
        fact_slot = self._normalize_slot(
            str(event.get("fact_slot", "")).strip().lower(),
            value_type,
            action,
        )
        role = str(event.get("role", "user")).strip().lower() or "user"
        fact_type = str(event.get("fact_type", "episodic_fact")).strip().lower() or "episodic_fact"
        if fact_type in {"state", "state_fact", "state-fact", "latest", "current"}:
            fact_type = "state_fact"
        else:
            fact_type = "episodic_fact"
        confidence = float(event.get("confidence", 0.0) or 0.0)
        source_model = str(event.get("source_model", self.m.extractor_model)).strip()
        raw_span = str(event.get("raw_span", event_text)).strip() or event_text
        source_content = str(event.get("source_content", "")).strip()
        source_date = str(event.get("source_date", "")).strip()

        if not keywords:
            keywords = self.m._build_keywords(
                model_keywords=[],
                subject=subject,
                action=action,
                obj=value_text,
                event_text=event_text,
                raw_span=raw_span,
                source_content=source_content,
                time_text=time_text,
                location_text=location_text,
            )

        value_text = self.m._text.normalize_value_text(
            value_text,
            fact_slot=fact_slot,
            value_type=value_type,
        )
        if not canonical_fact:
            canonical_fact = self._build_skeleton_text(
                subject=subject,
                action=action,
                value_text=value_text,
                canonical_fact="",
            )
        event_text = self._build_skeleton_text(
            subject=subject,
            action=action,
            value_text=value_text,
            canonical_fact=canonical_fact,
        )
        if not event_text:
            self.m._record_reject("empty_event_text")
            return

        self.m._ingest_event_total += 1
        self.m._ingest_event_accepted += 1
        self.m.current_step += 1

        if fact_type == "state_fact":
            fact_key = self._build_atomic_fact_key(
                subject=subject,
                action=action,
                fact_slot=fact_slot,
            ) or self.m._stable_id("fact", event_text)
        else:
            fact_key = self._build_episodic_fact_key(
                subject=subject,
                action=action,
                value_text=value_text,
                time_text=time_text,
                location_text=location_text,
                raw_span=raw_span,
            )
        event_id = self.m._stable_id(
            "event",
            "|".join(
                [
                    fact_key,
                    value_text,
                    time_text,
                    location_text,
                    raw_span,
                    role,
                ]
            ),
        )

        emb = self.m._safe_embed(
            " ".join(
                [
                    canonical_fact,
                    subject,
                    action,
                    value_text,
                    value_type,
                    fact_slot,
                    time_text,
                    location_text,
                    " ".join(keywords),
                ]
            ).strip()
        )

        self.m.store.upsert_event(
            event_id=event_id,
            fact_key=fact_key,
            subject_action_key=f"{self.m._normalize_fact_component(subject)}|{self.m._normalize_fact_component(action)}",
            fact_type=fact_type,
            skeleton_text=event_text,
            skeleton_embedding=emb,
            keywords=keywords,
            role=role,
            boundary_flag=0,
            extract_confidence=confidence,
            source_model=source_model,
            raw_span=raw_span,
            current_step=self.m.current_step,
        )

        self.m._upsert_event_nodes(
            event_id=event_id,
            subject=subject,
            predicate=action,
            value=value_text,
            time_text=time_text,
            location_text=location_text,
            keywords=keywords,
            raw_span=raw_span,
        )

        if fact_type == "state_fact":
            superseded_ids = self.m.store.supersede_active_by_fact_key(
                fact_key=fact_key,
                keep_event_id=event_id,
                new_value_text=value_text,
                new_time_text=time_text,
                new_raw_span=raw_span,
                min_evidence_overlap=self.m.supersede_min_evidence_overlap,
                current_step=self.m.current_step,
            )
            for old_id in superseded_ids:
                edge_id = self.m._stable_id("edge", f"{old_id}|{event_id}|updates")
                self.m.store.insert_edge(
                    edge_id=edge_id,
                    from_event_id=old_id,
                    to_event_id=event_id,
                    relation="updates",
                    weight=1.0,
                    current_step=self.m.current_step,
                )

        self._insert_detail(event_id=event_id, kind="subject", text=subject)
        self._insert_detail(event_id=event_id, kind="predicate", text=action)
        self._insert_detail(event_id=event_id, kind="value", text=value_text)
        self._insert_detail(event_id=event_id, kind="value_type", text=value_type)
        self._insert_detail(event_id=event_id, kind="fact_slot", text=fact_slot)
        self._insert_detail(event_id=event_id, kind="canonical_fact", text=canonical_fact or event_text)
        self._insert_detail(event_id=event_id, kind="evidence", text=raw_span)
        self._insert_detail(event_id=event_id, kind="source_model", text=source_model)
        if location_text:
            self._insert_detail(event_id=event_id, kind="location", text=location_text)
        if time_text:
            self._insert_detail(event_id=event_id, kind="time", text=time_text)
            time_vals = self._time_tokens(time_text)
            if time_vals:
                self._insert_detail(event_id=event_id, kind="time_start", text=time_vals[0])
                self._insert_detail(event_id=event_id, kind="time_end", text=time_vals[-1])
            granularity = self._infer_time_granularity(time_text)
            if granularity:
                self._insert_detail(event_id=event_id, kind="time_granularity", text=granularity)
        if source_date:
            self._insert_detail(event_id=event_id, kind="source_date", text=source_date)
        if source_content:
            self._insert_detail(
                event_id=event_id,
                kind="source",
                text=source_content[: self.m.context_max_chars_per_item],
            )

        if fact_type == "state_fact":
            self.link_fact_edges(event_id=event_id, fact_key=fact_key)
        self.m.store.save_current_step(self.m.current_step)

    def _insert_detail(self, *, event_id: str, kind: str, text: str) -> None:
        value = str(text).strip()
        if not value:
            return
        self.m.store.insert_detail(
            detail_id=self.m._stable_id("detail", f"{event_id}|{kind}|{value}"),
            event_id=event_id,
            kind=kind,
            text=value,
            current_step=self.m.current_step,
            max_per_event=self.m.details_per_event,
        )

    def link_fact_edges(self, *, event_id: str, fact_key: str) -> None:
        """Keep event-level edges minimal: only same-fact continuity links."""
        if self.m.graph_max_edges_per_event <= 0 or (not fact_key):
            return
        linked = 0
        for row in self.m.store.fetch_superseded_events(self.m.history_max_candidates):
            other_id = str(row["event_id"] or "").strip()
            other_fact_key = str(row["fact_key"] or "").strip()
            if (not other_id) or other_id == event_id or other_fact_key != fact_key:
                continue
            edge_id = self.m._stable_id("edge", f"{other_id}|{event_id}|extends")
            self.m.store.insert_edge(
                edge_id=edge_id,
                from_event_id=other_id,
                to_event_id=event_id,
                relation="extends",
                weight=0.6,
                current_step=self.m.current_step,
            )
            linked += 1
            if linked >= self.m.graph_max_edges_per_event:
                break
