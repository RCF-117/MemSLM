"""Fact persistence and minimal edge-linking engine for LongMemory."""

from __future__ import annotations

import json
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

    @staticmethod
    def _normalize_space(text: str) -> str:
        return " ".join(str(text or "").split()).strip()

    def _extract_numeric_value(self, value_text: str) -> str:
        value = str(value_text or "").strip()
        if not value:
            return ""
        match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
        if match:
            return str(match.group(0))
        word_to_num = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
            "eleven": "11",
            "twelve": "12",
        }
        low = value.lower()
        for word, num in word_to_num.items():
            if re.search(rf"\b{word}\b", low):
                return num
        return ""

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
        raw_value = str(event.get("raw_value", event.get("value", event.get("object", "")))).strip()
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
        answer_span_raw = (
            str(event.get("answer_span_raw", "")).strip()
            or raw_span
            or event_text
        )
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
        value_text_norm = self._normalize_space(value_text).lower()
        value_num_norm = self._extract_numeric_value(value_text)
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
        self._insert_detail(event_id=event_id, kind="raw_value", text=raw_value)
        self._insert_detail(event_id=event_id, kind="value", text=value_text)
        self._insert_detail(event_id=event_id, kind="value_norm", text=value_text_norm)
        if value_num_norm:
            self._insert_detail(event_id=event_id, kind="value_number", text=value_num_norm)
        self._insert_detail(event_id=event_id, kind="value_type", text=value_type)
        self._insert_detail(event_id=event_id, kind="fact_slot", text=fact_slot)
        self._insert_detail(event_id=event_id, kind="canonical_fact", text=canonical_fact or event_text)
        self._insert_detail(event_id=event_id, kind="evidence", text=raw_span)
        self._insert_detail(event_id=event_id, kind="answer_span_raw", text=answer_span_raw)
        self._insert_detail(event_id=event_id, kind="source_model", text=source_model)
        if location_text:
            self._insert_detail(event_id=event_id, kind="location", text=location_text)
        if time_text:
            self._insert_detail(event_id=event_id, kind="time", text=time_text)
            time_vals = self._time_tokens(time_text)
            if time_vals:
                self._insert_detail(event_id=event_id, kind="time_start", text=time_vals[0])
                self._insert_detail(event_id=event_id, kind="time_end", text=time_vals[-1])
                for token in time_vals:
                    self._insert_detail(event_id=event_id, kind="time_norm_token", text=token)
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

        self.link_fact_edges(
            event_id=event_id,
            fact_key=fact_key,
            subject_action_key=f"{self.m._normalize_fact_component(subject)}|{self.m._normalize_fact_component(action)}",
            subject=subject,
            action=action,
            obj=value_text,
            event_keywords=keywords,
            raw_span=raw_span,
            time_text=time_text,
            location_text=location_text,
            source_date=source_date,
            fact_type=fact_type,
        )
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

    @staticmethod
    def _keyword_tokens(value: str) -> set[str]:
        try:
            return {
                str(x).strip().lower()
                for x in list(json.loads(str(value or "[]")))
                if str(x).strip()
            }
        except (TypeError, ValueError, json.JSONDecodeError):
            return set(re.findall(r"[a-z0-9]+", str(value).lower()))

    def link_fact_edges(
        self,
        *,
        event_id: str,
        fact_key: str,
        subject_action_key: str,
        subject: str,
        action: str,
        obj: str,
        event_keywords: List[str],
        raw_span: str,
        time_text: str,
        location_text: str,
        source_date: str,
        fact_type: str,
    ) -> None:
        """Connect sparse factual relations between event sentences."""
        if self.m.graph_max_edges_per_event <= 0:
            return
        rows = self.m.store.fetch_recent_events(
            self.m.history_max_candidates,
            exclude_event_id=event_id,
        )
        subject_norm = str(subject_action_key or "").split("|", 1)[0].strip()
        seen_targets: set[str] = set()
        linked = 0

        def add_edge(*, from_id: str, relation: str, weight: float) -> None:
            nonlocal linked
            if linked >= self.m.graph_max_edges_per_event:
                return
            rid = str(from_id or "").strip()
            if (not rid) or rid == event_id:
                return
            dedup_key = f"{rid}|{relation}"
            if dedup_key in seen_targets:
                return
            seen_targets.add(dedup_key)
            edge_id = self.m._stable_id("edge", f"{rid}|{event_id}|{relation}")
            self.m.store.insert_edge(
                edge_id=edge_id,
                from_event_id=rid,
                to_event_id=event_id,
                relation=relation,
                weight=float(weight),
                current_step=self.m.current_step,
            )
            linked += 1

        for row in rows:
            other_id = str(row["event_id"] or "").strip()
            other_fact_key = str(row["fact_key"] or "").strip()
            if (not other_id) or other_id == event_id:
                continue
            if fact_type == "state_fact":
                if fact_key and other_fact_key and fact_key == other_fact_key:
                    add_edge(from_id=other_id, relation="updates", weight=1.0)
                    if linked >= self.m.graph_max_edges_per_event:
                        break

            if self.m.graph_same_subject_enabled and subject_norm:
                other_subject = str(row["subject_action_key"] or "").split("|", 1)[0].strip()
                if other_subject and other_subject == subject_norm:
                    add_edge(
                        from_id=other_id,
                        relation="same_subject",
                        weight=float(self.m.graph_same_subject_weight),
                    )
                    if linked >= self.m.graph_max_edges_per_event:
                        break

            if self.m.graph_co_source_enabled and str(source_date).strip():
                src_row = self.m.store.conn.execute(
                    """
                    SELECT text
                    FROM details
                    WHERE event_id=? AND kind='source_date'
                    ORDER BY created_step DESC
                    LIMIT 1
                    """,
                    (other_id,),
                ).fetchone()
                other_source_date = str(src_row["text"]).strip() if src_row else ""
                if other_source_date and other_source_date == str(source_date).strip():
                    add_edge(
                        from_id=other_id,
                        relation="co_source",
                        weight=float(self.m.graph_co_source_weight),
                    )
                    if linked >= self.m.graph_max_edges_per_event:
                        break
