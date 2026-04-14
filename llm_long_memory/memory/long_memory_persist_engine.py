"""Event persistence and fact-edge linking engine for LongMemory."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List


class LongMemoryPersistEngine:
    """Persist extracted events and maintain fact-link graph edges."""

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
            for m in re.findall(pat, low):
                token = str(m).strip()
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

    def persist_event(self, event: Dict[str, Any]) -> None:
        subject = str(event.get("subject", "")).strip()
        action = str(event.get("action", "")).strip()
        obj = str(event.get("object", "")).strip()
        event_text = str(event.get("event_text", "")).strip()
        keywords = [str(x).strip().lower() for x in list(event.get("keywords", [])) if str(x).strip()]
        time_text = str(event.get("time", "")).strip()
        location_text = str(event.get("location", "")).strip()
        role = str(event.get("role", "user")).strip().lower() or "user"
        fact_type = str(event.get("fact_type", "event")).strip().lower() or "event"
        confidence = float(event.get("confidence", 0.0) or 0.0)
        source_model = str(event.get("source_model", self.m.extractor_model)).strip()
        raw_span = str(event.get("raw_span", event_text)).strip() or event_text
        source_content = str(event.get("source_content", "")).strip()
        source_date = str(event.get("source_date", "")).strip()

        if not event_text:
            self.m._record_reject("empty_event_text")
            return
        if not keywords:
            keywords = self.m._build_keywords(
                model_keywords=[],
                subject=subject,
                action=action,
                obj=obj,
                event_text=event_text,
                raw_span=raw_span,
                source_content=source_content,
                time_text=time_text,
                location_text=location_text,
            )
        self.m._ingest_event_total += 1
        self.m._ingest_event_accepted += 1
        self.m.current_step += 1

        core = f"{subject}|{action}|{obj}|{time_text}|{location_text}|{role}"
        event_id = self.m._stable_id("event", core)
        fact_key = self.m._build_fact_key(subject, action, obj) or event_id
        if fact_key and time_text:
            dup = self.m.conn.execute(
                """
                SELECT event_id FROM events
                WHERE fact_key=? AND is_latest=1 AND event_id IN (
                    SELECT event_id FROM details WHERE kind='time' AND text=?
                )
                LIMIT 1
                """,
                (fact_key, time_text),
            ).fetchone()
            if dup is not None:
                event_id = str(dup["event_id"])
        emb = self.m._safe_embed(" ".join([event_text, " ".join(keywords), time_text, location_text]).strip())

        self.m.store.upsert_event(
            event_id=event_id,
            fact_key=fact_key,
            subject_action_key=fact_key,
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
            action=action,
            obj=obj,
            time_text=time_text,
            location_text=location_text,
            keywords=keywords,
            raw_span=raw_span,
        )
        superseded_ids = self.m.store.supersede_active_by_fact_key(
            fact_key=fact_key,
            keep_event_id=event_id,
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
        if raw_span:
            self.m.store.insert_detail(
                detail_id=self.m._stable_id("detail", f"{event_id}|evidence|{raw_span}"),
                event_id=event_id,
                kind="evidence",
                text=raw_span,
                current_step=self.m.current_step,
                max_per_event=self.m.details_per_event,
            )
        if time_text:
            time_vals = self._time_tokens(time_text)
            if time_vals:
                self.m.store.insert_detail(
                    detail_id=self.m._stable_id("detail", f"{event_id}|time_start|{time_vals[0]}"),
                    event_id=event_id,
                    kind="time_start",
                    text=time_vals[0],
                    current_step=self.m.current_step,
                    max_per_event=self.m.details_per_event,
                )
                self.m.store.insert_detail(
                    detail_id=self.m._stable_id("detail", f"{event_id}|time_end|{time_vals[-1]}"),
                    event_id=event_id,
                    kind="time_end",
                    text=time_vals[-1],
                    current_step=self.m.current_step,
                    max_per_event=self.m.details_per_event,
                )
            granularity = self._infer_time_granularity(time_text)
            if granularity:
                self.m.store.insert_detail(
                    detail_id=self.m._stable_id(
                        "detail", f"{event_id}|time_granularity|{granularity}"
                    ),
                    event_id=event_id,
                    kind="time_granularity",
                    text=granularity,
                    current_step=self.m.current_step,
                    max_per_event=self.m.details_per_event,
                )
        if source_date:
            self.m.store.insert_detail(
                detail_id=self.m._stable_id("detail", f"{event_id}|source_date|{source_date}"),
                event_id=event_id,
                kind="source_date",
                text=source_date,
                current_step=self.m.current_step,
                max_per_event=self.m.details_per_event,
            )
        if source_content:
            source_excerpt = source_content[: self.m.context_max_chars_per_item]
            self.m.store.insert_detail(
                detail_id=self.m._stable_id("detail", f"{event_id}|source|{source_excerpt}"),
                event_id=event_id,
                kind="source",
                text=source_excerpt,
                current_step=self.m.current_step,
                max_per_event=self.m.details_per_event,
            )
        self.m.store.insert_detail(
            detail_id=self.m._stable_id("detail", f"{event_id}|source_model|{source_model}"),
            event_id=event_id,
            kind="source_model",
            text=source_model,
            current_step=self.m.current_step,
            max_per_event=self.m.details_per_event,
        )
        self.link_fact_edges(
            event_id=event_id,
            fact_key=fact_key,
            subject=subject,
            action=action,
            obj=obj,
            event_keywords=keywords,
        )
        self.m.store.save_current_step(self.m.current_step)

    def link_fact_edges(
        self,
        *,
        event_id: str,
        fact_key: str,
        subject: str,
        action: str,
        obj: str,
        event_keywords: List[str],
    ) -> None:
        if self.m.graph_max_edges_per_event <= 0:
            return
        base_kw = set(self.m._keyword_candidates_from_text(" ".join(event_keywords)))
        subj_n = self.m._normalize_fact_component(subject)
        act_n = self.m._normalize_fact_component(action)
        obj_n = self.m._normalize_fact_component(obj)

        rows = self.m.store.fetch_active_events()
        candidates: List[Dict[str, Any]] = []
        for row in rows:
            other_id = str(row["event_id"])
            if other_id == event_id:
                continue
            other_fact_key = str(row["fact_key"] or "").strip()
            other_subject, other_action, other_object = self.m._split_skeleton_components(
                str(row["skeleton_text"] or "")
            )
            other_subj_n = self.m._normalize_fact_component(other_subject)
            other_act_n = self.m._normalize_fact_component(other_action)
            other_obj_n = self.m._normalize_fact_component(other_object)
            relation = ""
            weight = 0.0
            if fact_key and (other_fact_key == fact_key):
                relation = "extends"
                weight = self.m.graph_edge_weight_same_fact
            elif subj_n and obj_n and (subj_n == other_subj_n) and (obj_n == other_obj_n):
                relation = "same_subject_object"
                weight = self.m.graph_edge_weight_subject_object
            elif subj_n and (subj_n == other_subj_n) and act_n and (act_n == other_act_n):
                relation = "same_subject_action"
                weight = self.m.graph_edge_weight_subject_object
            elif subj_n and (subj_n == other_subj_n):
                relation = "same_subject"
                weight = self.m.graph_edge_weight_same_subject
            elif obj_n and (obj_n == other_obj_n):
                relation = "same_object"
                weight = self.m.graph_edge_weight_same_object
            else:
                try:
                    other_keywords = {
                        str(x).strip().lower()
                        for x in list(json.loads(str(row["keywords"] or "[]")))
                        if str(x).strip()
                    }
                except (TypeError, ValueError, json.JSONDecodeError):
                    other_keywords = set(self.m._tokenize(str(row["keywords"] or "")))
                shared_keywords = len(base_kw.intersection(other_keywords))
                if shared_keywords >= self.m.graph_edge_min_shared_keywords:
                    relation = "shared_keywords"
                    weight = self.m.graph_edge_weight_shared_keywords
            if relation:
                candidates.append({"other_id": other_id, "weight": float(weight), "relation": relation})
        candidates.sort(key=lambda x: float(x["weight"]), reverse=True)
        for item in candidates[: self.m.graph_max_edges_per_event]:
            other_id = str(item["other_id"])
            weight = float(item["weight"])
            relation = str(item["relation"])
            eid = self.m._stable_id("edge", f"{event_id}|{other_id}|{relation}")
            self.m.store.insert_edge(
                edge_id=eid,
                from_event_id=event_id,
                to_event_id=other_id,
                relation=relation,
                weight=weight,
                current_step=self.m.current_step,
            )
