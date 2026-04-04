"""Ingestion helper for LongMemory."""

from __future__ import annotations

from typing import Any, Dict, List

from llm_long_memory.processing.graph_builder import extract_events_from_message


Message = Dict[str, Any]


class LongMemoryIngestor:
    """Handle event extraction, acceptance filtering, and store upserts."""

    def __init__(self, owner: Any) -> None:
        self.owner = owner

    def should_accept_event(
        self,
        role: str,
        subject: str,
        action: str,
        obj: str,
        sentence: str,
        confidence: float,
        keywords: List[str],
        entity_count: int,
        has_time: bool,
        has_location: bool,
    ) -> tuple[bool, str]:
        o = self.owner
        if role not in o.ingest_allow_roles:
            return False, "role_not_allowed"

        lowered_sentence = str(sentence).strip().lower()
        for phrase in o.ingest_reject_phrases:
            if phrase and phrase in lowered_sentence:
                return False, "rejected_phrase"

        if o._ingest_event_accepted < o.ingest_bootstrap_min_accepts:
            has_core_fields = bool(o._normalize_phrase(subject)) and bool(
                o._normalize_phrase(action)
            ) and bool(o._normalize_phrase(obj))
            if has_core_fields and confidence >= o.ingest_bootstrap_min_confidence:
                norm_subject = o._canonical_subject(subject, role)
                norm_action = o._normalize_phrase(action)
                if (
                    (not o.ingest_bootstrap_allow_generic_action)
                    and (norm_action in o.ingest_generic_actions)
                ):
                    return False, "generic_action_disabled"
                if (
                    norm_subject in o.ingest_reject_subjects
                    and norm_action in o.ingest_generic_actions
                ):
                    return False, "generic_subject_action"
                return True, "bootstrap_accept"

        min_conf = o.ingest_assistant_min_confidence if role == "assistant" else o.ingest_min_confidence
        if confidence < min_conf:
            return False, "low_confidence"
        if len(keywords) < o.ingest_min_keywords:
            return False, "few_keywords"
        if len(o._tokenize(obj)) < o.ingest_min_object_tokens:
            return False, "short_object"
        if int(entity_count) < o.ingest_min_entity_count:
            return False, "few_entities"
        sentence_tokens = o._tokenize(sentence)
        if len(sentence_tokens) < o.ingest_min_sentence_tokens:
            return False, "short_sentence"
        if len(sentence_tokens) > o.ingest_max_sentence_tokens:
            return False, "long_sentence"
        if o.ingest_require_time_or_location and (not has_time) and (not has_location):
            return False, "missing_time_or_location"

        norm_subject = o._canonical_subject(subject, role)
        norm_action = o._normalize_phrase(action)
        if norm_subject in o.ingest_reject_subjects and norm_action in o.ingest_generic_actions:
            return False, "generic_subject_action"
        if (not o.ingest_allow_generic_action) and (norm_action in o.ingest_generic_actions):
            return False, "generic_action_disabled"
        return True, "accepted"

    def process_message(self, message: Message) -> None:
        o = self.owner
        events = extract_events_from_message(message, o.event_cfg)
        if not events:
            return

        for ev in events:
            o._ingest_event_total += 1
            o.current_step += 1

            subject = str(ev.get("subject", "")).strip()
            action = str(ev.get("action", "states")).strip() or "states"
            obj = str(ev.get("object", "")).strip()
            location = str(ev.get("location", "")).strip()
            time_value = str(ev.get("time", "")).strip()
            sentence = str(ev.get("sentence", "")).strip()
            role = str(ev.get("role", "user")).strip().lower() or "user"
            confidence = float(ev.get("confidence", 0.0) or 0.0)
            fact_type = str(ev.get("fact_type", "event")).strip().lower() or "event"
            entity_count = int(ev.get("entity_count", 0) or 0)
            has_time = bool(int(str(ev.get("has_time", "0"))))
            has_location = bool(int(str(ev.get("has_location", "0"))))

            normalized_keywords = o._tokenize(" ".join([subject, action, obj, location, time_value]))
            accepted, reason = self.should_accept_event(
                role=role,
                subject=subject,
                action=action,
                obj=obj,
                sentence=sentence,
                confidence=confidence,
                keywords=normalized_keywords,
                entity_count=entity_count,
                has_time=has_time,
                has_location=has_location,
            )
            if not accepted:
                o._ingest_event_rejected += 1
                o._ingest_reject_reasons[reason] = o._ingest_reject_reasons.get(reason, 0) + 1
                continue

            subject_key = o._canonical_subject(subject, role)
            action_key = o._normalize_phrase(action)
            object_key = o._canonical_object_key(obj)
            if not subject_key or not action_key or not object_key:
                o._ingest_event_rejected += 1
                o._ingest_reject_reasons["empty_key_component"] = (
                    o._ingest_reject_reasons.get("empty_key_component", 0) + 1
                )
                continue

            skeleton_text = f"{subject_key} {action_key} {object_key}".strip()
            o._ingest_event_accepted += 1

            subject_action_key = f"{subject_key}|{action_key}"
            fact_key = f"{subject_key}|{action_key}|{object_key}"
            event_id = o._stable_id("event", fact_key)
            keywords = normalized_keywords
            skeleton_embedding = o._safe_embed(skeleton_text)

            o.store.upsert_event(
                event_id=event_id,
                fact_key=fact_key,
                subject_action_key=subject_action_key,
                fact_type=fact_type,
                skeleton_text=skeleton_text,
                skeleton_embedding=skeleton_embedding,
                keywords=keywords,
                role=role,
                current_step=o.current_step,
            )

            for kind, value in (
                ("sentence", sentence),
                ("location", location),
                ("time", time_value),
                ("object", obj),
            ):
                if kind not in o.ingest_detail_kinds:
                    continue
                clean = str(value).strip()
                if not clean or len(clean) < o.ingest_detail_min_chars:
                    continue
                detail_id = o._stable_id("detail", f"{event_id}|{kind}|{clean}")
                o.store.insert_detail(
                    detail_id=detail_id,
                    event_id=event_id,
                    kind=kind,
                    text=clean,
                    current_step=o.current_step,
                    max_per_event=o.ingest_max_details_per_event,
                )

        if o.consolidation_every_updates > 0 and (
            o.current_step % o.consolidation_every_updates == 0
        ):
            o.store.resolve_conflicts()

        if o.forgetting_every_updates > 0 and (
            o.current_step % o.forgetting_every_updates == 0
        ):
            o.store.apply_forgetting(
                current_step=o.current_step,
                forget_decay=o.forget_decay,
                forget_threshold=o.forget_threshold,
                forget_min_age_steps=o.forget_min_age_steps,
                max_events=o.max_events,
            )

        o.store.save_current_step(o.current_step)
        o.store.commit()
