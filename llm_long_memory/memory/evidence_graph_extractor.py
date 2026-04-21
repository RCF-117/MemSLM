"""LLM-backed fixed-schema claim extraction from filtered evidence packs."""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

from llm_long_memory.llm.ollama_client import ollama_generate_with_retry
from llm_long_memory.memory.long_memory_json_utils import (
    extract_first_json_block,
    safe_json_loads_relaxed,
)


class EvidenceGraphExtractor:
    """Extract fixed-schema claims from filtered evidence with an 8B model."""

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
        self.force_json_output = bool(cfg.get("force_json_output", True))
        self.think = bool(cfg.get("think", False))
        self.max_batches = max(1, int(cfg.get("extractor_max_batches", 3)))

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

    def _prompt(
        self,
        *,
        filtered_pack: Dict[str, Any],
        batch: Sequence[Dict[str, Any]],
    ) -> str:
        lines: List[str] = []
        for item in batch:
            evidence_id = str(item.get("evidence_id", "")).strip()
            channel = str(item.get("channel", "")).strip()
            session_date = str(item.get("session_date", "")).strip()
            prefix = f"{evidence_id} [{channel}"
            if session_date:
                prefix += f", {session_date}"
            prefix += "]"
            lines.append(f"{prefix}: {self._normalize_space(str(item.get('text', '')))}")
        schema = {
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
        return (
            "You are extracting question-scoped memory claims from retrieved evidence.\n"
            "Only output claims explicitly supported by the evidence lines.\n"
            "Do not infer missing values. Preserve conflicting claims separately.\n"
            "Prefer complete factual phrases over single words.\n"
            "Leave compare_role empty unless the question explicitly compares two options.\n"
            "Use state_snapshot for changeable attributes, event_record for dated actions, "
            "fact_statement for stable facts.\n"
            "Return strict JSON only with this shape:\n"
            f"{json.dumps(schema, ensure_ascii=True)}\n"
            f"Question: {self._normalize_space(str(filtered_pack.get('query', '')))}\n"
            f"Intent: {self._normalize_space(str(filtered_pack.get('intent', 'lookup')))}\n"
            f"Answer type: {self._normalize_space(str(filtered_pack.get('answer_type', 'factoid')))}\n"
            f"Focus phrases: {json.dumps(list(filtered_pack.get('focus_phrases', [])), ensure_ascii=True)}\n"
            f"Max claims: {self.max_claims_per_batch}\n"
            "Evidence lines:\n"
            + "\n".join(lines)
        )

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
        return {
            "claim_id": fallback_claim_id,
            "claim_type": self._valid_claim_type(raw_claim.get("claim_type", "")),
            "subject": subject,
            "predicate": predicate,
            "value": value,
            "time_anchor": self._normalize_space(str(raw_claim.get("time_anchor", ""))),
            "state_key": self._normalize_space(str(raw_claim.get("state_key", ""))),
            "status": self._valid_status(raw_claim.get("status", "")),
            "modality": self._valid_modality(raw_claim.get("modality", "")),
            "compare_role": self._valid_compare_role(raw_claim.get("compare_role", "")),
            "numeric_value": self._normalize_space(str(raw_claim.get("numeric_value", ""))),
            "unit": self._normalize_space(str(raw_claim.get("unit", ""))),
            "confidence": round(confidence_value, 4),
            "evidence_ids": evidence_ids,
            "verbatim_span": self._normalize_space(str(raw_claim.get("verbatim_span", ""))),
        }

    def extract_claims(self, filtered_pack: Dict[str, Any]) -> Dict[str, Any]:
        selected = (
            list(filtered_pack.get("core_evidence", []))
            + list(filtered_pack.get("supporting_evidence", []))
            + list(filtered_pack.get("conflict_evidence", []))
        )
        if not self.enabled:
            return {
                "enabled": False,
                "model": self.model,
                "claims": [],
                "raw_batches": [],
                "stats": {"selected_evidence": len(selected), "batches": 0},
            }
        if not selected:
            return {
                "enabled": True,
                "model": self.model,
                "claims": [],
                "raw_batches": [],
                "stats": {"selected_evidence": 0, "batches": 0},
            }

        valid_ids = [str(x.get("evidence_id", "")).strip() for x in selected if str(x.get("evidence_id", "")).strip()]
        allow_compare_role = str(filtered_pack.get("answer_type", "")).strip().lower() == "temporal_comparison"
        batches: List[List[Dict[str, Any]]] = []
        cursor = 0
        while cursor < len(selected) and len(batches) < self.max_batches:
            batches.append(selected[cursor : cursor + self.batch_size])
            cursor += self.batch_size

        raw_batches: List[Dict[str, Any]] = []
        claims: List[Dict[str, Any]] = []
        seen_keys = set()
        claim_counter = 0

        for batch_idx, batch in enumerate(batches, start=1):
            prompt = self._prompt(filtered_pack=filtered_pack, batch=batch)
            raw = self._call_model(prompt)
            parsed = safe_json_loads_relaxed(extract_first_json_block(raw))
            batch_claims = parsed.get("claims", []) if isinstance(parsed, dict) else []
            if not isinstance(batch_claims, list):
                batch_claims = []
            normalized_batch: List[Dict[str, Any]] = []
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
                normalized_batch.append(claim)
                claims.append(claim)
            raw_batches.append(
                {
                    "batch_index": batch_idx,
                    "evidence_ids": [str(x.get("evidence_id", "")) for x in batch],
                    "raw_response": raw,
                    "claim_count": len(normalized_batch),
                }
            )

        return {
            "enabled": True,
            "model": self.model,
            "claims": claims,
            "raw_batches": raw_batches,
            "stats": {
                "selected_evidence": len(selected),
                "batches": len(batches),
                "claims": len(claims),
            },
        }
