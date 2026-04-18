"""Minimal long-term memory: 4B extraction + SQLite storage + lightweight retrieval."""

from __future__ import annotations

import hashlib
import re
import urllib.request
from typing import Any, Dict, List, Tuple

import numpy as np

from llm_long_memory.memory.long_memory_anchor_utils import sentence_feature_score
from llm_long_memory.memory.long_memory_extractor import LongMemoryExtractor
from llm_long_memory.memory.long_memory_graph_nodes import upsert_event_nodes
from llm_long_memory.llm.ollama_client import ollama_generate_with_retry
from llm_long_memory.memory.long_memory_entity_norm import LongMemoryEntityNormalizer
from llm_long_memory.memory.long_memory_gate import LongMemoryGate, LongMemoryGateConfig
from llm_long_memory.memory.long_memory_json_utils import (
    extract_first_json_block,
    safe_json_loads_relaxed,
)
from llm_long_memory.memory.long_memory_query_engine import LongMemoryQueryEngine
from llm_long_memory.memory.long_memory_pack_utils import build_evidence_packs
from llm_long_memory.memory.long_memory_persist_engine import LongMemoryPersistEngine
from llm_long_memory.memory.long_memory_store import LongMemoryStore
from llm_long_memory.memory.long_memory_text_utils import LongMemoryTextUtils
from llm_long_memory.memory.memory_manager_utils import dedup_chunks_keep_best
from llm_long_memory.utils.embedding import embed
from llm_long_memory.utils.helpers import load_config
from llm_long_memory.utils.logger import logger


Message = Dict[str, Any]


class LongMemory:
    """Config-driven minimal long memory module."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or load_config()
        self.cfg = dict(self.config["memory"]["long_memory"])
        llm_cfg = dict(self.config["llm"])
        extractor_cfg = dict(self.cfg.get("extractor", {}))
        gating_cfg = dict(extractor_cfg.get("gating", {}))
        norm_cfg = dict(extractor_cfg.get("entity_normalization", {}))
        fact_filter_cfg = dict(extractor_cfg.get("fact_filter", {}))
        retrieval_cfg = dict(self.cfg.get("retrieval_scoring", {}))
        graph_cfg = dict(self.cfg.get("graph", {}))
        node_graph_cfg = dict(self.cfg.get("node_graph", {}))
        query_struct_cfg = dict(self.cfg.get("query_struct", {}))

        self.enabled = bool(self.cfg.get("enabled", False))

        self.embedding_dim = int(self.config["embedding"]["dim"])
        self.details_per_event = int(self.cfg.get("details_per_event", 0))
        self.context_max_items = int(self.cfg.get("context_max_items", 4))
        self.context_max_chars_per_item = int(self.cfg.get("context_max_chars_per_item", 220))
        self.context_evidence_max_chars = int(self.cfg.get("context_evidence_max_chars", 240))
        self.context_source_max_chars = int(self.cfg.get("context_source_max_chars", 180))
        self.context_include_source = bool(self.cfg.get("context_include_source", True))
        self.context_chain_enabled = bool(self.cfg.get("context_chain_enabled", True))
        self.context_chain_size = int(self.cfg.get("context_chain_size", 3))
        self.context_chain_max_chars = int(self.cfg.get("context_chain_max_chars", 900))
        self.retrieval_top_k = int(self.cfg.get("retrieval_top_k", 6))
        self.retrieval_min_score = float(self.cfg.get("retrieval_min_score", 0.0))
        offline_graph_cfg = dict(self.cfg.get("offline_graph", {}))

        self.retrieval_use_embedding = bool(retrieval_cfg.get("use_embedding", True))
        self.lexical_weight = float(retrieval_cfg.get("lexical_weight", 0.5))
        self.embedding_weight = float(retrieval_cfg.get("embedding_weight", 0.4))
        self.embedding_fallback_weight = float(
            retrieval_cfg.get("embedding_fallback_weight", self.embedding_weight)
        )
        self.value_overlap_weight = float(retrieval_cfg.get("value_overlap_weight", 0.0))
        self.evidence_overlap_weight = float(retrieval_cfg.get("evidence_overlap_weight", 0.0))
        self.value_overlap_min_tokens = int(retrieval_cfg.get("value_overlap_min_tokens", 1))
        self.evidence_overlap_min_tokens = int(
            retrieval_cfg.get("evidence_overlap_min_tokens", 1)
        )
        self.temporal_filter_enabled = bool(retrieval_cfg.get("temporal_filter_enabled", True))
        self.temporal_query_time_boost = float(retrieval_cfg.get("temporal_query_time_boost", 1.15))
        self.temporal_query_no_time_penalty = float(
            retrieval_cfg.get("temporal_query_no_time_penalty", 0.75)
        )
        self.keyword_primary_min_overlap = int(retrieval_cfg.get("keyword_primary_min_overlap", 1))
        self.keyword_density_weight = float(retrieval_cfg.get("keyword_density_weight", 0.3))
        self.recency_weight = float(retrieval_cfg.get("recency_weight", 0.1))
        self.history_enabled = bool(retrieval_cfg.get("history_enabled", True))
        self.history_weight = float(retrieval_cfg.get("history_weight", 0.7))
        self.history_max_candidates = int(retrieval_cfg.get("history_max_candidates", 128))
        self.retrieval_include_hints = bool(retrieval_cfg.get("include_hints", False))
        self.node_graph_enabled = bool(node_graph_cfg.get("enabled", True))
        self.node_boost_weight = float(node_graph_cfg.get("node_boost_weight", 0.2))
        self.node_edge_boost_weight = float(node_graph_cfg.get("node_edge_boost_weight", 0.12))
        self.node_context_per_event = int(node_graph_cfg.get("context_nodes_per_event", 6))
        self.node_keyword_limit = int(node_graph_cfg.get("keyword_nodes_limit", 8))
        self.graph_neighbor_limit = int(graph_cfg.get("neighbor_limit", 4))
        self.graph_neighbor_weight = float(graph_cfg.get("neighbor_weight", 0.15))
        self.graph_max_edges_per_event = int(graph_cfg.get("max_edges_per_event", 6))
        self.graph_same_subject_enabled = bool(graph_cfg.get("same_subject_enabled", True))
        self.graph_same_subject_weight = float(graph_cfg.get("same_subject_weight", 0.25))
        self.graph_co_source_enabled = bool(graph_cfg.get("co_source_enabled", True))
        self.graph_co_source_weight = float(graph_cfg.get("co_source_weight", 0.18))
        self.offline_pack_enabled = bool(offline_graph_cfg.get("evidence_pack_enabled", True))
        self.offline_pack_max_packs = int(offline_graph_cfg.get("evidence_pack_max_packs", 4))
        self.offline_pack_max_chars = int(offline_graph_cfg.get("evidence_pack_max_chars", 220))
        self.offline_pack_min_sentence_chars = int(
            offline_graph_cfg.get("evidence_pack_min_sentence_chars", 24)
        )
        self.offline_pack_top_sentences_per_chunk = int(
            offline_graph_cfg.get("evidence_pack_top_sentences_per_chunk", 3)
        )
        self.offline_use_full_chunk_text = bool(
            offline_graph_cfg.get("use_full_chunk_text", True)
        )
        self.offline_user_priority_enabled = bool(
            offline_graph_cfg.get("user_priority_enabled", True)
        )
        self.offline_adaptive_role_cap_enabled = bool(
            offline_graph_cfg.get("adaptive_role_cap_enabled", True)
        )
        self.offline_assistant_max_ratio = float(
            offline_graph_cfg.get("assistant_max_ratio", 0.35)
        )
        self.offline_system_max_ratio = float(
            offline_graph_cfg.get("system_max_ratio", 0.15)
        )
        self.offline_assistant_min_keep = int(
            offline_graph_cfg.get("assistant_min_keep", 1)
        )
        self.offline_assistant_signal_threshold = float(
            offline_graph_cfg.get("assistant_signal_threshold", 0.45)
        )
        self.offline_assistant_hard_filter_enabled = bool(
            offline_graph_cfg.get("assistant_hard_filter_enabled", False)
        )
        self.offline_assistant_hard_filter_when_user_exists = bool(
            offline_graph_cfg.get("assistant_hard_filter_when_user_exists", True)
        )
        self.offline_assistant_answer_cue_keywords = {
            str(x).strip().lower()
            for x in list(offline_graph_cfg.get("assistant_answer_cue_keywords", []))
            if str(x).strip()
        }
        self.offline_adaptive_top_chunks_enabled = bool(
            offline_graph_cfg.get("adaptive_top_chunks_enabled", True)
        )
        self.offline_adaptive_top_chunks_min = int(
            offline_graph_cfg.get("adaptive_top_chunks_min", 8)
        )
        self.offline_adaptive_top_chunks_max = int(
            offline_graph_cfg.get("adaptive_top_chunks_max", 18)
        )
        self.offline_adaptive_score_gap_threshold = float(
            offline_graph_cfg.get("adaptive_score_gap_threshold", 0.08)
        )
        self.offline_min_pieces_per_query = int(
            offline_graph_cfg.get("min_pieces_per_query", 8)
        )
        self.supersede_min_evidence_overlap = float(
            self.cfg.get("supersede_min_evidence_overlap", 0.25)
        )
        self._stopwords = {
            str(x).strip().lower()
            for x in list(self.config.get("memory", {}).get("mid_memory", {}).get("stopwords", []))
            if str(x).strip()
        }

        self.extractor_enabled = bool(extractor_cfg.get("enabled", False))
        self.extractor_mode = str(extractor_cfg.get("mode", "sentence_fact")).strip().lower()
        self.extractor_model = str(extractor_cfg.get("model", llm_cfg["default_model"]))
        self.extractor_temperature = float(extractor_cfg.get("temperature", 0.1))
        self.extractor_timeout_sec = int(extractor_cfg.get("timeout_sec", 60))
        self.extractor_retry_max_attempts = int(extractor_cfg.get("retry_max_attempts", 1))
        self.extractor_retry_backoff_sec = float(extractor_cfg.get("retry_backoff_sec", 0.0))
        self.extractor_retry_on_timeout = bool(extractor_cfg.get("retry_on_timeout", True))
        self.extractor_retry_on_http_502 = bool(extractor_cfg.get("retry_on_http_502", True))
        self.extractor_retry_on_url_error = bool(extractor_cfg.get("retry_on_url_error", False))
        self.extractor_think = bool(extractor_cfg.get("think", False))
        self.extractor_min_confidence = float(extractor_cfg.get("min_confidence", 0.0))
        self.extractor_max_events_per_message = int(extractor_cfg.get("max_events_per_message", 4))
        self.extractor_process_roles = {
            str(x).strip().lower()
            for x in list(extractor_cfg.get("process_roles", ["user"]))
            if str(x).strip()
        }
        self.extractor_min_chars = int(extractor_cfg.get("min_chars", 1))
        self.extractor_max_chars = int(extractor_cfg.get("max_chars", 100000))
        self.extractor_input_max_chars = int(
            extractor_cfg.get("input_max_chars", self.extractor_max_chars)
        )
        self.offline_ingest_input_max_chars = int(
            offline_graph_cfg.get("ingest_input_max_chars", self.extractor_max_chars)
        )
        self.extractor_max_output_tokens = int(extractor_cfg.get("max_output_tokens", 160))
        self.extractor_force_json_output = bool(extractor_cfg.get("force_json_output", True))
        self.extractor_every_n_messages = max(
            1, int(extractor_cfg.get("extract_every_n_messages", 1))
        )
        self.extractor_warmup_enabled = bool(extractor_cfg.get("warmup_enabled", False))
        self.extractor_warmup_timeout_sec = int(
            extractor_cfg.get("warmup_timeout_sec", self.extractor_timeout_sec)
        )
        self.extractor_warmup_prompt = str(extractor_cfg.get("warmup_prompt", "Return OK."))
        self.extractor_require_action = bool(extractor_cfg.get("require_action", True))
        self.extractor_min_filled_fields = int(extractor_cfg.get("min_filled_fields", 2))
        self.extractor_require_evidence_span = bool(extractor_cfg.get("require_evidence_span", True))
        self.extractor_reject_pronoun_subject = bool(
            extractor_cfg.get("reject_pronoun_subject", True)
        )
        self.extractor_keyword_max_count = int(extractor_cfg.get("keyword_max_count", 12))
        self.extractor_sentence_overlap = int(extractor_cfg.get("sentence_overlap", 1))
        self.extractor_sentence_min_chars = int(extractor_cfg.get("sentence_min_chars", 40))
        self.extractor_compact_retry_enabled = bool(
            extractor_cfg.get("compact_retry_enabled", True)
        )
        self.extractor_compact_window_chars = int(
            extractor_cfg.get("compact_window_chars", 260)
        )
        self.extractor_span_grounding_min_overlap = float(
            extractor_cfg.get("span_grounding_min_overlap", 0.35)
        )
        self.extractor_no_source_keyword_fallback = bool(
            extractor_cfg.get("no_source_keyword_fallback", True)
        )
        self.extractor_keyword_blacklist = {
            str(x).strip().lower()
            for x in list(extractor_cfg.get("keyword_blacklist", []))
            if str(x).strip()
        }
        self.fact_filter_enabled = bool(fact_filter_cfg.get("enabled", True))
        self.fact_filter_hint_keywords = {
            str(x).strip().lower()
            for x in list(fact_filter_cfg.get("hint_keywords", []))
            if str(x).strip()
        }
        self.fact_filter_fact_keywords = {
            str(x).strip().lower()
            for x in list(fact_filter_cfg.get("fact_keywords", []))
            if str(x).strip()
        }
        # Soft-scoring gate (config-driven) to avoid over-rigid hard rules.
        self.gating_enabled = bool(gating_cfg.get("enabled", True))
        self.gating_hard_require_action = bool(
            gating_cfg.get(
                "hard_require_action",
                bool(extractor_cfg.get("require_action", True)),
            )
        )
        self.gating_hard_require_subject_or_object = bool(
            gating_cfg.get("hard_require_subject_or_object", True)
        )
        self.gating_hard_min_confidence = float(
            gating_cfg.get(
                "hard_min_confidence",
                float(extractor_cfg.get("min_confidence", 0.0)),
            )
        )
        self.gating_quality_threshold = float(gating_cfg.get("quality_threshold", 0.55))
        self.gating_weight_completeness = float(gating_cfg.get("weight_completeness", 0.45))
        self.gating_weight_grounding = float(gating_cfg.get("weight_grounding", 0.35))
        self.gating_weight_keyword = float(gating_cfg.get("weight_keyword", 0.15))
        self.gating_weight_structure = float(gating_cfg.get("weight_structure", 0.05))
        self.gating_pronoun_penalty = float(gating_cfg.get("pronoun_penalty", 0.10))
        self.gating_keyword_empty_score = float(gating_cfg.get("keyword_empty_score", 0.40))
        self.gating_grounding_missing_score = float(
            gating_cfg.get("grounding_missing_score", 0.50)
        )
        self.gating_log_rejections = bool(gating_cfg.get("log_rejections", True))
        self._subject_pronouns = {
            str(x).strip().lower()
            for x in list(
                extractor_cfg.get(
                    "subject_pronouns",
                    ["i", "you", "we", "he", "she", "they", "it", "me", "us", "him", "her", "them"],
                )
            )
            if str(x).strip()
        }
        self._entity_norm = LongMemoryEntityNormalizer(
            enabled=bool(norm_cfg.get("enabled", True)),
            pronoun_map=dict(
                norm_cfg.get(
                    "pronoun_map",
                    {
                        "i": "user",
                        "me": "user",
                        "my": "user",
                        "mine": "user",
                        "myself": "user",
                        "we": "user",
                        "us": "user",
                        "our": "user",
                        "ours": "user",
                        "you": "user",
                        "your": "user",
                        "yours": "user",
                    },
                )
            ),
            possessive_prefixes=list(
                norm_cfg.get("possessive_prefixes", ["my", "your", "our", "their", "his", "her"])
            ),
        )
        self._text = LongMemoryTextUtils(
            stopwords=self._stopwords,
            keyword_max_count=self.extractor_keyword_max_count,
            no_source_keyword_fallback=self.extractor_no_source_keyword_fallback,
            sentence_overlap=self.extractor_sentence_overlap,
            sentence_min_chars=self.extractor_sentence_min_chars,
        )
        self._gate = LongMemoryGate(
            config=LongMemoryGateConfig(
                enabled=self.gating_enabled,
                hard_only=bool(gating_cfg.get("hard_only", False)),
                hard_require_action=self.gating_hard_require_action,
                hard_require_subject_or_object=self.gating_hard_require_subject_or_object,
                hard_min_confidence=self.gating_hard_min_confidence,
                quality_threshold=self.gating_quality_threshold,
                weight_completeness=self.gating_weight_completeness,
                weight_grounding=self.gating_weight_grounding,
                weight_keyword=self.gating_weight_keyword,
                weight_structure=self.gating_weight_structure,
                pronoun_penalty=self.gating_pronoun_penalty,
                keyword_empty_score=self.gating_keyword_empty_score,
                grounding_missing_score=self.gating_grounding_missing_score,
            ),
            text_utils=self._text,
            subject_pronouns=self._subject_pronouns,
        )
        self.query_struct_enabled = bool(query_struct_cfg.get("enabled", True))
        self.query_struct_model = str(query_struct_cfg.get("model", self.extractor_model))
        self.query_struct_temperature = float(query_struct_cfg.get("temperature", 0.0))
        self.query_struct_timeout_sec = int(query_struct_cfg.get("timeout_sec", 30))
        self.query_struct_max_keywords = int(query_struct_cfg.get("max_keywords", 8))
        self.query_struct_max_output_tokens = int(query_struct_cfg.get("max_output_tokens", 96))
        self.query_struct_think = bool(query_struct_cfg.get("think", False))
        self.query_struct_force_json_output = bool(
            query_struct_cfg.get("force_json_output", True)
        )

        self.ollama_host = str(llm_cfg["host"]).rstrip("/")
        self._extract_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self._extractor_warmed_up = False
        self._extractor_seen_messages = 0
        self._extractor_calls = 0
        self._extractor_success = 0
        self._extractor_failures = 0
        self._extractor_json_success = 0
        self._extractor_schema_pass = 0
        self._extractor_empty_payload = 0
        self._extractor_retry_compact = 0
        self.staging_enabled = bool(self.cfg.get("staging_enabled", True))

        self.query_rewrite_enabled = bool(self.cfg.get("query_rewrite_enabled", False))
        self.query_rewrite_max = max(0, int(self.cfg.get("query_rewrite_max", 2)))

        self.store = LongMemoryStore(
            database_file=str(self.cfg["database_file"]),
            sqlite_busy_timeout_ms=int(self.cfg.get("sqlite_busy_timeout_ms", 5000)),
            sqlite_journal_mode=str(self.cfg.get("sqlite_journal_mode", "WAL")),
            sqlite_synchronous=str(self.cfg.get("sqlite_synchronous", "NORMAL")),
            embedding_dim=self.embedding_dim,
        )
        self.conn = self.store.conn
        self.db_path = self.store.db_path
        self.current_step = self.store.load_current_step()

        self._ingest_event_total = 0
        self._ingest_event_accepted = 0
        self._ingest_event_rejected = 0
        self._ingest_reject_reasons: Dict[str, int] = {}
        self._extractor_engine = LongMemoryExtractor(self)
        self._persist_engine = LongMemoryPersistEngine(self)
        logger.info(
            "LongMemory initialized "
            f"(enabled={self.enabled}, database={self.db_path})."
        )

    @staticmethod
    def _sentence_feature_score(text: str) -> float:
        return sentence_feature_score(text)

    @staticmethod
    def _dedup_chunks_keep_best(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return dedup_chunks_keep_best(chunks)

    def _build_evidence_packs(self, texts: List[str], max_chars: int) -> List[str]:
        return build_evidence_packs(
            texts=texts,
            split_sentences_fn=self._split_sentences,
            normalize_space_fn=self._normalize_space,
            sentence_score_fn=self._sentence_feature_score,
            min_sentence_chars=self.offline_pack_min_sentence_chars,
            top_sentences_per_chunk=self.offline_pack_top_sentences_per_chunk,
            max_packs=self.offline_pack_max_packs,
            max_chars=max_chars,
        )

    def _build_sentence_window_pieces(self, text: str, max_chars: int) -> List[str]:
        sentences = [
            self._normalize_space(sent)
            for sent in self._split_sentences(text)
            if len(self._normalize_space(sent)) >= int(self.offline_pack_min_sentence_chars)
        ]
        if not sentences:
            fallback = self._normalize_space(str(text))
            return [fallback[: max(64, int(max_chars))]] if fallback else []

        candidates: List[Dict[str, Any]] = []
        for idx, sent in enumerate(sentences):
            score = float(self._sentence_feature_score(sent))
            candidates.append({"idx": idx, "text": sent, "score": score})
        candidates.sort(key=lambda item: float(item["score"]), reverse=True)

        top_k = max(1, int(self.offline_pack_top_sentences_per_chunk))
        radius = max(0, int(self.extractor_sentence_overlap))
        selected = sorted(candidates[:top_k], key=lambda item: int(item["idx"]))
        out: List[str] = []
        seen = set()
        max_len = max(64, int(max_chars))

        for item in selected:
            idx = int(item["idx"])
            start = max(0, idx - radius)
            end = min(len(sentences), idx + radius + 1)
            window = self._normalize_space(" ".join(sentences[start:end])).strip()
            if not window:
                continue
            if len(window) > max_len:
                window = window[:max_len].rsplit(" ", 1)[0].strip() or window[:max_len].strip()
            key = window.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(window)
        return out

    def _build_sentence_window_source_items(
        self,
        source_items: List[Dict[str, str]],
        max_chars: int,
    ) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        for item in source_items:
            role = str(item.get("role", "user")).strip().lower() or "user"
            text = str(item.get("text", "")).strip()
            session_date = str(item.get("session_date", "")).strip()
            if not text:
                continue
            pieces = self._build_sentence_window_pieces(text, max_chars)
            if not pieces:
                continue
            for piece in pieces:
                out.append({"role": role, "text": piece, "session_date": session_date})
        return out

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        # Compatibility wrapper for internal callers.
        return LongMemoryTextUtils(
            stopwords=[],
            keyword_max_count=16,
            no_source_keyword_fallback=True,
            sentence_overlap=1,
            sentence_min_chars=1,
        ).tokenize(text)

    def _keyword_candidates_from_text(self, text: str) -> List[str]:
        out = self._text.keyword_candidates(text)
        if not self.extractor_keyword_blacklist:
            return out
        return [x for x in out if str(x).strip().lower() not in self.extractor_keyword_blacklist]

    def _build_keywords(
        self,
        *,
        model_keywords: List[str],
        subject: str,
        action: str,
        obj: str,
        event_text: str,
        raw_span: str,
        source_content: str,
        time_text: str,
        location_text: str,
    ) -> List[str]:
        out = self._text.build_keywords(
            model_keywords=model_keywords,
            subject=subject,
            action=action,
            obj=obj,
            event_text=event_text,
            raw_span=raw_span,
            source_content=source_content,
            time_text=time_text,
            location_text=location_text,
        )
        if not self.extractor_keyword_blacklist:
            return out
        return [k for k in out if str(k).strip().lower() not in self.extractor_keyword_blacklist]

    @staticmethod
    def _normalize_space(text: str) -> str:
        return LongMemoryTextUtils.normalize_space(text)

    def _span_grounded(self, span: str, content: str) -> bool:
        return self._gate.span_overlap_ratio(span, content) >= 0.8

    def _span_overlap_ratio(self, span: str, content: str) -> float:
        return self._gate.span_overlap_ratio(span, content)

    def _normalize_fact_component(self, text: str) -> str:
        return self._text.normalize_fact_component(text)

    def _build_fact_key(self, subject: str, action: str, obj: str) -> str:
        return self._text.build_fact_key(subject, action, obj)

    def _is_pronoun_subject(self, subject: str) -> bool:
        return self._gate.is_pronoun_subject(subject)

    def _event_is_valid(
        self,
        *,
        subject: str,
        action: str,
        obj: str,
        confidence: float,
        evidence_span: str,
        source_content: str,
    ) -> bool:
        if confidence < self.extractor_min_confidence:
            return False
        filled = int(bool(subject)) + int(bool(action)) + int(bool(obj))
        if filled < self.extractor_min_filled_fields:
            return False
        if self.extractor_require_action and not action:
            return False
        if self.extractor_reject_pronoun_subject and self._is_pronoun_subject(subject):
            return False
        if self.extractor_require_evidence_span:
            if not evidence_span:
                return False
            if not self._span_grounded(evidence_span, source_content):
                return False
        return True

    def _record_reject(self, reason: str) -> None:
        self._ingest_event_rejected += 1
        key = str(reason).strip() or "unknown"
        self._ingest_reject_reasons[key] = int(self._ingest_reject_reasons.get(key, 0)) + 1
        if self.gating_log_rejections:
            logger.info(f"LongMemory reject: reason={key}")

    def _stage_rejected_event(
        self,
        *,
        reason: str,
        subject: str,
        action: str,
        obj: str,
        event_text: str,
        keywords: List[str],
        role: str,
        confidence: float,
        source_model: str,
        raw_span: str,
        source_content: str,
    ) -> None:
        if not self.staging_enabled:
            return
        skeleton = str(event_text).strip() or f"{subject} | {action} | {obj}".strip()
        if not skeleton:
            return
        fact_key = self._build_fact_key(subject, action, obj)
        staging_id = self._stable_id(
            "staging",
            f"{fact_key}|{skeleton}|{raw_span}|{reason}|{self.current_step + 1}",
        )
        emb = self._safe_embed(" ".join([skeleton, " ".join(keywords)]).strip())
        self.store.insert_staging_event(
            staging_id=staging_id,
            fact_key=fact_key,
            skeleton_text=skeleton,
            skeleton_embedding=emb,
            keywords=keywords,
            role=role,
            extract_confidence=confidence,
            source_model=source_model,
            raw_span=raw_span,
            source_content=source_content[: max(0, self.extractor_input_max_chars)],
            reject_reason=reason,
            current_step=self.current_step + 1,
        )

    def _upsert_event_nodes(
        self,
        *,
        event_id: str,
        subject: str,
        predicate: str,
        value: str,
        time_text: str,
        location_text: str,
        keywords: List[str],
        raw_span: str,
    ) -> None:
        if not self.node_graph_enabled:
            return
        upsert_event_nodes(
            store=self.store,
            event_id=event_id,
            subject=subject,
            predicate=predicate,
            value=value,
            time_text=time_text,
            location_text=location_text,
            keywords=keywords,
            raw_span=raw_span,
            context_max_chars_per_item=self.context_max_chars_per_item,
            node_keyword_limit=self.node_keyword_limit,
            current_step=self.current_step,
            normalize_space_fn=self._normalize_space,
            stable_id_fn=self._stable_id,
            safe_embed_fn=self._safe_embed,
        )

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        return LongMemoryTextUtils.split_sentences(text)

    def _pack_sentences(self, text: str, max_chars: int) -> List[str]:
        return self._text.pack_sentences(text, max_chars)

    @staticmethod
    def _extract_first_json_block(text: str) -> str:
        return extract_first_json_block(text)

    @staticmethod
    def _safe_json_loads_relaxed(text: str) -> Any:
        return safe_json_loads_relaxed(text)

    @staticmethod
    def _stable_id(prefix: str, text: str) -> str:
        digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]
        return f"{prefix}:{digest}"

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na <= 0.0 or nb <= 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _safe_embed(self, text: str) -> np.ndarray:
        if not self.retrieval_use_embedding:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        try:
            return embed(text, self.embedding_dim)
        except (RuntimeError, ValueError, TypeError, OSError) as exc:
            logger.warn(f"LongMemory embedding fallback to zeros: {exc}")
            return np.zeros(self.embedding_dim, dtype=np.float32)

    def _keyword_overlap_features(
        self, query_tokens: set[str], corpus_tokens: set[str]
    ) -> Tuple[int, float]:
        if (not query_tokens) or (not corpus_tokens):
            return 0, 0.0
        overlap = query_tokens.intersection(corpus_tokens)
        overlap_count = len(overlap)
        coverage = float(overlap_count) / float(max(1, len(query_tokens)))
        density = float(overlap_count) / float(max(1, len(corpus_tokens)))
        keyword_score = ((1.0 - self.keyword_density_weight) * coverage) + (
            self.keyword_density_weight * density
        )
        return overlap_count, float(keyword_score)

    def _extract_query_struct(self, query: str) -> Dict[str, Any]:
        if not self.query_struct_enabled:
            return {"keywords": self._keyword_candidates_from_text(query), "skeleton": query}
        prompt = (
            "Extract retrieval structure from one question.\n"
            'Return JSON only: {"keywords":[],"skeleton":""}\n'
            f"max keywords: {self.query_struct_max_keywords}\n"
            f"question: {query}\n"
        )
        try:
            raw = ollama_generate_with_retry(
                host=self.ollama_host,
                model=self.query_struct_model,
                prompt=prompt,
                temperature=self.query_struct_temperature,
                timeout_sec=self.query_struct_timeout_sec,
                opener=self._extract_opener,
                max_attempts=1,
                backoff_sec=0.0,
                retry_on_timeout=False,
                retry_on_http_502=False,
                retry_on_url_error=False,
                max_output_tokens=self.query_struct_max_output_tokens,
                think=self.query_struct_think,
                response_format="json" if self.query_struct_force_json_output else None,
            )
            payload = self._safe_json_loads_relaxed(self._extract_first_json_block(raw))
            raw_keywords = payload.get("keywords", [])
            if isinstance(raw_keywords, list):
                model_keywords = [str(x).strip().lower() for x in raw_keywords if str(x).strip()]
            else:
                model_keywords = self._keyword_candidates_from_text(str(raw_keywords))
            skeleton = str(payload.get("skeleton", "")).strip() or query
            keywords = self._build_keywords(
                model_keywords=model_keywords,
                subject="",
                action="",
                obj="",
                event_text=skeleton,
                raw_span=query,
                source_content=query,
                time_text="",
                location_text="",
            )
            return {"keywords": keywords[: self.query_struct_max_keywords], "skeleton": skeleton}
        except (RuntimeError, ValueError, TypeError, OSError):
            return {"keywords": self._keyword_candidates_from_text(query), "skeleton": query}

    @staticmethod
    def _strip_role_prefix(text: str) -> str:
        value = str(text or "").strip()
        if not value:
            return ""
        return re.sub(r"^\((?:user|assistant|system)\)\s*", "", value, flags=re.IGNORECASE).strip()

    def _normalize_long_memory_chunk(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize mid-memory chunk payload into long-memory ingestion shape."""
        raw_text = str(item.get("text", "")).strip()
        role = str(item.get("role", "")).strip().lower()
        if role not in {"user", "assistant", "system"}:
            match = re.match(r"^\((user|assistant|system)\)\s*", raw_text, flags=re.IGNORECASE)
            if match:
                role = str(match.group(1)).strip().lower()
        if role not in {"user", "assistant", "system"}:
            role = "user"

        text = self._strip_role_prefix(raw_text)

        session_date = str(
            item.get("session_date")
            or item.get("date")
            or item.get("timestamp")
            or ""
        ).strip()
        if not session_date:
            raw_session_dates = item.get("session_dates", [])
            if isinstance(raw_session_dates, list):
                for value in raw_session_dates:
                    candidate = str(value or "").strip()
                    if candidate:
                        session_date = candidate
                        break
        if not session_date:
            raw_time_sources = item.get("time_sources", [])
            if isinstance(raw_time_sources, list):
                for ts in raw_time_sources:
                    if not isinstance(ts, dict):
                        continue
                    candidate = str(ts.get("time", "")).strip()
                    if candidate:
                        session_date = candidate
                        break

        return {
            "role": role,
            "text": text,
            "session_date": session_date,
            "score": float(item.get("score", 0.0) or 0.0),
            "answer_density": float(item.get("answer_density", 0.0) or 0.0),
            "has_answer_count": int(item.get("has_answer_count", 0) or 0),
        }

    def _warmup_extractor(self) -> None:
        if (not self.extractor_warmup_enabled) or self._extractor_warmed_up:
            return
        try:
            _ = ollama_generate_with_retry(
                host=self.ollama_host,
                model=self.extractor_model,
                prompt=self.extractor_warmup_prompt,
                temperature=self.extractor_temperature,
                timeout_sec=self.extractor_warmup_timeout_sec,
                opener=self._extract_opener,
                max_attempts=1,
                backoff_sec=0.0,
                retry_on_timeout=False,
                retry_on_http_502=False,
                retry_on_url_error=False,
                max_output_tokens=8,
                think=False,
            )
            self._extractor_warmed_up = True
            logger.info("LongMemory extractor warmup done.")
        except (RuntimeError, ValueError, TypeError, OSError) as exc:
            logger.warn(f"LongMemory extractor warmup skipped: {exc}")

    def extract_events_structured(self, message: Message, force: bool = False) -> List[Dict[str, Any]]:
        return self._extractor_engine.extract_events_structured(message, force=force)

    def ingest_from_chunks(
        self,
        *,
        chunks: List[Dict[str, Any]],
        top_chunks: int,
        max_chars_per_chunk: int,
    ) -> int:
        """Offline graph build: extract structured events from retrieved chunks and persist."""
        if not self.enabled:
            return 0
        if not self.extractor_enabled:
            return 0
        if not chunks:
            return 0

        selected = self._select_offline_chunks(chunks=chunks, top_chunks=top_chunks)
        source_items = [
            {
                "role": str(item.get("role", "user")).strip().lower() or "user",
                "text": str(item.get("text", "")).strip(),
                "session_date": str(item.get("session_date", "")).strip(),
            }
            for item in selected
            if str(item.get("text", "")).strip()
        ]
        source_items = self._prioritize_source_items(source_items)
        if not source_items:
            return 0

        pieces = self._build_sentence_window_source_items(
            source_items,
            int(max_chars_per_chunk),
        )
        if len(pieces) < max(0, self.offline_min_pieces_per_query):
            supplement: List[Dict[str, str]] = []
            for item in source_items:
                role = str(item.get("role", "user")).strip().lower() or "user"
                text = str(item.get("text", "")).strip()
                session_date = str(item.get("session_date", "")).strip()
                if not text:
                    continue
                if self.offline_pack_enabled:
                    packs = self._build_evidence_packs([text], int(max_chars_per_chunk))
                else:
                    packs = self._pack_sentences(text, int(max_chars_per_chunk))
                for packed in packs:
                    supplement.append({"role": role, "text": packed, "session_date": session_date})
            merged: List[Dict[str, str]] = []
            seen_piece_keys = set()
            for item in pieces + supplement:
                key = self._normalize_space(str(item.get("text", ""))).lower()
                if not key or key in seen_piece_keys:
                    continue
                seen_piece_keys.add(key)
                merged.append(item)
            pieces = merged
        if not pieces:
            return 0
        accepted = 0
        seen_event_keys: set[str] = set()
        for item in pieces:
            piece = str(item.get("text", "")).strip()
            role = str(item.get("role", "user")).strip().lower() or "user"
            session_date = str(item.get("session_date", "")).strip()
            if not piece:
                continue
            msg = {
                "role": role,
                "content": piece[: max(32, int(self.offline_ingest_input_max_chars))],
            }
            events = self.extract_events_structured(msg, force=True)
            for event in events:
                if session_date:
                    event["source_date"] = session_date
                if (not str(event.get("time", "")).strip()) and session_date:
                    event["time"] = session_date
                dedup_key = self._normalize_space(
                    "|".join(
                        [
                            str(event.get("subject", "")),
                            str(event.get("action", "")),
                            str(event.get("object", "")),
                            str(event.get("raw_span", "")),
                        ]
                    )
                ).lower()
                if dedup_key in seen_event_keys:
                    continue
                seen_event_keys.add(dedup_key)
                self._persist_event(event)
                accepted += 1
        if accepted > 0:
            self.store.commit()
        return accepted

    def _source_item_signal_score(self, text: str) -> float:
        value = str(text).strip()
        if not value:
            return 0.0
        score = 0.0
        lowered = value.lower()
        if re.search(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", lowered):
            score += 0.35
        if re.search(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b", lowered):
            score += 0.20
        if re.search(r"\b\d+\b", lowered):
            score += 0.15
        if re.search(r"\b(?:in|at|from|to)\s+[A-Z][a-zA-Z]+\b", value):
            score += 0.15
        if len(self._tokenize(value)) >= 10:
            score += 0.10
        if any(tok in lowered for tok in ("bought", "moved", "met", "booked", "scheduled", "diagnosed")):
            score += 0.15
        return min(1.0, score)

    def _source_item_has_answer_cue(self, text: str) -> bool:
        lowered = str(text).strip().lower()
        if not lowered:
            return False
        if re.search(r"\b\d+\b", lowered):
            return True
        return any(tok in lowered for tok in self.offline_assistant_answer_cue_keywords)

    def _prioritize_source_items(self, source_items: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Prefer user evidence and cap assistant/system noise in a configurable, generic way."""
        if (not self.offline_user_priority_enabled) or (not source_items):
            return source_items
        users = [x for x in source_items if str(x.get("role", "")).lower() == "user"]
        assistants = [x for x in source_items if str(x.get("role", "")).lower() == "assistant"]
        systems = [x for x in source_items if str(x.get("role", "")).lower() == "system"]
        others = [
            x
            for x in source_items
            if str(x.get("role", "")).lower() not in {"user", "assistant", "system"}
        ]

        user_count = len(users)
        if user_count <= 0:
            user_count = len(source_items)
        max_assistant = max(
            int(self.offline_assistant_min_keep),
            int(round(user_count * max(0.0, self.offline_assistant_max_ratio))),
        )
        max_system = max(0, int(round(user_count * max(0.0, self.offline_system_max_ratio))))

        if self.offline_adaptive_role_cap_enabled and assistants:
            scored_assistants = [
                (self._source_item_signal_score(str(item.get("text", ""))), item)
                for item in assistants
            ]
            scored_assistants.sort(key=lambda x: float(x[0]), reverse=True)
            high_signal_count = sum(
                1
                for score, _item in scored_assistants
                if float(score) >= float(self.offline_assistant_signal_threshold)
            )
            max_assistant = max(max_assistant, high_signal_count)
            assistants = [item for _score, item in scored_assistants]
            if self.offline_assistant_hard_filter_enabled and (
                (not self.offline_assistant_hard_filter_when_user_exists) or len(users) > 0
            ):
                filtered_assistants: List[Dict[str, str]] = []
                for item in assistants:
                    text = str(item.get("text", ""))
                    signal = self._source_item_signal_score(text)
                    if signal >= float(self.offline_assistant_signal_threshold):
                        filtered_assistants.append(item)
                        continue
                    if self._source_item_has_answer_cue(text):
                        filtered_assistants.append(item)
                assistants = filtered_assistants

        prioritized: List[Dict[str, str]] = []
        prioritized.extend(users)
        prioritized.extend(assistants[:max_assistant])
        prioritized.extend(systems[:max_system])
        prioritized.extend(others)
        return prioritized

    def _select_offline_chunks(
        self, *, chunks: List[Dict[str, Any]], top_chunks: int
    ) -> List[Dict[str, Any]]:
        """Adaptive chunk selection to improve extraction coverage without dataset-specific rules."""
        if not chunks:
            return []
        base = max(1, int(top_chunks))
        ranked = [
            self._normalize_long_memory_chunk(dict(x))
            for x in chunks
        ]
        ranked.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        if not self.offline_adaptive_top_chunks_enabled:
            return ranked[:base]

        min_k = max(1, int(self.offline_adaptive_top_chunks_min))
        max_k = max(min_k, int(self.offline_adaptive_top_chunks_max))
        keep = max(min_k, min(base, max_k))

        while keep < min(len(ranked), max_k):
            prev_score = float(ranked[keep - 1].get("score", 0.0))
            next_score = float(ranked[keep].get("score", 0.0))
            if (prev_score - next_score) > self.offline_adaptive_score_gap_threshold:
                break
            keep += 1

        chosen = ranked[:keep]
        tail_budget = min(2, max(1, keep // 4))
        if tail_budget > 0 and keep < len(ranked):
            tail_candidates: List[tuple[float, float, Dict[str, Any]]] = []
            for item in ranked[keep:]:
                text = str(item.get("text", "")).strip()
                if not text:
                    continue
                signal = self._source_item_signal_score(text)
                # Pull in only a tiny tail of high-signal chunks so that
                # answer-bearing lower-ranked chunks can still enter extraction.
                if signal < 0.30:
                    continue
                tail_candidates.append((signal, float(item.get("score", 0.0)), item))
            tail_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
            for _signal, _score, item in tail_candidates[:tail_budget]:
                chosen.append(dict(item))

        return self._dedup_chunks_keep_best(chosen)

    def _persist_event(self, event: Dict[str, Any]) -> None:
        self._persist_engine.persist_event(event)

    def query(self, query_text: str) -> List[Dict[str, Any]]:
        return LongMemoryQueryEngine(self).query(query_text)

    def query_multi(self, query_texts: List[str]) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for q in query_texts:
            for item in self.query(q):
                eid = str(item.get("event_id", ""))
                if not eid:
                    continue
                prev = merged.get(eid)
                if prev is None or float(item["score"]) > float(prev["score"]):
                    merged[eid] = dict(item)
        out = sorted(merged.values(), key=lambda x: float(x["score"]), reverse=True)
        return out[: self.retrieval_top_k]

    def build_context_snippets(self, query_text: str) -> List[str]:
        items = self.query(query_text)
        out: List[str] = []
        start_idx = 0
        if self.context_chain_enabled and items:
            chain_size = max(1, int(self.context_chain_size))
            chain_items = items[:chain_size]
            chain_lines: List[str] = []
            for idx, item in enumerate(chain_items, start=1):
                text = str(item.get("text", "")).strip()
                if not text:
                    continue
                chain_lines.append(f"[{idx}] {text[: self.context_max_chars_per_item]}")
            if chain_lines:
                out.append(" || ".join(chain_lines)[: self.context_chain_max_chars])
                start_idx = len(chain_items)

        for item in items[start_idx : start_idx + max(0, self.context_max_items)]:
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            out.append(text[: self.context_max_chars_per_item])
        return out

    def build_context_snippets_multi(self, query_texts: List[str]) -> List[str]:
        items = self.query_multi(query_texts)
        out: List[str] = []
        start_idx = 0
        if self.context_chain_enabled and items:
            chain_size = max(1, int(self.context_chain_size))
            chain_items = items[:chain_size]
            chain_lines: List[str] = []
            for idx, item in enumerate(chain_items, start=1):
                text = str(item.get("text", "")).strip()
                if not text:
                    continue
                chain_lines.append(f"[{idx}] {text[: self.context_max_chars_per_item]}")
            if chain_lines:
                out.append(" || ".join(chain_lines)[: self.context_chain_max_chars])
                start_idx = len(chain_items)

        for item in items[start_idx : start_idx + max(0, self.context_max_items)]:
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            out.append(text[: self.context_max_chars_per_item])
        return out

    def debug_stats(self) -> Dict[str, int]:
        counts = self.store.debug_counts()
        return {
            "nodes": int(counts["events"]),
            "edges": int(counts.get("relations", 0)),
            "events": int(counts["events"]),
            "staging_events": int(counts.get("staging_events", 0)),
            "event_nodes": int(counts.get("event_nodes", 0)),
            "event_node_edges": int(counts.get("event_node_edges", 0)),
            "details": int(counts["details"]),
            "active_events": int(counts["active_events"]),
            "superseded_events": int(counts["superseded_events"]),
            "queued_updates": 0,
            "applied_updates": int(self.current_step),
            "ingest_event_total": int(self._ingest_event_total),
            "ingest_event_accepted": int(self._ingest_event_accepted),
            "ingest_event_rejected": int(self._ingest_event_rejected),
            "extractor_calls": int(self._extractor_calls),
            "extractor_success": int(self._extractor_success),
            "extractor_failures": int(self._extractor_failures),
            "extractor_json_success": int(self._extractor_json_success),
            "extractor_schema_pass": int(self._extractor_schema_pass),
            "extractor_empty_payload": int(self._extractor_empty_payload),
            "extractor_retry_compact": int(self._extractor_retry_compact),
            "extractor_seen_messages": int(self._extractor_seen_messages),
            "candidate_events": 0,
            "reject_reason_low_confidence": 0,
            "reject_reason_few_keywords": 0,
            "reject_reason_short_object": 0,
            "reject_reason_few_entities": 0,
            "reject_reason_short_sentence": 0,
            "reject_reason_long_sentence": 0,
            "reject_reason_missing_time_or_location": 0,
            "reject_reason_rejected_phrase": 0,
            "reject_reason_generic_subject_action": 0,
            "reject_reason_generic_action_disabled": 0,
            "reject_reason_empty_key_component": 0,
            **self._gate.reject_stats(self._ingest_reject_reasons),
            "reject_reason_legacy_gate_reject": int(
                self._ingest_reject_reasons.get("legacy_gate_reject", 0)
            ),
            "reject_reason_empty_event_text": int(
                self._ingest_reject_reasons.get("empty_event_text", 0)
            ),
        }

    def export_snapshot_to_store(self, target_store: LongMemoryStore) -> None:
        """Copy the current long-memory snapshot into another SQLite store."""
        if target_store is None:
            return
        target_store.conn.executemany(
            """
            INSERT OR REPLACE INTO events(
              event_id, fact_key, subject_action_key, fact_type, skeleton_text, skeleton_embedding,
              keywords, role, boundary_flag, extract_confidence, source_model, raw_span,
              status, is_latest, salience, first_seen_step, last_seen_step
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    str(row["event_id"] or ""),
                    str(row["fact_key"] or ""),
                    str(row["subject_action_key"] or ""),
                    str(row["fact_type"] or ""),
                    str(row["skeleton_text"] or ""),
                    row["skeleton_embedding"],
                    str(row["keywords"] or "[]"),
                    str(row["role"] or ""),
                    int(row["boundary_flag"] or 0),
                    float(row["extract_confidence"] or 0.0),
                    str(row["source_model"] or ""),
                    str(row["raw_span"] or ""),
                    str(row["status"] or ""),
                    int(row["is_latest"] or 0),
                    float(row["salience"] or 0.0),
                    int(row["first_seen_step"] or 0),
                    int(row["last_seen_step"] or 0),
                )
                for row in self.store.fetch_all_events()
            ],
        )
        target_store.conn.executemany(
            """
            INSERT OR REPLACE INTO details(detail_id, event_id, kind, text, created_step)
            VALUES(?, ?, ?, ?, ?)
            """,
            [
                (
                    str(row["detail_id"] or ""),
                    str(row["event_id"] or ""),
                    str(row["kind"] or ""),
                    str(row["text"] or ""),
                    int(row["created_step"] or 0),
                )
                for row in self.store.fetch_all_details()
            ],
        )
        target_store.conn.executemany(
            """
            INSERT OR REPLACE INTO event_nodes(
              node_id, event_id, node_kind, node_text, is_core, node_embedding, created_step
            ) VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    str(row["node_id"] or ""),
                    str(row["event_id"] or ""),
                    str(row["node_kind"] or ""),
                    str(row["node_text"] or ""),
                    int(row["is_core"] or 0),
                    row["node_embedding"],
                    int(row["created_step"] or 0),
                )
                for row in self.store.fetch_all_event_nodes()
            ],
        )
        target_store.conn.executemany(
            """
            INSERT OR REPLACE INTO event_node_edges(
              node_edge_id, event_id, from_node_id, to_node_id, relation, weight, created_step
            ) VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    str(row["node_edge_id"] or ""),
                    str(row["event_id"] or ""),
                    str(row["from_node_id"] or ""),
                    str(row["to_node_id"] or ""),
                    str(row["relation"] or ""),
                    float(row["weight"] or 0.0),
                    int(row["created_step"] or 0),
                )
                for row in self.store.fetch_all_event_node_edges()
            ],
        )
        target_store.conn.executemany(
            """
            INSERT OR REPLACE INTO edges(edge_id, from_event_id, to_event_id, relation, weight, created_step)
            VALUES(?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    str(row["edge_id"] or ""),
                    str(row["from_event_id"] or ""),
                    str(row["to_event_id"] or ""),
                    str(row["relation"] or ""),
                    float(row["weight"] or 0.0),
                    int(row["created_step"] or 0),
                )
                for row in self.store.fetch_all_edges()
            ],
        )
        target_store.commit()

    def clear_all(self) -> None:
        if not self.enabled:
            logger.info("LongMemory.clear_all: disabled, no-op.")
            return
        self.store.clear_all()
        self.current_step = 0
        self.store.save_current_step(self.current_step)
        self.store.commit()
        self._ingest_event_total = 0
        self._ingest_event_accepted = 0
        self._ingest_event_rejected = 0
        self._extractor_calls = 0
        self._extractor_success = 0
        self._extractor_failures = 0
        self._extractor_json_success = 0
        self._extractor_schema_pass = 0
        self._extractor_empty_payload = 0
        self._extractor_retry_compact = 0
        self._extractor_seen_messages = 0
        self._ingest_reject_reasons.clear()
        logger.info("LongMemory.clear_all: cleared SQLite event/detail tables and queue.")

    def close(self) -> None:
        self.store.commit()
        self.store.close()
