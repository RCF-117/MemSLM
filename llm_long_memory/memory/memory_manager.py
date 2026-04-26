"""Memory manager that orchestrates the active MemSLM retrieval pipeline."""

from __future__ import annotations

import json
import re
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from llm_long_memory.llm.ollama_client import LLM
from llm_long_memory.llm.ollama_client import ollama_generate_with_retry
from llm_long_memory.memory.answer_grounding_pipeline import AnswerGroundingPipeline
from llm_long_memory.memory.evidence_filter import EvidenceFilter
from llm_long_memory.memory.final_answer_composer import FinalAnswerComposer
from llm_long_memory.memory.final_answer_router import FinalAnswerRouter
from llm_long_memory.memory.evidence_graph_extractor import EvidenceGraphExtractor
from llm_long_memory.memory.evidence_light_graph import EvidenceLightGraph
from llm_long_memory.memory.graph_reasoning_toolkit import GraphReasoningToolkit
from llm_long_memory.memory.memory_manager_chat_runtime import MemoryManagerChatRuntime
from llm_long_memory.memory.specialist_layer import SpecialistLayer
from llm_long_memory.memory.memory_manager_utils import (
    build_gap_queries,
    build_query_plan,
    build_temporal_anchor_queries,
    detect_missing_slots,
    dedup_chunks_keep_best,
    is_temporal_query,
    merge_anchor_chunks,
    slot_coverage_score,
)
from llm_long_memory.memory.mid_memory import MidMemory
from llm_long_memory.memory.short_memory import ShortMemory
from llm_long_memory.utils.helpers import load_config
from llm_long_memory.utils.logger import logger


Message = Dict[str, Any]


class _NoOpLongMemory:
    """Removed long-memory compatibility stub.

    The active system no longer reads or writes the legacy long-memory graph.
    We keep this no-op object only so older debug and reset call sites remain
    safe while the rest of the codebase converges on the new graph pipeline.
    """

    cleared = False
    closed = False

    def query(self, query_text: str) -> List[Dict[str, Any]]:
        return []

    def build_context_snippets(self, query_text: str) -> List[str]:
        return []

    def ingest_from_chunks(
        self,
        *,
        chunks: List[Dict[str, Any]],
        top_chunks: int,
        max_chars_per_chunk: int,
    ) -> int:
        return 0

    def debug_stats(self) -> Dict[str, int]:
        return {
            "nodes": 0,
            "edges": 0,
            "events": 0,
            "details": 0,
            "active_events": 0,
            "superseded_events": 0,
            "queued_updates": 0,
            "applied_updates": 0,
            "ingest_event_total": 0,
            "ingest_event_accepted": 0,
            "ingest_event_rejected": 0,
            "extractor_calls": 0,
            "extractor_success": 0,
            "extractor_failures": 0,
            "extractor_seen_messages": 0,
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
        }

    def clear_all(self) -> None:
        self.cleared = True
        return None

    def close(self) -> None:
        self.closed = True
        return None


class MemoryManager:
    """Central controller for the active MemSLM pipeline.

    Main online path:
    RAG retrieval -> evidence filter -> fixed-schema claims -> light graph ->
    graph-only toolkit -> final answer composer -> 8B answer generation.
    """

    def __init__(self, llm: LLM, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize manager with config-driven memory sizes and modules."""
        self.config = config or load_config()
        self.llm = llm
        short_size = int(self.config["memory"]["short_memory_size"])
        self.short_memory = ShortMemory(max_turns=short_size, config=self.config)
        self.mid_memory = MidMemory(config=self.config)
        long_mem_cfg = dict(self.config["memory"]["long_memory"])
        requested_long_memory = bool(long_mem_cfg.get("enabled", False))
        self.long_memory_enabled = False
        self.long_memory = _NoOpLongMemory()
        if requested_long_memory:
            logger.warning(
                "MemoryManager: legacy long-memory is no longer part of the active "
                "runtime; config.memory.long_memory.enabled is ignored."
            )
        answer_grounding_cfg = dict(self.config["retrieval"]["answering"])
        self.answer_grounding = AnswerGroundingPipeline(answer_grounding_cfg)
        self.final_answer_composer = FinalAnswerComposer(answer_grounding_cfg)
        self.final_answer_router = FinalAnswerRouter(answer_grounding_cfg)
        self.final_answer_guard_enabled = bool(
            answer_grounding_cfg.get("final_answer_guard_enabled", False)
        )
        self.final_answer_second_pass_enabled = bool(
            answer_grounding_cfg.get("final_answer_second_pass_enabled", False)
        )
        self.toolkit_direct_answer_enabled = bool(
            answer_grounding_cfg.get("toolkit_direct_answer_enabled", True)
        )
        self.toolkit_direct_min_confidence = float(
            answer_grounding_cfg.get("toolkit_direct_min_confidence", 0.65)
        )
        self.toolkit_direct_allowed_intents = {
            str(x).strip().lower()
            for x in list(
                answer_grounding_cfg.get(
                    "toolkit_direct_allowed_intents",
                    ["count", "temporal_count", "temporal_compare", "update"],
                )
            )
            if str(x).strip()
        }
        self.specialist_layer = SpecialistLayer(
            self,
            dict(answer_grounding_cfg.get("specialist_layer", {})),
        )
        self.graph_refiner_enabled = bool(answer_grounding_cfg.get("graph_refiner_enabled", False))
        self.graph_toolkit = None
        self.retrieval_execution_mode = str(
            self.config["retrieval"].get("execution_mode", "memslm")
        ).strip().lower() or "memslm"
        self.model_only_enabled = bool(
            self.config["retrieval"].get("model_only_enabled", False)
        ) or self.retrieval_execution_mode == "model_only"
        self.classic_rag_enabled = bool(
            self.config["retrieval"].get("classic_rag_enabled", False)
        ) or self.retrieval_execution_mode == "naive_rag"
        self.filter_only_enabled = self.retrieval_execution_mode == "filter_only"
        specialist_cfg = dict(answer_grounding_cfg.get("specialist_layer", {}))
        specialist_enabled = bool(specialist_cfg.get("enabled", False))
        graph_toolkit_enabled = bool(specialist_cfg.get("graph_toolkit_enabled", True))
        if specialist_enabled and graph_toolkit_enabled:
            self.graph_toolkit = GraphReasoningToolkit(self)

        temporal_anchor_cfg = dict(self.config["retrieval"].get("temporal_anchor_retrieval", {}))
        self.temporal_anchor_enabled = bool(temporal_anchor_cfg.get("enabled", False))
        self.temporal_anchor_require_temporal_cue = bool(
            temporal_anchor_cfg.get("require_temporal_cue", True)
        )
        self.temporal_anchor_max_options = int(temporal_anchor_cfg.get("max_options", 3))
        self.temporal_anchor_extra_queries_per_option = int(
            temporal_anchor_cfg.get("extra_queries_per_option", 1)
        )
        self.temporal_anchor_top_n_per_query = int(
            temporal_anchor_cfg.get("top_n_per_query", 10)
        )
        self.temporal_anchor_additive_limit = int(
            temporal_anchor_cfg.get("additive_limit", 8)
        )
        self.temporal_anchor_cue_keywords = [
            str(x).strip().lower()
            for x in list(temporal_anchor_cfg.get("cue_keywords", []))
            if str(x).strip()
        ]
        query_focus_cfg = dict(self.config["retrieval"].get("query_focus_retrieval", {}))
        self.query_focus_enabled = bool(query_focus_cfg.get("enabled", False))
        self.query_focus_require_cue = bool(query_focus_cfg.get("require_cue", False))
        self.query_focus_cue_keywords = {
            str(x).strip().lower()
            for x in list(
                query_focus_cfg.get(
                    "cue_keywords",
                    [],
                )
            )
            if str(x).strip()
        }
        self.query_focus_model = str(
            query_focus_cfg.get("model", self.config["llm"]["default_model"])
        )
        self.query_focus_temperature = float(query_focus_cfg.get("temperature", 0.0))
        self.query_focus_timeout_sec = int(query_focus_cfg.get("timeout_sec", 60))
        self.query_focus_max_queries = int(query_focus_cfg.get("max_queries", 4))
        self.query_focus_top_n_per_query = int(query_focus_cfg.get("top_n_per_query", 12))
        self.query_focus_additive_limit = int(query_focus_cfg.get("additive_limit", 12))
        self.query_focus_max_output_tokens = int(
            query_focus_cfg.get("max_output_tokens", 120)
        )
        self.query_focus_force_json_output = bool(
            query_focus_cfg.get("force_json_output", True)
        )
        self.query_focus_think = bool(query_focus_cfg.get("think", False))
        query_plan_cfg = dict(self.config["retrieval"].get("query_plan", {}))
        self.query_plan_enabled = bool(query_plan_cfg.get("enabled", True))
        self.query_plan_max_sub_queries = int(query_plan_cfg.get("max_sub_queries", 4))
        self.query_plan_slot_weight = float(query_plan_cfg.get("slot_coverage_weight", 0.35))
        self.query_plan_llm_assist_enabled = bool(
            query_plan_cfg.get("llm_assist_enabled", True)
        )
        self.query_plan_llm_phrase_max = int(query_plan_cfg.get("llm_phrase_max", 3))
        self.query_plan_llm_phrase_min_tokens = int(
            query_plan_cfg.get("llm_phrase_min_tokens", 2)
        )
        gap_cfg = dict(self.config["retrieval"].get("gap_detector", {}))
        self.gap_detector_enabled = bool(gap_cfg.get("enabled", True))
        self.gap_detector_max_rounds = int(gap_cfg.get("max_rounds", 2))
        self.gap_detector_max_queries = int(gap_cfg.get("max_queries", 2))
        self.gap_detector_top_n_per_query = int(gap_cfg.get("top_n_per_query", 10))
        self.gap_detector_additive_limit = int(gap_cfg.get("additive_limit", 10))
        self.evidence_graph_cfg = dict(self.config["retrieval"].get("evidence_graph", {}))
        self.evidence_graph_enabled = bool(self.evidence_graph_cfg.get("enabled", False))
        self.evidence_filter_enabled = bool(
            self.evidence_graph_cfg.get("filter_enabled", self.evidence_graph_enabled)
        )
        self.evidence_graph_claims_enabled = bool(
            self.evidence_graph_cfg.get("claims_enabled", self.evidence_graph_enabled)
        )
        self.evidence_light_graph_enabled = bool(
            self.evidence_graph_cfg.get("light_graph_enabled", self.evidence_graph_enabled)
        )
        self.evidence_filter = EvidenceFilter(self.evidence_graph_cfg)
        self.evidence_graph_extractor = EvidenceGraphExtractor(self, self.evidence_graph_cfg)
        self.evidence_light_graph = EvidenceLightGraph(self.evidence_graph_cfg)

        self.chat_runtime = MemoryManagerChatRuntime(self)
        self.last_prompt_eval_chunks: List[Dict[str, str]] = []
        self.last_query_plan: Dict[str, object] = {}
        self.last_evidence_graph_bundle: Dict[str, object] = {}
        self.last_stage_latency_sec: Dict[str, float] = {}
        logger.info("MemoryManager initialized.")

    @staticmethod
    def _dedup_chunks_keep_best(chunks: List[Dict[str, object]]) -> List[Dict[str, object]]:
        return dedup_chunks_keep_best(chunks)

    def _is_temporal_query(self, query: str) -> bool:
        return is_temporal_query(query, self.temporal_anchor_cue_keywords)

    def _build_temporal_anchor_queries(self, query: str) -> List[str]:
        return build_temporal_anchor_queries(
            query=query,
            temporal_anchor_enabled=self.temporal_anchor_enabled,
            temporal_anchor_require_temporal_cue=self.temporal_anchor_require_temporal_cue,
            temporal_anchor_cue_keywords=self.temporal_anchor_cue_keywords,
            temporal_anchor_max_options=self.temporal_anchor_max_options,
            temporal_anchor_extra_queries_per_option=self.temporal_anchor_extra_queries_per_option,
        )

    @staticmethod
    def _extract_first_json_block(raw: str) -> str:
        text = str(raw or "").strip()
        if not text:
            return "{}"
        if text.startswith("{") and text.endswith("}"):
            return text
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return text[start : end + 1]
        return "{}"

    @staticmethod
    def _is_noun_like_phrase(text: str) -> bool:
        toks = re.findall(r"[a-z0-9]+", str(text or "").lower())
        if not toks or len(toks) > 8:
            return False
        banned = {
            "first",
            "last",
            "before",
            "after",
            "when",
            "did",
            "do",
            "does",
            "is",
            "are",
            "was",
            "were",
            "have",
            "has",
            "had",
            "why",
            "how",
            "what",
            "which",
            "who",
        }
        if toks[0] in banned:
            return False
        return not all(t in banned for t in toks)

    @staticmethod
    def _retrieval_noise_penalty(text: str) -> float:
        low = re.sub(r"\s+", " ", str(text or "")).strip().lower()
        if not low:
            return 0.0
        penalty = 0.0
        if (
            "as an ai" in low
            or "i'm just an ai" in low
            or "large language model" in low
            or "don't have personal experiences" in low
            or "do not have personal experiences" in low
            or "don't have access" in low
            or "do not have access" in low
        ):
            penalty += 0.35
        if re.search(r"\b(example script|tips?|how to|guide|best practices?)\b", low):
            penalty += 0.18
        if re.search(r"\b(you can|consider|try|should|recommended?)\b", low):
            has_fact = bool(
                re.search(
                    r"\b(i|my|we|our)\b.{0,30}\b(was|were|have|had|did|moved|set|completed|own|bought|met|serviced|tried|led|graduated)\b",
                    low,
                )
            )
            if not has_fact:
                penalty += 0.12
        if low.endswith("?"):
            penalty += 0.08
        if len(low) > 600 and not re.search(
            r"\b(i|my|we|our)\b.{0,40}\b(was|were|have|had|did|moved|set|completed|own|bought|met|serviced|tried|led|graduated)\b",
            low,
        ):
            penalty += 0.12
        return min(0.45, float(penalty))

    def _extract_query_focus_struct(self, query: str, force: bool = False) -> Dict[str, List[str]]:
        if (not self.query_focus_enabled) and (not force):
            return {"objects": [], "anchors": [], "keywords": []}
        # Keep tests/offline stubs safe: only invoke structured Ollama path when llm has transport attrs.
        if (not hasattr(self.llm, "host")) or (not hasattr(self.llm, "model_name")):
            return {"objects": [], "anchors": [], "keywords": []}
        try:
            host = str(getattr(self.llm, "host", self.config["llm"]["host"])).rstrip("/")
            opener = getattr(self.llm, "_opener", None)
            if opener is None:
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            prompt = (
                "Extract retrieval focus from one QA query.\n"
                'Return JSON only: {"objects":[],"anchors":[],"keywords":[]}\n'
                "objects: noun phrases being asked about (for count/update/lookup).\n"
                "anchors: compared entities or key mentions.\n"
                "keywords: short retrieval terms.\n"
                f"max each list: {max(1, self.query_focus_max_queries)}\n"
                f"query: {query}\n"
            )
            raw = ollama_generate_with_retry(
                host=host,
                model=self.query_focus_model,
                prompt=prompt,
                temperature=self.query_focus_temperature,
                timeout_sec=self.query_focus_timeout_sec,
                opener=opener,
                max_attempts=1,
                backoff_sec=0.0,
                retry_on_timeout=False,
                retry_on_http_502=False,
                retry_on_url_error=False,
                max_output_tokens=self.query_focus_max_output_tokens,
                think=self.query_focus_think,
                response_format="json" if self.query_focus_force_json_output else None,
            )
            data = json.loads(self._extract_first_json_block(raw))
            out: Dict[str, List[str]] = {}
            for key in ("objects", "anchors", "keywords"):
                raw_list = data.get(key, [])
                if isinstance(raw_list, list):
                    values = [str(x).strip() for x in raw_list if str(x).strip()]
                else:
                    values = [str(raw_list).strip()] if str(raw_list).strip() else []
                uniq: List[str] = []
                seen: set[str] = set()
                for v in values:
                    norm = re.sub(r"\s+", " ", v).strip()
                    if not norm:
                        continue
                    low = norm.lower()
                    if low in seen:
                        continue
                    seen.add(low)
                    uniq.append(norm)
                    if len(uniq) >= max(1, self.query_focus_max_queries):
                        break
                out[key] = uniq
            return {
                "objects": out.get("objects", []),
                "anchors": out.get("anchors", []),
                "keywords": out.get("keywords", []),
            }
        except (RuntimeError, ValueError, TypeError, OSError, json.JSONDecodeError):
            return {"objects": [], "anchors": [], "keywords": []}

    def _build_query_focus_queries(self, query: str) -> List[str]:
        base = re.sub(r"\s+", " ", str(query or "")).strip()
        if not base:
            return []
        if self.query_plan_enabled:
            plan = build_query_plan(base, max_sub_queries=max(1, self.query_plan_max_sub_queries))
            planned = [str(x).strip() for x in list(plan.get("sub_queries", [])) if str(x).strip()]
            if planned:
                return planned[: max(1, int(self.query_focus_max_queries))]
        if not self.query_focus_enabled:
            return [base]
        qlow = base.lower()
        if self.query_focus_require_cue and (
            (not self.query_focus_cue_keywords)
            or (not any(c in qlow for c in self.query_focus_cue_keywords))
        ):
            return [base]
        focus = self._extract_query_focus_struct(base)
        out: List[str] = [base]
        seen: set[str] = {base.lower()}
        for key in ("objects", "anchors", "keywords"):
            for phrase in list(focus.get(key, [])):
                p = re.sub(r"\s+", " ", str(phrase or "")).strip()
                if not p:
                    continue
                low = p.lower()
                if low in seen:
                    continue
                seen.add(low)
                out.append(p)
                if len(out) >= max(1, self.query_focus_max_queries):
                    return out
        return out

    def _build_query_plan(self, query: str) -> Dict[str, object]:
        if not self.query_plan_enabled:
            return {
                "intent": "lookup",
                "answer_type": "factoid",
                "focus_phrases": [],
                "sub_queries": [str(query or "").strip()],
                "entities": [],
                "time_range": "",
                "constraints": [],
                "need_latest_state": False,
                "target_object": "",
                "compare_options": [],
                "time_terms": [],
                "state_keys": [],
                "count_unit": "",
                "plan_keywords": [],
                "must_keywords": [],
                "constraint_keywords": [],
                "keyword_query": "",
            }
        plan = build_query_plan(
            str(query or ""),
            max_sub_queries=max(1, self.query_plan_max_sub_queries),
        )
        if not self.query_plan_llm_assist_enabled:
            return plan

        focus = self._extract_query_focus_struct(str(query or ""), force=True)
        llm_phrases: List[str] = []
        seen: set[str] = set()
        for key in ("objects", "anchors", "keywords"):
            values = list(focus.get(key, []))
            for raw in values:
                txt = re.sub(r"\s+", " ", str(raw or "")).strip(" ,.;:!?\"'")
                if not txt:
                    continue
                if len(re.findall(r"[a-z0-9]+", txt.lower())) < max(
                    1, int(self.query_plan_llm_phrase_min_tokens)
                ):
                    continue
                low = txt.lower()
                if low in seen:
                    continue
                seen.add(low)
                llm_phrases.append(txt)
                if len(llm_phrases) >= max(1, int(self.query_plan_llm_phrase_max)):
                    break
            if len(llm_phrases) >= max(1, int(self.query_plan_llm_phrase_max)):
                break

        if not llm_phrases:
            return plan

        # Merge LLM focus phrases with heuristic focus phrases, then rebuild compact plan queries.
        merged_focus: List[str] = []
        seen_focus: set[str] = set()
        for raw_focus in list(plan.get("focus_phrases", [])) + list(llm_phrases):
            f = re.sub(r"\s+", " ", str(raw_focus or "")).strip(" ,.;:!?\"'")
            if not f:
                continue
            low = f.lower()
            if low in seen_focus:
                continue
            seen_focus.add(low)
            merged_focus.append(f)
            if len(merged_focus) >= 8:
                break
        plan["focus_phrases"] = merged_focus

        entities = [str(x).strip() for x in list(plan.get("entities", [])) if str(x).strip()]
        for p in merged_focus:
            if p.lower() not in {e.lower() for e in entities}:
                entities.append(p)
        plan["entities"] = entities[:8]

        if (not str(plan.get("target_object", "")).strip()) and merged_focus:
            at = str(plan.get("answer_type", "factoid")).strip().lower()
            if at in {"count", "update"} and self._is_noun_like_phrase(merged_focus[0]):
                plan["target_object"] = merged_focus[0]

        base_query = re.sub(r"\s+", " ", str(query or "")).strip()
        keyword_stop = {
            "i",
            "me",
            "my",
            "we",
            "our",
            "you",
            "your",
            "they",
            "their",
            "the",
            "a",
            "an",
            "of",
            "to",
            "for",
            "with",
            "in",
            "on",
            "at",
            "do",
            "did",
            "does",
            "is",
            "are",
            "was",
            "were",
            "what",
            "which",
            "who",
            "when",
            "where",
            "how",
        }
        keyword_tokens: List[str] = []
        for phrase in merged_focus[:4]:
            toks = re.findall(r"[a-z0-9]+", phrase.lower())
            content = [t for t in toks if len(t) >= 3 and t not in keyword_stop]
            keyword_tokens.extend(content[:2] if len(content) >= 2 else content)
        keyword_tokens = list(dict.fromkeys([t for t in keyword_tokens if t]))
        keyword_query = re.sub(r"\s+", " ", " ".join(keyword_tokens[:10])).strip()
        if len(re.findall(r"[a-z0-9]+", keyword_query.lower())) >= 2:
            plan["keyword_query"] = keyword_query

        sub_queries: List[str] = [base_query]
        if keyword_query and keyword_query.lower() != base_query.lower():
            sub_queries.append(keyword_query)
        focus_pair = re.sub(r"\s+", " ", " ".join(merged_focus[:2])).strip()
        if len(re.findall(r"[a-z0-9]+", focus_pair.lower())) >= 2:
            sub_queries.append(focus_pair)
        answer_type = str(plan.get("answer_type", "factoid")).strip().lower()
        if answer_type == "temporal_comparison" and len(merged_focus) >= 2:
            sub_queries.append(f"{merged_focus[0]} {merged_focus[1]} first before after")
        elif answer_type == "update" and merged_focus:
            sub_queries.append(f"{merged_focus[0]} current latest update status")
        elif answer_type.startswith("temporal") and merged_focus:
            sub_queries.append(f"{merged_focus[0]} date time first before after")

        dedup_sub_queries: List[str] = []
        seen_sq: set[str] = set()
        max_sub_q = min(max(1, int(self.query_plan_max_sub_queries)), 4)
        for sq in sub_queries:
            normalized = re.sub(r"\s+", " ", str(sq or "")).strip()
            if not normalized:
                continue
            low = normalized.lower()
            if low in seen_sq:
                continue
            seen_sq.add(low)
            dedup_sub_queries.append(normalized)
            if len(dedup_sub_queries) >= max_sub_q:
                break
        plan["sub_queries"] = dedup_sub_queries
        return plan

    def _apply_slot_coverage_rerank(
        self,
        units: List[Dict[str, object]],
        plan: Dict[str, object],
    ) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        weight = max(0.0, float(self.query_plan_slot_weight))
        for item in units:
            row = dict(item)
            text = str(row.get("text", "")).strip()
            cov = slot_coverage_score(text, plan)
            base = float(row.get("score", 0.0))
            noise_penalty = self._retrieval_noise_penalty(text)
            row["score_base"] = base
            row["slot_coverage"] = cov
            row["retrieval_noise_penalty"] = noise_penalty
            row["score"] = base + (weight * cov) - noise_penalty
            out.append(row)
        out.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return out

    def _merge_anchor_chunks(
        self,
        *,
        base_chunks: List[Dict[str, object]],
        extra_chunks: List[Dict[str, object]],
        additive_limit: int,
    ) -> List[Dict[str, object]]:
        return merge_anchor_chunks(
            base_chunks=base_chunks,
            extra_chunks=extra_chunks,
            additive_limit=additive_limit,
        )

    def retrieve_context(
        self, query: str
    ) -> Tuple[str, List[Dict[str, object]], List[Dict[str, object]]]:
        """Retrieve reranked chunks for final prompt context."""
        if self.model_only_enabled:
            return "", [], []
        if self.classic_rag_enabled:
            top_n = max(1, int(self.config["retrieval"].get("top_k", 5)))
            reranked_chunks = []
            if hasattr(self.mid_memory, "search_chunks_global_with_limit"):
                reranked_chunks = self.mid_memory.search_chunks_global_with_limit(
                    query,
                    top_n=top_n,
                )
            elif hasattr(self.mid_memory, "search_chunks_global"):
                reranked_chunks = self.mid_memory.search_chunks_global(query)
            context_parts: List[str] = []
            for index, item in enumerate(reranked_chunks, start=1):
                text = str(item.get("text", "")).strip()
                if not text:
                    continue
                context_parts.append(f"[Chunk {index}]\n{text}")
            return "\n\n".join(context_parts), [], reranked_chunks
        query_plan = self._build_query_plan(query)
        self.last_query_plan = dict(query_plan)
        global_chunk_enabled = bool(
            getattr(self.mid_memory, "global_chunk_retrieval_enabled", False)
        )
        if global_chunk_enabled and hasattr(self.mid_memory, "search_chunks_global"):
            reranked_chunks = self.mid_memory.search_chunks_global(query)
        elif hasattr(self.mid_memory, "search_chunks_global_with_limit"):
            reranked_chunks = self.mid_memory.search_chunks_global_with_limit(
                query=query,
                top_n=int(getattr(self.mid_memory, "global_chunk_top_n", 40)),
            )
        else:
            reranked_chunks = []

        if self.temporal_anchor_enabled and hasattr(self.mid_memory, "search_chunks_global_with_limit"):
            anchor_queries = self._build_temporal_anchor_queries(query)
            if anchor_queries:
                anchor_chunks: List[Dict[str, object]] = []
                for aq in anchor_queries:
                    anchor_chunks.extend(
                        self.mid_memory.search_chunks_global_with_limit(
                            aq,
                            top_n=self.temporal_anchor_top_n_per_query,
                        )
                    )
                if anchor_chunks:
                    reranked_chunks = self._merge_anchor_chunks(
                        base_chunks=reranked_chunks,
                        extra_chunks=self._dedup_chunks_keep_best(anchor_chunks),
                        additive_limit=self.temporal_anchor_additive_limit,
                    )
                    logger.info(
                        "MemoryManager.retrieve_context: "
                        f"temporal_anchor_queries={len(anchor_queries)}, "
                        f"anchor_candidates={len(anchor_chunks)}."
                    )
        if self.query_focus_enabled and hasattr(self.mid_memory, "search_chunks_global_with_limit"):
            focus_queries = self._build_query_focus_queries(query)
            if len(focus_queries) > 1:
                focus_chunks: List[Dict[str, object]] = []
                for fq in focus_queries[1:]:
                    focus_chunks.extend(
                        self.mid_memory.search_chunks_global_with_limit(
                            fq,
                            top_n=self.query_focus_top_n_per_query,
                        )
                    )
                if focus_chunks:
                    reranked_chunks = self._merge_anchor_chunks(
                        base_chunks=reranked_chunks,
                        extra_chunks=self._dedup_chunks_keep_best(focus_chunks),
                        additive_limit=self.query_focus_additive_limit,
                    )
                    logger.info(
                        "MemoryManager.retrieve_context: "
                        f"query_focus_queries={len(focus_queries)-1}, "
                        f"query_focus_candidates={len(focus_chunks)}."
                    )
        elif hasattr(self.mid_memory, "search_chunks_global_with_limit"):
            focus_queries = self._build_query_focus_queries(query)
            if len(focus_queries) > 1:
                plan_focus_chunks: List[Dict[str, object]] = []
                for fq in focus_queries[1:]:
                    plan_focus_chunks.extend(
                        self.mid_memory.search_chunks_global_with_limit(
                            fq,
                            top_n=max(6, int(self.query_focus_top_n_per_query)),
                        )
                    )
                if plan_focus_chunks:
                    reranked_chunks = self._merge_anchor_chunks(
                        base_chunks=reranked_chunks,
                        extra_chunks=self._dedup_chunks_keep_best(plan_focus_chunks),
                        additive_limit=max(4, int(self.query_focus_additive_limit)),
                    )
                    logger.info(
                        "MemoryManager.retrieve_context: "
                        f"planned_sub_queries={len(focus_queries)-1}, "
                        f"planned_candidates={len(plan_focus_chunks)}."
                    )
        merged_units: List[Dict[str, object]] = list(reranked_chunks)
        sentence_enabled = bool(
            getattr(self.mid_memory, "global_sentence_retrieval_enabled", False)
        )
        if sentence_enabled and hasattr(self.mid_memory, "search_sentences_global"):
            sentence_results = self.mid_memory.search_sentences_global(query)
            planned_queries = self._build_query_focus_queries(query)
            if (
                len(planned_queries) > 1
                and hasattr(self.mid_memory, "search_sentences_global_with_limit")
            ):
                for sq in planned_queries[1:]:
                    sentence_results.extend(
                        self.mid_memory.search_sentences_global_with_limit(
                            sq,
                            top_n=max(6, int(self.query_focus_top_n_per_query)),
                        )
                    )
                sentence_results = self._dedup_chunks_keep_best(sentence_results)
            if sentence_results:
                seen_texts = {
                    re.sub(r"\s+", " ", str(x.get("text", "")).strip().lower())
                    for x in merged_units
                    if str(x.get("text", "")).strip()
                }
                added = 0
                max_add = max(
                    4,
                    int(getattr(self.mid_memory, "global_sentence_top_n", 48)) // 2,
                )
                for item in sentence_results:
                    text = str(item.get("text", "")).strip()
                    norm = re.sub(r"\s+", " ", text.lower())
                    if (not text) or (norm in seen_texts):
                        continue
                    seen_texts.add(norm)
                    merged_units.append(dict(item))
                    added += 1
                    if added >= max_add:
                        break
                merged_units.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
                logger.info(
                    "MemoryManager.retrieve_context: "
                    f"sentence_candidates={len(sentence_results)}, added={added}."
                )
        merged_units = self._apply_slot_coverage_rerank(merged_units, query_plan)
        if self.gap_detector_enabled and self.gap_detector_max_rounds >= 2:
            missing_slots = detect_missing_slots(query_plan, merged_units, top_n=12)
            if missing_slots:
                gap_queries = build_gap_queries(
                    query=query,
                    plan=query_plan,
                    missing_slots=missing_slots,
                    max_queries=max(1, int(self.gap_detector_max_queries)),
                )
                if gap_queries and hasattr(self.mid_memory, "search_chunks_global_with_limit"):
                    gap_chunks: List[Dict[str, object]] = []
                    for gq in gap_queries:
                        gap_chunks.extend(
                            self.mid_memory.search_chunks_global_with_limit(
                                gq,
                                top_n=max(4, int(self.gap_detector_top_n_per_query)),
                            )
                        )
                    gap_chunks = self._dedup_chunks_keep_best(gap_chunks)
                    if gap_chunks:
                        merged_units = self._merge_anchor_chunks(
                            base_chunks=merged_units,
                            extra_chunks=gap_chunks,
                            additive_limit=max(1, int(self.gap_detector_additive_limit)),
                        )
                    if (
                        sentence_enabled
                        and hasattr(self.mid_memory, "search_sentences_global_with_limit")
                    ):
                        gap_sentences: List[Dict[str, object]] = []
                        for gq in gap_queries:
                            gap_sentences.extend(
                                self.mid_memory.search_sentences_global_with_limit(
                                    gq,
                                    top_n=max(4, int(self.gap_detector_top_n_per_query)),
                                )
                            )
                        gap_sentences = self._dedup_chunks_keep_best(gap_sentences)
                        if gap_sentences:
                            merged_units = self._merge_anchor_chunks(
                                base_chunks=merged_units,
                                extra_chunks=gap_sentences,
                                additive_limit=max(1, int(self.gap_detector_additive_limit)),
                            )
                    merged_units = self._apply_slot_coverage_rerank(merged_units, query_plan)
                    logger.info(
                        "MemoryManager.retrieve_context: "
                        f"gap_missing={missing_slots}, gap_queries={gap_queries}."
                    )
        context_parts: List[str] = []
        for index, item in enumerate(merged_units, start=1):
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            context_parts.append(f"[Chunk {index}]\n{text}")
        context_text = "\n\n".join(context_parts)
        return context_text, [], merged_units

    @staticmethod
    def _top_text_items(items: List[Dict[str, object]], limit: int = 5) -> List[Dict[str, object]]:
        ranked = sorted(items, key=lambda x: float(x.get("score", 0.0)), reverse=True)
        out: List[Dict[str, object]] = []
        for item in ranked[: max(1, int(limit))]:
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            out.append(
                {
                    "text": text,
                    "score": float(item.get("score", 0.0)),
                    "chunk_id": int(item.get("chunk_id", 0) or 0),
                    "session_date": str(item.get("session_date", "")),
                }
            )
        return out

    def _build_plan_combined_evidence(
        self,
        *,
        chunks: List[Dict[str, object]],
        plan: Dict[str, object],
        limit: int = 5,
    ) -> List[Dict[str, object]]:
        focus_phrases = [
            str(x).strip()
            for x in (
                list(plan.get("focus_phrases", []))
                + list(plan.get("entities", []))
                + list(plan.get("compare_options", []))
                + list(plan.get("state_keys", []))
            )
            if str(x).strip()
        ]
        target_object = str(plan.get("target_object", "")).strip()
        if target_object:
            focus_phrases.insert(0, target_object)
        dedup_focus: List[str] = []
        seen_focus: set[str] = set()
        for phrase in focus_phrases:
            low = phrase.lower()
            if low in seen_focus:
                continue
            seen_focus.add(low)
            dedup_focus.append(phrase)
            if len(dedup_focus) >= 8:
                break

        plan_keywords = [
            str(x).strip().lower()
            for x in (
                list(plan.get("must_keywords", []))
                + list(plan.get("constraint_keywords", []))
                + list(plan.get("plan_keywords", []))
            )
            if str(x).strip()
        ]

        def _contains_phrase(text_low: str, phrase: str) -> bool:
            p = str(phrase or "").strip().lower()
            if not p:
                return False
            ptoks = re.findall(r"[a-z0-9]+", p)
            ttoks = set(re.findall(r"[a-z0-9]+", text_low))
            if not ptoks or not ttoks:
                return False
            if len(ptoks) == 1:
                return ptoks[0] in ttoks
            return all(tok in ttoks for tok in ptoks)

        ranked: List[Dict[str, object]] = []
        for item in chunks:
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            low = text.lower()
            phrase_hits = sum(1 for phrase in dedup_focus if _contains_phrase(low, phrase))
            keyword_hits = sum(1 for kw in plan_keywords if kw and kw in low)
            if phrase_hits <= 0 and keyword_hits <= 1:
                continue
            score = (
                float(item.get("score", 0.0))
                + (0.26 * min(phrase_hits, 3))
                + (0.08 * min(keyword_hits, 4))
            )
            ranked.append(
                {
                    "text": text,
                    "score": score,
                    "chunk_id": int(item.get("chunk_id", 0) or 0),
                    "session_date": str(item.get("session_date", "")),
                    "channel": "plan_combined_evidence",
                }
            )
        ranked.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        out: List[Dict[str, object]] = []
        seen: set[str] = set()
        for item in ranked:
            norm = re.sub(r"\s+", " ", str(item.get("text", "")).strip().lower())
            if not norm or norm in seen:
                continue
            seen.add(norm)
            out.append(item)
            if len(out) >= max(1, int(limit)):
                break
        return out

    def _build_evidence_graph_source(
        self,
        *,
        query: str,
        chunks: List[Dict[str, object]],
        evidence_sentences: List[Dict[str, object]],
        evidence_pack: Dict[str, object],
        plan: Dict[str, object],
    ) -> List[Dict[str, object]]:
        answer_type = str(plan.get("answer_type", "")).strip().lower()
        rag_items = [
            {**item, "channel": "rag_evidence"}
            for item in self._top_text_items(evidence_sentences, limit=6)
        ]
        if not rag_items:
            rag_items = [
                {**item, "channel": "rag_evidence"}
                for item in self._top_text_items(chunks, limit=4)
            ]
        pack_items: List[Dict[str, object]] = []
        for raw in list(evidence_pack.get("lines", []))[:8]:
            text = str(raw).strip()
            if text.startswith("- "):
                text = text[2:].strip()
            if not text:
                continue
            pack_items.append(
                {
                    "text": text,
                    "score": 0.0,
                    "chunk_id": 0,
                    "session_date": "",
                    "channel": "evidence_pack",
                }
            )
        if answer_type == "temporal_comparison":
            merged: List[Dict[str, object]] = []
            seen: set[str] = set()
            for item in pack_items:
                text = str(item.get("text", "")).strip()
                key = re.sub(r"\s+", " ", text.lower())
                if not text or key in seen:
                    continue
                seen.add(key)
                merged.append(
                    {
                        "text": text,
                        "score": 0.92,
                        "chunk_id": 0,
                        "session_date": "",
                        "channel": "evidence_pack",
                    }
                )
            return merged
        plan_items = self._build_plan_combined_evidence(chunks=chunks, plan=plan, limit=6)

        merged: List[Dict[str, object]] = []
        seen: set[str] = set()
        for item in rag_items + pack_items + plan_items:
            text = str(item.get("text", "")).strip()
            key = re.sub(r"\s+", " ", text.lower())
            if not text or key in seen:
                continue
            seen.add(key)
            merged.append(
                {
                    "text": text,
                    "score": float(item.get("score", 0.0)),
                    "chunk_id": int(item.get("chunk_id", 0) or 0),
                    "session_date": str(item.get("session_date", "")),
                    "channel": str(item.get("channel", "")),
                }
            )
            if len(merged) >= 24:
                break
        return merged

    def build_evidence_graph_bundle(
        self,
        query: str,
        precomputed_context: Optional[Tuple[str, List[Dict[str, object]], List[Dict[str, object]]]] = None,
        evidence_sentences: Optional[List[Dict[str, object]]] = None,
        evidence_pack: Optional[Dict[str, object]] = None,
        enable_filter: Optional[bool] = None,
        enable_claims: Optional[bool] = None,
        enable_light_graph: Optional[bool] = None,
    ) -> Dict[str, object]:
        started_total = time.perf_counter()
        if precomputed_context is None:
            _context_text, _topics, chunks = self.retrieve_context(query)
        else:
            _context_text, _topics, chunks = precomputed_context
            if not self.last_query_plan:
                self.last_query_plan = self._build_query_plan(query)
        plan = dict(self.last_query_plan or self._build_query_plan(query))
        use_filter = self.evidence_filter_enabled if enable_filter is None else bool(enable_filter)
        use_claims = (
            self.evidence_graph_claims_enabled if enable_claims is None else bool(enable_claims)
        )
        use_light_graph = (
            self.evidence_light_graph_enabled
            if enable_light_graph is None
            else bool(enable_light_graph)
        )
        if use_light_graph and not use_claims:
            use_claims = True
        if use_claims and not use_filter:
            use_filter = True
        stage_latency_sec: Dict[str, float] = {
            "filter": 0.0,
            "claims": 0.0,
            "light_graph": 0.0,
            "graph_total": 0.0,
        }
        evidence_sentences = list(evidence_sentences or [])
        if not evidence_sentences:
            evidence_sentences = self.answer_grounding.collect_evidence_sentences(query, chunks)
        pack_payload = dict(evidence_pack or {})
        if not pack_payload:
            pack_payload = self.chat_runtime._build_evidence_pack(
                query=query,
                evidence_sentences=evidence_sentences,
                chunks=chunks,
            )
        unified_source = self._build_evidence_graph_source(
            query=query,
            chunks=chunks,
            evidence_sentences=evidence_sentences,
            evidence_pack=pack_payload,
            plan=plan,
        )
        filtered_pack: Dict[str, object] = {}
        if use_filter:
            stage_started = time.perf_counter()
            filtered_pack = self.evidence_filter.build_filtered_pack(
                query=query,
                query_plan=plan,
                unified_source=unified_source,
            )
            stage_latency_sec["filter"] = time.perf_counter() - stage_started
        extraction: Dict[str, object] = {
            "enabled": False,
            "model": getattr(self.evidence_graph_extractor, "model", ""),
            "support_units": [],
            "claims": [],
            "raw_batches": [],
            "stats": {
                "selected_evidence": 0,
                "batches": 0,
                "support_units": 0,
                "claims": 0,
            },
        }
        if use_claims and filtered_pack:
            stage_started = time.perf_counter()
            extraction = self.evidence_graph_extractor.extract_claims(filtered_pack)
            stage_latency_sec["claims"] = time.perf_counter() - stage_started
        graph: Dict[str, object] = {"query": str(query or ""), "answer_type": str(plan.get("answer_type", "")), "nodes": [], "edges": [], "stats": {"node_count": 0, "edge_count": 0, "entity_count": 0, "claim_count": 0}}
        if use_light_graph and filtered_pack:
            stage_started = time.perf_counter()
            graph = self.evidence_light_graph.build_graph(
                query=query,
                filtered_pack=filtered_pack,
                claims=list(extraction.get("claims", [])),
            )
            stage_latency_sec["light_graph"] = time.perf_counter() - stage_started
        stage_latency_sec["graph_total"] = time.perf_counter() - started_total
        bundle = {
            "query": str(query or ""),
            "query_plan": plan,
            "stage_flags": {
                "filter_enabled": use_filter,
                "claims_enabled": use_claims,
                "light_graph_enabled": use_light_graph,
            },
            "stage_latency_sec": stage_latency_sec,
            "unified_source": unified_source,
            "filtered_pack": filtered_pack,
            "claim_result": extraction,
            "light_graph": graph,
        }
        self.last_evidence_graph_bundle = bundle
        return bundle

    def ingest_message(self, message: Message) -> None:
        """Ingest one dataset message into memory without calling the LLM."""
        role = str(message.get("role", "user")).strip().lower() or "user"
        content = str(message.get("content", "")).strip()
        if not content:
            return
        normalized: Message = {"role": role, "content": content}
        for key in ("session_id", "session_date", "turn_index"):
            if key in message:
                normalized[key] = message.get(key)
        if bool(message.get("has_answer", False)):
            normalized["has_answer"] = True
        logger.debug(f"MemoryManager.ingest_message: role={role}, content_len={len(content)}")
        self.short_memory.add(normalized)
        self.short_memory.flush_to_mid_memory(self.mid_memory)

    def finalize_ingest(self) -> None:
        """Flush pending dynamic chunk buffer after dataset ingestion."""
        self.mid_memory.flush_pending()
        logger.info("MemoryManager.finalize_ingest: flushed pending mid-memory buffer.")

    def archive_short_to_mid(self, clear_short: bool = True) -> int:
        """Persist all current short-memory messages into mid memory."""
        pending = self.short_memory.get()
        moved = 0
        for message in pending:
            self.mid_memory.add(message)
            moved += 1
        self.mid_memory.flush_pending()
        if clear_short:
            self.short_memory.clear()
        logger.info(
            "MemoryManager.archive_short_to_mid: "
            f"moved={moved}, clear_short={clear_short}."
        )
        return moved

    def reset_for_new_instance(self) -> None:
        """Reset short and mid memory for isolated per-instance evaluation."""
        self.short_memory.clear()
        self.mid_memory.clear_all()
        self.long_memory.clear_all()
        logger.info("MemoryManager.reset_for_new_instance: memory reset completed.")

    def chat(
        self,
        input_text: str,
        retrieval_query: Optional[str] = None,
        precomputed_context: Optional[Tuple[str, List[Dict[str, object]], List[Dict[str, object]]]] = None,
    ) -> str:
        """Handle one user message with retrieval, LLM call, and memory updates."""
        chat_started = time.perf_counter()
        self.last_prompt_eval_chunks = []
        self.last_stage_latency_sec = {}
        logger.info(f"MemoryManager.chat: user input='{input_text}'")
        query = retrieval_query if retrieval_query is not None else input_text
        (
            retrieved_context_text,
            _topics,
            chunks,
            evidence_sentences,
        ) = self._prepare_answer_inputs(query, precomputed_context)
        prompt_text = self._build_generation_prompt(
            input_text=input_text,
            retrieved_context_text=retrieved_context_text,
            evidence_sentences=evidence_sentences,
            chunks=chunks,
        )
        ai_response, fallback_path, not_found_reason = self._generate_final_answer(
            input_text=input_text,
            query=query,
            prompt_text=prompt_text,
            evidence_sentences=evidence_sentences,
        )
        logger.info(
            "MemoryManager.answer_debug: "
            f"fallback_path={fallback_path}, "
            f"not_found_reason={not_found_reason or 'none'}, "
            "best evidence routing fallback chain disabled."
        )
        logger.info(f"MemoryManager.chat: LLM response='{ai_response}'")
        self._record_turn(input_text, ai_response)
        self.last_stage_latency_sec["chat_total"] = time.perf_counter() - chat_started
        return ai_response

    def _prepare_answer_inputs(
        self,
        query: str,
        precomputed_context: Optional[Tuple[str, List[Dict[str, object]], List[Dict[str, object]]]],
    ) -> Tuple[
        str,
        List[Dict[str, object]],
        List[Dict[str, object]],
        List[Dict[str, object]],
    ]:
        return self.chat_runtime.prepare_answer_inputs(
            query=query,
            precomputed_context=precomputed_context,
        )

    def _set_prompt_eval_chunks(
        self, generation_context: List[Dict[str, str]] | str
    ) -> None:
        if isinstance(generation_context, str):
            text = str(generation_context).strip()
            self.last_prompt_eval_chunks = [{"section": "prompt", "text": text}] if text else []
            return
        sections: List[Dict[str, str]] = []
        for item in generation_context:
            section = str(item.get("section", "")).strip()
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            payload = {"text": text}
            if section:
                payload["section"] = section
            sections.append(payload)
        self.last_prompt_eval_chunks = sections

    def _build_generation_prompt(
        self,
        *,
        input_text: str,
        retrieved_context_text: str,
        evidence_sentences: List[Dict[str, object]],
        chunks: List[Dict[str, object]],
    ) -> str:
        return self.chat_runtime.build_generation_prompt(
            input_text=input_text,
            retrieved_context_text=retrieved_context_text,
            evidence_sentences=evidence_sentences,
            chunks=chunks,
        )

    def offline_build_long_graph_from_chunks(
        self,
        chunks: List[Dict[str, object]],
        query: Optional[str] = None,
    ) -> int:
        """Deprecated compatibility hook.

        The active pipeline builds only question-scoped evidence graphs, so this
        method intentionally does nothing and returns 0.
        """
        _ = chunks
        _ = query
        return 0

    def _generate_final_answer(
        self,
        *,
        input_text: str,
        query: str,
        prompt_text: str,
        evidence_sentences: List[Dict[str, object]],
    ) -> Tuple[str, str, str]:
        return self.chat_runtime.generate_final_answer(
            input_text=input_text,
            query=query,
            prompt_text=prompt_text,
            evidence_sentences=evidence_sentences,
        )

    def get_last_prompt_eval_chunks(self) -> List[Dict[str, str]]:
        """Return prompt-grounded chunks used by the most recent chat call."""
        return [
            {
                "section": str(item.get("section", "")),
                "text": str(item.get("text", "")),
            }
            for item in self.last_prompt_eval_chunks
        ]

    def close(self) -> None:
        """Close owned resources."""
        self.long_memory.close()
        self.mid_memory.close()

    def _record_turn(self, user_input: str, assistant_output: str) -> None:
        """Persist one user-assistant turn into short memory and flush overflow."""
        user_message: Message = {"role": "user", "content": user_input}
        assistant_message: Message = {"role": "assistant", "content": assistant_output}
        self.short_memory.add(user_message)
        self.short_memory.add(assistant_message)
        self.short_memory.flush_to_mid_memory(self.mid_memory)
