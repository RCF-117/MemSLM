"""Chat orchestration runtime extracted from MemoryManager."""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple

from llm_long_memory.memory.lexical_resources import BASIC_STOPWORDS, UPDATE_CUES
from llm_long_memory.utils.logger import logger


class MemoryManagerChatRuntime:
    """Keep MemoryManager.chat-related orchestration out of the main class body."""

    def __init__(self, manager: Any) -> None:
        self.m = manager
        self._last_specialist_payload: Dict[str, object] = {}
        self._last_evidence_pack: Dict[str, object] = {}
        self._last_compact_prompt_text = ""
        self._last_expanded_prompt_text = ""
        self._last_compact_prompt_support_sources: List[Dict[str, object]] = []
        self._last_expanded_prompt_support_sources: List[Dict[str, object]] = []
        self._last_compact_route_packet: Dict[str, object] = {}
        self._last_expanded_route_packet: Dict[str, object] = {}

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", str(text or "").lower())

    @staticmethod
    def _normalize_space(text: str) -> str:
        return " ".join(str(text or "").split())

    @staticmethod
    def _contains_digit_or_number_word(text: str) -> bool:
        low = str(text or "").lower()
        if re.search(r"\b\d+\b", low):
            return True
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
            "once",
            "twice",
        }
        toks = set(re.findall(r"[a-z0-9]+", low))
        return bool(number_words.intersection(toks))

    def _is_not_found_like(self, normalized: str) -> bool:
        norm = str(normalized or "").strip().lower()
        if not norm:
            return False
        return (
            norm == "not found"
            or norm.startswith("not found in retrieved")
            or norm.startswith("not found in the retrieved")
            or norm.startswith("not found in context")
        )

    def _maybe_get_toolkit_direct_answer(self, query: str) -> Tuple[str, str]:
        if not bool(getattr(self.m, "toolkit_direct_answer_enabled", False)):
            return "", ""
        tool_payload = dict(dict(self._last_specialist_payload or {}).get("tool_payload", {}) or {})
        if not bool(tool_payload.get("verified", False)):
            return "", ""
        intent = self._normalize_space(str(tool_payload.get("intent", ""))).lower()
        allowed_intents = {
            str(x).strip().lower()
            for x in set(getattr(self.m, "toolkit_direct_allowed_intents", set()) or set())
            if str(x).strip()
        }
        if intent not in allowed_intents:
            return "", ""
        confidence = float(tool_payload.get("confidence", 0.0) or 0.0)
        if confidence < float(getattr(self.m, "toolkit_direct_min_confidence", 0.65)):
            return "", ""
        verification_reason = self._normalize_space(
            str(tool_payload.get("verification_reason", ""))
        ).lower()
        if intent == "update" and verification_reason not in {
            "update_edge_verified",
            "update_numeric_compare_verified",
        }:
            return "", ""
        if intent == "count" and verification_reason not in {
            "count_verified_by_enumeration",
            "count_verified_by_duplicate_count_claims",
            "count_verified_by_latest_supported_state",
        }:
            return "", ""
        if (
            intent == "temporal_count"
            and verification_reason != "temporal_count_dual_anchor_verified"
        ):
            return "", ""
        if intent == "temporal_compare" and verification_reason not in {
            "temporal_compare_graph_edge_verified",
            "temporal_compare_dual_dates_verified",
        }:
            return "", ""
        candidate = self._normalize_space(
            str(tool_payload.get("verified_candidate", ""))
            or str(tool_payload.get("answer_candidate", ""))
        )
        if not candidate:
            return "", ""
        normalized = self.m.answer_grounding.normalize_final_answer(candidate, query)
        if not normalized:
            return "", ""
        return normalized, f"toolkit_direct:{verification_reason}"

    def _score_overlap(self, query: str, sentence: str) -> float:
        q = set(self._tokenize(query))
        s = set(self._tokenize(sentence))
        if not q or not s:
            return 0.0
        return float(len(q.intersection(s))) / float(len(q))

    def _build_evidence_pack(
        self,
        *,
        query: str,
        evidence_sentences: List[Dict[str, object]],
        chunks: Optional[List[Dict[str, object]]] = None,
    ) -> Dict[str, object]:
        plan = dict(getattr(self.m, "last_query_plan", {}) or {})
        answer_type = str(plan.get("answer_type", "factoid")).strip().lower()
        top = list(evidence_sentences[:12])
        if not top:
            return {"answer_type": answer_type, "lines": [], "insufficient": True}
        chunk_pool = list(chunks or [])
        query_tokens = set(self._tokenize(query))
        temporal_keywords = {
            "first",
            "second",
            "third",
            "earlier",
            "later",
            "before",
            "after",
            "when",
            "date",
            "time",
            "since",
            "until",
            "during",
            "ago",
            "previous",
            "next",
            "recent",
            "latest",
            "last",
            "current",
            "today",
            "yesterday",
            "tomorrow",
            "day",
            "week",
            "month",
            "year",
        }
        generic_stopwords = set(BASIC_STOPWORDS).union(
            {
                "who",
                "what",
                "which",
                "where",
                "when",
                "why",
                "how",
                "does",
                "was",
                "were",
                "have",
                "has",
                "had",
                "we",
                "they",
                "he",
                "she",
                "it",
                "and",
                "or",
                "our",
                "your",
                "their",
                "us",
                "them",
            }
        )

        def _contains_phrase(text_low: str, phrase: str) -> bool:
            p = str(phrase).strip().lower()
            if not p:
                return False
            p_tokens = self._tokenize(p)
            if not p_tokens:
                return False
            t_tokens = set(self._tokenize(text_low))
            if len(p_tokens) == 1:
                return p_tokens[0] in t_tokens
            return all(tok in t_tokens for tok in p_tokens)

        def _pick(limit: int, must: Optional[List[str]] = None) -> List[str]:
            out: List[str] = []
            seen: set[str] = set()
            must_l = [str(x).strip().lower() for x in (must or []) if str(x).strip()]
            for item in top:
                text = self._normalize_space(str(item.get("text", "")))
                if not text:
                    continue
                low = text.lower()
                if must_l and (not any(_contains_phrase(low, m) for m in must_l)):
                    continue
                key = low
                if key in seen:
                    continue
                seen.add(key)
                out.append(text)
                if len(out) >= max(1, int(limit)):
                    break
            return out

        def _split_sentences(text: str) -> List[str]:
            parts = re.split(r"(?<=[.!?。！？])\s+|\n+", str(text or ""))
            return [self._normalize_space(x) for x in parts if self._normalize_space(x)]

        def _option_sentence_pool(
            option: str,
            raw_texts: List[str],
            *,
            limit: int = 3,
            signal_terms: Optional[set[str]] = None,
        ) -> List[str]:
            out: List[tuple[float, str]] = []
            seen: set[str] = set()
            opt = str(option).strip()
            opt_tokens = set(self._tokenize(opt))
            sig_terms = set(signal_terms or set())

            for txt in raw_texts:
                for sent in _split_sentences(txt):
                    low = sent.lower()
                    if not _contains_phrase(low, opt):
                        continue
                    key = low
                    if key in seen:
                        continue
                    seen.add(key)

                    stoks = set(self._tokenize(low))
                    score = 0.0
                    if opt_tokens and stoks:
                        score += len(opt_tokens.intersection(stoks)) / float(len(opt_tokens))
                    if query_tokens and stoks:
                        score += 0.5 * (
                            len(query_tokens.intersection(stoks)) / float(len(query_tokens))
                        )
                    if stoks.intersection(temporal_keywords):
                        score += 0.6
                    if sig_terms and stoks.intersection(sig_terms):
                        score += 0.8
                    # prevent location/profile-only noise from dominating temporal compare
                    if any(
                        x in stoks
                        for x in {"live", "hometown", "town", "city", "culture", "history"}
                    ):
                        score -= 0.2
                    compact = sent if len(sent) <= 260 else (sent[:260].rstrip(" ,.;:!?") + "...")
                    out.append((score, compact))

            out.sort(key=lambda x: x[0], reverse=True)
            picked: List[str] = []
            used: set[str] = set()
            for _, sent in out:
                k = sent.lower()
                if k in used:
                    continue
                used.add(k)
                picked.append(sent)
                if len(picked) >= max(1, int(limit)):
                    break
            return picked

        lines: List[str] = []
        insufficient = False
        if answer_type == "count":
            target = str(plan.get("target_object", "")).strip()
            if not target:
                focus_list = [
                    str(x).strip() for x in list(plan.get("focus_phrases", [])) if str(x).strip()
                ]
                if focus_list:
                    target = focus_list[0]
            count_unit = str(plan.get("count_unit", "")).strip().lower()
            target_tokens = set(self._tokenize(target))

            def _object_alignment(text: str) -> float:
                low = text.lower()
                stoks = set(self._tokenize(low))
                if target and _contains_phrase(low, target):
                    return 1.0
                if target_tokens and stoks:
                    return len(target_tokens.intersection(stoks)) / float(len(target_tokens))
                return 0.0

            def _count_fact_score(text: str) -> float:
                low = text.lower()
                align = _object_alignment(text)
                if target and align <= 0.0:
                    return -1.0
                score = (0.9 * align) + (0.4 * self._score_overlap(query, text))
                if self._contains_digit_or_number_word(text):
                    score += 0.25
                first_person_fact = bool(
                    re.search(
                        r"\b(i|my|we|our)\b.{0,40}\b(have|had|own|owned|bought|got|kept|led|managed|tried|completed|serviced|plan(?:ned)?|attended)\b",
                        low,
                    )
                )
                if first_person_fact:
                    score += 0.15
                if count_unit and count_unit in low:
                    score += 0.1
                if re.search(r"\b(tips?|how to|guide|example script)\b", low):
                    score -= 0.5
                return score

            if target:
                lines.append(f"target_object: {target}")
                ranked: List[tuple[float, str]] = []
                seen_count: set[str] = set()
                for item in top:
                    text = self._normalize_space(str(item.get("text", "")))
                    if not text:
                        continue
                    low = text.lower()
                    if _object_alignment(text) < 0.5:
                        continue
                    score = _count_fact_score(text)
                    if score < 0.58:
                        continue
                    key = low
                    if key in seen_count:
                        continue
                    seen_count.add(key)
                    ranked.append((score, text))
                ranked.sort(key=lambda x: x[0], reverse=True)
                picked = [text for _, text in ranked[:6]]
                if len(picked) < 2:
                    relaxed: List[tuple[float, str]] = []
                    for item in top:
                        text = self._normalize_space(str(item.get("text", "")))
                        if not text:
                            continue
                        low = text.lower()
                        if low in seen_count:
                            continue
                        if _object_alignment(text) < 0.34:
                            continue
                        score = _count_fact_score(text)
                        if score < 0.44:
                            continue
                        relaxed.append((score, text))
                    relaxed.sort(key=lambda x: x[0], reverse=True)
                    picked.extend([text for _, text in relaxed[: max(0, 6 - len(picked))]])
                if not picked:
                    insufficient = True
                    picked = _pick(limit=4, must=[target])
            else:
                picked = _pick(limit=5)
            lines.extend([f"- {x}" for x in picked[:6]])
        elif answer_type == "temporal_comparison":
            options = [
                str(x).strip() for x in list(plan.get("compare_options", [])) if str(x).strip()
            ]
            if len(options) < 2:
                focus_list = [
                    str(x).strip() for x in list(plan.get("focus_phrases", [])) if str(x).strip()
                ]
                options = focus_list[:2]
            if len(options) >= 2:
                a, b = options[0], options[1]
                pool_a: List[str] = []
                pool_b: List[str] = []
                option_tokens_all: set[str] = set()
                for opt in options:
                    option_tokens_all.update(self._tokenize(opt))
                # Dynamic query signals: query content words excluding options + stopwords.
                signal_terms = {
                    tok
                    for tok in query_tokens
                    if (
                        len(tok) >= 3
                        and tok not in generic_stopwords
                        and tok not in option_tokens_all
                    )
                }
                signal_terms.update({tok for tok in query_tokens if tok in temporal_keywords})

                # 1) Dedicated per-option retrieval pool: each option gets its own top-3 chunk search.
                top_n = 3
                cfg = dict(self.m.config.get("retrieval", {}).get("compare_option_pool", {}))
                if cfg:
                    top_n = max(1, int(cfg.get("top_n_chunks_per_option", 3)))
                search_limit = max(top_n, 4)

                def _retrieve_option_pool(option: str) -> List[str]:
                    out: List[str] = []
                    seen: set[str] = set()
                    option_low = option.lower()
                    option_tokens = set(self._tokenize(option_low))
                    # Remove other compare options from query so each pool is object-centric.
                    cleaned_query = str(query)
                    for other in options:
                        if other.strip().lower() == option_low:
                            continue
                        cleaned_query = re.sub(
                            re.escape(other),
                            " ",
                            cleaned_query,
                            flags=re.IGNORECASE,
                        )
                    cleaned_query = self._normalize_space(cleaned_query)

                    def _match_option(text: str) -> bool:
                        low = text.lower()
                        if _contains_phrase(low, option):
                            return True
                        txt_tokens = set(self._tokenize(low))
                        if not option_tokens or not txt_tokens:
                            return False
                        overlap = len(option_tokens.intersection(txt_tokens)) / float(
                            max(1, len(option_tokens))
                        )
                        return overlap >= 0.66

                    if hasattr(self.m.mid_memory, "search_chunks_global_with_limit"):
                        queries = [
                            f"{cleaned_query} {option}".strip(),
                            (f"first {option}" if "first" in query.lower() else "").strip(),
                            option,
                        ]
                        for oq in queries:
                            if not oq:
                                continue
                            for item in self.m.mid_memory.search_chunks_global_with_limit(
                                oq,
                                top_n=search_limit,
                            ):
                                text = self._normalize_space(str(item.get("text", "")))
                                if not text:
                                    continue
                                if not _match_option(text):
                                    continue
                                key = text.lower()
                                if key in seen:
                                    continue
                                seen.add(key)
                                out.append(text)
                                if len(out) >= top_n:
                                    return out
                    return out

                pool_a = _retrieve_option_pool(a)
                pool_b = _retrieve_option_pool(b)

                # 2) Fallback to existing mixed evidence if dedicated pool is sparse.
                if not pool_a:
                    pool_a = _pick(limit=3, must=[a])
                if not pool_b:
                    pool_b = _pick(limit=3, must=[b])

                # 3) Last fallback from current chunk pool by overlap with each option.
                if chunk_pool and (len(pool_a) < 2 or len(pool_b) < 2):
                    for item in chunk_pool:
                        text = self._normalize_space(str(item.get("text", "")))
                        if not text:
                            continue
                        low = text.lower()
                        if _contains_phrase(low, a) and text not in pool_a and len(pool_a) < top_n:
                            pool_a.append(text)
                        if _contains_phrase(low, b) and text not in pool_b and len(pool_b) < top_n:
                            pool_b.append(text)
                        if len(pool_a) >= top_n and len(pool_b) >= top_n:
                            break

                # Convert chunk pool -> compact sentence pool with action/time relevance.
                pool_a = _option_sentence_pool(
                    a,
                    pool_a,
                    limit=3,
                    signal_terms=signal_terms,
                )
                pool_b = _option_sentence_pool(
                    b,
                    pool_b,
                    limit=3,
                    signal_terms=signal_terms,
                )

                if not pool_a or not pool_b:
                    insufficient = True
                lines.append(f"option_a: {a}")
                lines.extend([f"- {x}" for x in pool_a[:3]])
                lines.append(f"option_b: {b}")
                lines.extend([f"- {x}" for x in pool_b[:3]])
                lines.append(
                    "- compare_rule: prioritize evidence matching query action/time cues and choose the earlier timeline."
                )
            else:
                lines.extend([f"- {x}" for x in _pick(limit=6)])
        elif answer_type == "temporal":
            terms = [str(x).strip() for x in list(plan.get("time_terms", [])) if str(x).strip()]
            if not terms:
                terms = [
                    str(x).strip() for x in list(plan.get("focus_phrases", [])) if str(x).strip()
                ][:2]
            picked = _pick(limit=6, must=terms[:2]) if terms else _pick(limit=6)
            if terms and not picked:
                insufficient = True
                picked = _pick(limit=4)
            lines.extend([f"- {x}" for x in picked[:6]])
        elif answer_type == "update":
            state_keys = [
                str(x).strip() for x in list(plan.get("state_keys", [])) if str(x).strip()
            ]
            anchors = state_keys[:2]
            if not anchors:
                target = str(plan.get("target_object", "")).strip()
                if target:
                    anchors = [target]
                else:
                    anchors = [
                        str(x).strip()
                        for x in list(plan.get("focus_phrases", []))
                        if str(x).strip()
                    ][:1]
            if not anchors:
                anchors = [
                    str(x).strip() for x in list(plan.get("entities", [])) if str(x).strip()
                ][:2]

            def _matches_anchor(text: str, anchor: str) -> bool:
                low = text.lower()
                if _contains_phrase(low, anchor):
                    return True
                anchor_tokens = set(self._tokenize(anchor))
                text_tokens = set(self._tokenize(low))
                if not anchor_tokens or not text_tokens:
                    return False
                overlap = len(anchor_tokens.intersection(text_tokens)) / float(len(anchor_tokens))
                return overlap >= 0.5

            update_cues = set(UPDATE_CUES)
            ranked_update: List[tuple[tuple[str, float], str]] = []
            seen_update: set[str] = set()
            for item in top:
                text = self._normalize_space(str(item.get("text", "")))
                if not text:
                    continue
                low = text.lower()
                if anchors and not any(_matches_anchor(text, a) for a in anchors):
                    continue
                score = float(item.get("score", 0.0))
                text_tokens = set(self._tokenize(low))
                if update_cues.intersection(text_tokens):
                    score += 0.35
                if re.search(r"\b(now|currently|recently|latest|last)\b", low):
                    score += 0.2
                if re.search(r"\b(tips?|how to|guide|example script)\b", low):
                    score -= 0.4
                key = low
                if key in seen_update:
                    continue
                seen_update.add(key)
                ranked_update.append(((str(item.get("session_date", "")), score), text))

            ranked_update.sort(key=lambda x: x[0], reverse=True)
            picked: List[str] = [text for _, text in ranked_update[:6]]
            if not picked:
                insufficient = True
                recent = sorted(
                    top,
                    key=lambda x: str(x.get("session_date", "")),
                    reverse=True,
                )
                picked = [self._normalize_space(str(x.get("text", ""))) for x in recent[:4]]
            lines.extend([f"- {x}" for x in picked[:6] if x])
        elif answer_type == "preference":
            picked = _pick(
                limit=6, must=["prefer", "favorite", "recommend", "suggest", "like", "enjoy"]
            )
            if not picked:
                picked = _pick(limit=5)
            lines.extend([f"- {x}" for x in picked[:6]])
        else:
            lines.extend([f"- {x}" for x in _pick(limit=6)])

        if not lines:
            insufficient = True
            lines = [
                f"- {self._normalize_space(str(x.get('text', '')))}"
                for x in top[:4]
                if str(x.get("text", "")).strip()
            ]

        return {
            "answer_type": answer_type,
            "lines": lines[:14],
            "insufficient": bool(insufficient),
            "plan": plan,
        }

    def _format_evidence_pack(self, pack: Dict[str, object]) -> str:
        lines = [str(x).strip() for x in list(pack.get("lines", [])) if str(x).strip()]
        if not lines:
            return ""
        return "[Evidence Pack]\n" + "\n".join(lines)

    @staticmethod
    def _claim_to_text(claim: Dict[str, object]) -> str:
        subject = str(claim.get("subject", "")).strip()
        predicate = str(claim.get("predicate", "")).strip()
        value = str(claim.get("value", "")).strip()
        if not subject or not predicate or not value:
            return ""
        parts = [f"{subject} | {predicate} | {value}"]
        time_anchor = str(claim.get("time_anchor", "")).strip()
        status = str(claim.get("status", "")).strip().lower()
        if time_anchor:
            parts.append(f"time={time_anchor}")
        if status and status != "unknown":
            parts.append(f"status={status}")
        return " | ".join(parts)

    def graph_bundle_to_chunks(
        self,
        bundle: Dict[str, object],
        *,
        max_claims: int = 6,
        max_evidence: int = 4,
    ) -> List[Dict[str, str]]:
        chunks: List[Dict[str, str]] = []
        claim_result = dict(bundle.get("claim_result", {}) or {})
        claims = list(claim_result.get("claims", []))
        support_units = list(claim_result.get("support_units", []))
        seen: set[str] = set()
        for claim in claims[: max(1, int(max_claims))]:
            text = self._normalize_space(self._claim_to_text(dict(claim)))
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            chunks.append({"text": text, "section": "graph_claim"})

        if not chunks:
            for unit in support_units[: max(1, int(max_claims))]:
                text = self._normalize_space(
                    str(unit.get("verbatim_span", "")) or str(unit.get("text", ""))
                )
                if not text:
                    continue
                key = text.lower()
                if key in seen:
                    continue
                seen.add(key)
                chunks.append({"text": text, "section": "graph_support_unit"})

        if not chunks:
            filtered = dict(bundle.get("filtered_pack", {}) or {})
            selected = list(filtered.get("core_evidence", [])) + list(
                filtered.get("supporting_evidence", [])
            )
            for item in selected[: max(1, int(max_evidence))]:
                text = self._normalize_space(str(item.get("text", "")))
                if not text:
                    continue
                key = text.lower()
                if key in seen:
                    continue
                seen.add(key)
                chunks.append({"text": text, "section": "graph_evidence"})

        conflict_items = list(
            dict(bundle.get("filtered_pack", {}) or {}).get("conflict_evidence", [])
        )
        for item in conflict_items[:2]:
            text = self._normalize_space(str(item.get("text", "")))
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            chunks.append({"text": f"Conflict candidate: {text}", "section": "graph_conflict"})
        return chunks

    def prepare_answer_inputs(
        self,
        query: str,
        precomputed_context: Optional[Tuple[str, List[Dict[str, object]], List[Dict[str, object]]]],
    ) -> Tuple[
        str,
        List[Dict[str, object]],
        List[Dict[str, object]],
        List[Dict[str, object]],
    ]:
        stage_latency_sec: Dict[str, float] = {
            "rag": 0.0,
            "filter": 0.0,
            "claims": 0.0,
            "light_graph": 0.0,
            "toolkit": 0.0,
        }
        rag_started = time.perf_counter()
        if precomputed_context is not None:
            context_text, topics, chunks = precomputed_context
        else:
            context_text, topics, chunks = self.m.retrieve_context(query)
        logger.info(f"MemoryManager.chat: retrieved chunks={len(chunks)}")

        if self.m.retrieval_execution_mode in {"model_only", "naive_rag"}:
            evidence_sentences: List[Dict[str, object]] = []
            self._last_specialist_payload = {}
            stage_latency_sec["rag"] = time.perf_counter() - rag_started
        elif self.m.retrieval_execution_mode == "filter_only":
            raw_evidence_sentences = self.m.answer_grounding.collect_evidence_sentences(
                query, chunks
            )
            self._last_evidence_pack = self._build_evidence_pack(
                query=query,
                evidence_sentences=raw_evidence_sentences,
                chunks=chunks,
            )
            graph_bundle: Dict[str, object] = self.m.build_evidence_graph_bundle(
                query,
                precomputed_context=(context_text, topics, chunks),
                evidence_sentences=raw_evidence_sentences,
                evidence_pack=self._last_evidence_pack,
                enable_filter=True,
                enable_claims=False,
                enable_light_graph=False,
            )
            stage_latency_sec["rag"] = time.perf_counter() - rag_started
            graph_stage_latency = dict(graph_bundle.get("stage_latency_sec", {}) or {})
            stage_latency_sec["filter"] = float(graph_stage_latency.get("filter", 0.0) or 0.0)
            stage_latency_sec["claims"] = 0.0
            stage_latency_sec["light_graph"] = 0.0
            stage_latency_sec["toolkit"] = 0.0
            evidence_sentences = self.m.final_answer_composer.bundle_to_evidence_sentences(
                graph_bundle,
                raw_fallback=raw_evidence_sentences,
            )
            if not evidence_sentences:
                evidence_sentences = list(raw_evidence_sentences[:8])
            self._last_specialist_payload = {}
        else:
            raw_evidence_sentences = self.m.answer_grounding.collect_evidence_sentences(
                query, chunks
            )
            self._last_evidence_pack = self._build_evidence_pack(
                query=query,
                evidence_sentences=raw_evidence_sentences,
                chunks=chunks,
            )
            graph_bundle: Dict[str, object] = {}
            try:
                graph_bundle = self.m.build_evidence_graph_bundle(
                    query,
                    precomputed_context=(context_text, topics, chunks),
                    evidence_sentences=raw_evidence_sentences,
                    evidence_pack=self._last_evidence_pack,
                    enable_filter=True,
                    enable_claims=True,
                    enable_light_graph=True,
                )
            except (RuntimeError, ValueError, TypeError):
                try:
                    graph_bundle = self.m.build_evidence_graph_bundle(
                        query,
                        precomputed_context=(context_text, topics, chunks),
                        evidence_sentences=raw_evidence_sentences,
                        evidence_pack=self._last_evidence_pack,
                        enable_filter=True,
                        enable_claims=False,
                        enable_light_graph=False,
                    )
                except (RuntimeError, ValueError, TypeError):
                    self.m.last_evidence_graph_bundle = {}
                    graph_bundle = {}
            stage_latency_sec["rag"] = time.perf_counter() - rag_started
            graph_stage_latency = dict(graph_bundle.get("stage_latency_sec", {}) or {})
            for key in ("filter", "claims", "light_graph"):
                stage_latency_sec[key] = float(graph_stage_latency.get(key, 0.0) or 0.0)
            evidence_sentences = self.m.final_answer_composer.bundle_to_evidence_sentences(
                graph_bundle,
                raw_fallback=raw_evidence_sentences,
            )
            if not evidence_sentences:
                evidence_sentences = list(raw_evidence_sentences[:8])
            self._last_specialist_payload = self.m.specialist_layer.run(
                query=query,
                graph_bundle=graph_bundle,
            )
            stage_latency_sec["toolkit"] = float(
                dict(self._last_specialist_payload or {}).get("latency_sec", 0.0) or 0.0
            )
        self.m.last_stage_latency_sec = dict(stage_latency_sec)
        return (
            context_text,
            topics,
            chunks,
            evidence_sentences,
        )

    def build_generation_prompt(
        self,
        *,
        input_text: str,
        retrieved_context_text: str,
        evidence_sentences: List[Dict[str, object]],
        chunks: List[Dict[str, object]],
    ) -> str:
        execution_mode = str(getattr(self.m, "retrieval_execution_mode", "memslm")).strip().lower()
        retrieved_context_text = str(retrieved_context_text or "").strip()
        bundle = dict(getattr(self.m, "last_evidence_graph_bundle", {}) or {})
        filtered_pack = dict(bundle.get("filtered_pack", {}) or {})
        claim_result = dict(bundle.get("claim_result", {}) or {})
        light_graph = dict(bundle.get("light_graph", {}) or {})
        toolkit_payload = dict(self._last_specialist_payload or {})

        if execution_mode == "model_only":
            prompt_sections: List[Dict[str, str]] = [
                {
                    "section": "answer_rules",
                    "text": "Return only the final answer.",
                }
            ]
            compact_prompt = "[Answer Rules]\nReturn only the final answer.\n\nUser: " + input_text
            self._last_compact_prompt_text = compact_prompt
            self._last_expanded_prompt_text = compact_prompt
            self._last_compact_prompt_support_sources = []
            self._last_expanded_prompt_support_sources = []
            self.m._set_prompt_trace_sections(prompt_sections)
            return compact_prompt

        if execution_mode == "naive_rag":
            retrieved_text = retrieved_context_text or "None"
            prompt_sections = [
                {"section": "retrieved_context", "text": retrieved_text},
                {
                    "section": "answer_rules",
                    "text": (
                        "Use only the retrieved context.\n"
                        "Do not add graph reasoning or fallback heuristics.\n"
                        "Return only the final answer."
                    ),
                },
            ]
            compact_prompt = (
                "[Retrieved Context]\n"
                f"{retrieved_text}\n\n"
                "[Answer Rules]\n"
                "Use only the retrieved context.\n"
                "Return only the final answer.\n\n"
                f"User: {input_text}"
            )
            self._last_compact_prompt_text = compact_prompt
            self._last_expanded_prompt_text = ""
            self._last_compact_prompt_support_sources = []
            self._last_expanded_prompt_support_sources = []
            self.m._set_prompt_trace_sections(prompt_sections)
            return compact_prompt
        if execution_mode == "filter_only":
            route_packet = {
                "mode": "evidence-heavy",
                "primary_source": "filtered_evidence",
                "prompt_schema": "source-direct",
                "compact_sections": ["filtered_evidence", "answer_rules"],
                "expanded_sections": ["filtered_evidence", "answer_rules"],
                "section_roles": {"filtered_evidence": "primary"},
                "expanded_section_roles": {"filtered_evidence": "primary"},
                "answer_type": str(filtered_pack.get("answer_type", "")),
                "abstention_policy": "standard",
            }
            rules = (
                "Use only the filtered evidence below. "
                "If it is insufficient, answer Not found in retrieved context. "
                "Return only the final answer."
            )
            compact_prompt, prompt_sections = self.m.final_answer_composer.build_prompt(
                input_text=input_text,
                filtered_pack=filtered_pack,
                claim_result={},
                light_graph={},
                toolkit_payload={},
                prompt_mode="compact",
                route_packet=route_packet,
                answer_rules_text=rules,
            )
            compact_support_sources = self.m.final_answer_composer.build_support_sources(
                filtered_pack=filtered_pack,
                claim_result={},
                light_graph={},
                toolkit_payload={},
                prompt_mode="compact",
                route_packet=route_packet,
            )
            self._last_compact_prompt_text = compact_prompt
            self._last_expanded_prompt_text = compact_prompt
            self._last_compact_prompt_support_sources = compact_support_sources
            self._last_expanded_prompt_support_sources = compact_support_sources
            self._last_compact_route_packet = dict(route_packet)
            self._last_expanded_route_packet = dict(route_packet)
            self.m._set_prompt_trace_sections(prompt_sections)
            return compact_prompt

        stage_started = time.perf_counter()
        router = getattr(self.m, "final_answer_router", None)
        route_packet = (
            router.route(
                query=input_text,
                filtered_pack=filtered_pack,
                claim_result=claim_result,
                light_graph=light_graph,
                toolkit_payload=toolkit_payload,
            )
            if router is not None
            else {}
        )
        compact_answer_rules_text = (
            router.build_answer_rules(route_packet, prompt_mode="compact")
            if router is not None and route_packet
            else None
        )
        expanded_answer_rules_text = (
            router.build_answer_rules(route_packet, prompt_mode="expanded")
            if router is not None and route_packet
            else None
        )
        compact_prompt, prompt_sections = self.m.final_answer_composer.build_prompt(
            input_text=input_text,
            filtered_pack=filtered_pack,
            claim_result=claim_result,
            light_graph=light_graph,
            toolkit_payload=toolkit_payload,
            prompt_mode="compact",
            route_packet=route_packet,
            answer_rules_text=compact_answer_rules_text,
        )
        compact_support_sources = self.m.final_answer_composer.build_support_sources(
            filtered_pack=filtered_pack,
            claim_result=claim_result,
            light_graph=light_graph,
            toolkit_payload=toolkit_payload,
            prompt_mode="compact",
            route_packet=route_packet,
        )
        self.m.last_stage_latency_sec["composer"] = time.perf_counter() - stage_started
        self._last_compact_prompt_text = compact_prompt
        self._last_compact_prompt_support_sources = compact_support_sources
        self._last_expanded_prompt_text = ""
        self._last_expanded_prompt_support_sources = []
        self._last_compact_route_packet = dict(route_packet or {})
        self._last_expanded_route_packet = dict(route_packet or {})
        self.m._set_prompt_trace_sections(prompt_sections)
        return compact_prompt

    def _build_expanded_generation_prompt(
        self,
        *,
        input_text: str,
        retrieved_context_text: str,
        evidence_sentences: List[Dict[str, object]],
        chunks: List[Dict[str, object]],
    ) -> Tuple[str, List[Dict[str, str]]]:
        if self._last_expanded_prompt_text and self._last_expanded_prompt_support_sources:
            return (
                self._last_expanded_prompt_text,
                list(self._last_expanded_prompt_support_sources),
            )
        execution_mode = str(getattr(self.m, "retrieval_execution_mode", "memslm")).strip().lower()
        if execution_mode in {"model_only", "naive_rag"}:
            prompt_text = "[Answer Rules]\nReturn only the final answer.\n\nUser: " + input_text
            self._last_expanded_prompt_text = prompt_text
            self._last_expanded_prompt_support_sources = []
            return prompt_text, []
        if execution_mode == "filter_only":
            if self._last_expanded_prompt_text:
                return (
                    self._last_expanded_prompt_text,
                    list(self._last_expanded_prompt_support_sources),
                )
            return self._last_compact_prompt_text, list(self._last_compact_prompt_support_sources)

        retrieved_context_text = str(retrieved_context_text or "").strip()
        _ = evidence_sentences, chunks
        bundle = dict(getattr(self.m, "last_evidence_graph_bundle", {}) or {})
        filtered_pack = dict(bundle.get("filtered_pack", {}) or {})
        claim_result = dict(bundle.get("claim_result", {}) or {})
        light_graph = dict(bundle.get("light_graph", {}) or {})
        toolkit_payload = dict(self._last_specialist_payload or {})
        route_packet = dict(self._last_expanded_route_packet or {})
        if not route_packet:
            router = getattr(self.m, "final_answer_router", None)
            if router is not None:
                route_packet = router.route(
                    query=input_text,
                    filtered_pack=filtered_pack,
                    claim_result=claim_result,
                    light_graph=light_graph,
                    toolkit_payload=toolkit_payload,
                )
            else:
                route_packet = {}
            self._last_expanded_route_packet = dict(route_packet)
        router = getattr(self.m, "final_answer_router", None)
        expanded_answer_rules_text = (
            router.build_answer_rules(route_packet, prompt_mode="expanded")
            if router is not None and route_packet
            else None
        )
        expanded_prompt, _ = self.m.final_answer_composer.build_prompt(
            input_text=input_text,
            filtered_pack=filtered_pack,
            claim_result=claim_result,
            light_graph=light_graph,
            toolkit_payload=toolkit_payload,
            prompt_mode="expanded",
            route_packet=route_packet,
            answer_rules_text=expanded_answer_rules_text,
        )
        expanded_support_sources = self.m.final_answer_composer.build_support_sources(
            filtered_pack=filtered_pack,
            claim_result=claim_result,
            light_graph=light_graph,
            toolkit_payload=toolkit_payload,
            prompt_mode="expanded",
            route_packet=route_packet,
        )
        self._last_expanded_prompt_text = expanded_prompt
        self._last_expanded_prompt_support_sources = list(expanded_support_sources)
        return expanded_prompt, list(expanded_support_sources)

    def generate_final_answer(
        self,
        *,
        input_text: str,
        query: str,
        prompt_text: str,
        evidence_sentences: List[Dict[str, object]],
    ) -> Tuple[str, str, str]:
        execution_mode = str(getattr(self.m, "retrieval_execution_mode", "memslm")).strip().lower()
        if execution_mode == "memslm":
            direct_answer, direct_path = self._maybe_get_toolkit_direct_answer(query)
            if direct_answer:
                self.m.last_stage_latency_sec["final_generation"] = 0.0
                return direct_answer, direct_path, ""
        prompt_messages: List[Dict[str, str]] = [{"role": "user", "content": prompt_text}]
        model_name = str(getattr(self.m.llm, "model_name", "unknown"))
        logger.info(
            "MemoryManager.chat: invoking main LLM "
            f"(model={model_name}, prompt_chars={len(prompt_text)})."
        )
        stage_started = time.perf_counter()
        response = self.m.llm.chat(prompt_messages)
        self.m.last_stage_latency_sec["final_generation"] = time.perf_counter() - stage_started
        if execution_mode in {"model_only", "naive_rag", "filter_only"}:
            ai_response = self.m.answer_grounding.normalize_final_answer(response, query)
            if not ai_response.strip():
                ai_response = "Not found in retrieved context."
            return ai_response, f"{execution_mode}_direct", ""
        if not bool(getattr(self.m, "final_answer_guard_enabled", False)):
            ai_response = self.m.answer_grounding.normalize_final_answer(
                response,
                query,
            )
            if not ai_response.strip():
                ai_response = "Not found in retrieved context."
            return ai_response, "single_pass_direct", ""
        fallback_result = self.m.answer_grounding.evaluate_response_guard(
            response=response,
            evidence_sentences=evidence_sentences,
            candidates=[],
            evidence_candidate=None,
            fallback_answer=None,
            support_sources=self._last_compact_prompt_support_sources,
        )
        ai_response = str(fallback_result.get("response", "")).strip()
        fallback_path = str(fallback_result.get("fallback_path", "none"))
        not_found_reason = str(fallback_result.get("not_found_reason", ""))

        normalized_ai_response = ai_response.strip().lower()
        should_retry_second_pass = bool(
            getattr(self.m, "final_answer_second_pass_enabled", False)
        ) and (
            fallback_path.startswith("retry_due_to_")
            or (
                self.m.answer_grounding.second_pass_llm_enabled
                and evidence_sentences
                and normalized_ai_response == "not found in retrieved context."
                and fallback_path in {"fallback_to_not_found", "llm_not_found_accepted"}
            )
        )

        if should_retry_second_pass:
            model_name = str(getattr(self.m.llm, "model_name", "unknown"))
            logger.info("MemoryManager.chat: invoking second-pass LLM " f"(model={model_name}).")
            expanded_source_prompt, expanded_support_sources = (
                self._build_expanded_generation_prompt(
                    input_text=input_text,
                    retrieved_context_text="",
                    evidence_sentences=evidence_sentences,
                    chunks=[],
                )
            )
            if expanded_source_prompt:
                self._last_expanded_prompt_text = expanded_source_prompt
            if expanded_support_sources:
                self._last_expanded_prompt_support_sources = expanded_support_sources
            second_prompt = self.m.answer_grounding.build_second_pass_retry_prompt(
                prompt_text=expanded_source_prompt,
                first_answer=response,
            )
            second_response = self.m.llm.chat([{"role": "user", "content": second_prompt}])
            second_result = self.m.answer_grounding.evaluate_response_guard(
                response=second_response,
                evidence_sentences=evidence_sentences,
                candidates=[],
                evidence_candidate=None,
                fallback_answer=None,
                support_sources=self._last_expanded_prompt_support_sources,
            )
            second_path = str(second_result.get("fallback_path", "none"))
            ai_response = str(second_result.get("response", "")).strip()
            if second_path.startswith("retry_due_to_"):
                ai_response = "Not found in retrieved context."
                second_path = "fallback_to_not_found"
            fallback_path = "second_pass:" + second_path
            not_found_reason = str(second_result.get("not_found_reason", not_found_reason))

        ai_response = self.m.answer_grounding.normalize_final_answer(
            ai_response,
            query,
        )
        return ai_response, fallback_path, not_found_reason
