"""Chat orchestration runtime extracted from MemoryManager."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from llm_long_memory.utils.logger import logger


class MemoryManagerChatRuntime:
    """Keep MemoryManager.chat-related orchestration out of the main class body."""

    def __init__(self, manager: Any) -> None:
        self.m = manager
        self._last_specialist_payload: Dict[str, object] = {}
        self._last_evidence_pack: Dict[str, object] = {}

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

    def _plan_slots(self, query_plan: Dict[str, object]) -> set[str]:
        slots: set[str] = set()
        for key in ("compare_options", "state_keys", "entities", "focus_phrases", "sub_queries"):
            for item in list(query_plan.get(key, [])):
                value = str(item).strip().lower()
                if value:
                    slots.update(self._tokenize(value))
        return slots

    def _fallback_confident(
        self,
        *,
        text: str,
        query_plan: Dict[str, object],
        evidence_candidate: Optional[Dict[str, str]],
    ) -> bool:
        norm = self._normalize_space(text)
        if not norm:
            return False
        tokens = self._tokenize(norm)
        if not tokens or len(tokens) > 14:
            return False
        low = norm.lower()
        if low.startswith("(user)") or low.startswith("(assistant)") or low.startswith("(system)"):
            return False
        answer_type = str(query_plan.get("answer_type", "")).strip().lower()
        if answer_type == "count" and re.fullmatch(r"\d+", norm):
            return False
        slots = self._plan_slots(query_plan)
        if slots:
            overlap = len(slots.intersection(set(tokens)))
            if overlap <= 0:
                score = 0.0
                if evidence_candidate is not None:
                    try:
                        score = float(str(evidence_candidate.get("score", "0") or "0"))
                    except ValueError:
                        score = 0.0
                return score >= 0.55
        return True

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
        generic_stopwords = {
            "who",
            "what",
            "which",
            "where",
            "when",
            "why",
            "how",
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
            "i",
            "we",
            "you",
            "they",
            "he",
            "she",
            "it",
            "a",
            "an",
            "the",
            "and",
            "or",
            "to",
            "of",
            "in",
            "on",
            "for",
            "with",
            "my",
            "our",
            "your",
            "their",
            "me",
            "us",
            "them",
        }

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
                        score += 0.5 * (len(query_tokens.intersection(stoks)) / float(len(query_tokens)))
                    if stoks.intersection(temporal_keywords):
                        score += 0.6
                    if sig_terms and stoks.intersection(sig_terms):
                        score += 0.8
                    # prevent location/profile-only noise from dominating temporal compare
                    if any(x in stoks for x in {"live", "hometown", "town", "city", "culture", "history"}):
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

            def _count_fact_score(text: str) -> float:
                low = text.lower()
                score = self._score_overlap(query, text)
                if target and _contains_phrase(low, target):
                    score += 0.8
                if self._contains_digit_or_number_word(text):
                    score += 0.7
                first_person_fact = bool(
                    re.search(
                        r"\b(i|my|we|our)\b.{0,40}\b(have|had|own|owned|bought|got|kept|led|managed|tried|completed|serviced|plan(?:ned)?|attended)\b",
                        low,
                    )
                )
                if first_person_fact:
                    score += 0.4
                if count_unit and count_unit in low:
                    score += 0.2
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
                    if not _contains_phrase(low, target):
                        continue
                    score = _count_fact_score(text)
                    if score < 0.55:
                        continue
                    key = low
                    if key in seen_count:
                        continue
                    seen_count.add(key)
                    ranked.append((score, text))
                ranked.sort(key=lambda x: x[0], reverse=True)
                picked = [text for _, text in ranked[:6]]
                if not picked:
                    insufficient = True
                    picked = _pick(limit=4)
            else:
                picked = _pick(limit=5)
            lines.extend([f"- {x}" for x in picked[:6]])
        elif answer_type == "temporal_comparison":
            options = [str(x).strip() for x in list(plan.get("compare_options", [])) if str(x).strip()]
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
                lines.append("- compare_rule: prioritize evidence matching query action/time cues and choose the earlier timeline.")
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
            state_keys = [str(x).strip() for x in list(plan.get("state_keys", [])) if str(x).strip()]
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
                anchors = [str(x).strip() for x in list(plan.get("entities", [])) if str(x).strip()][:2]

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

            update_cues = {
                "now",
                "currently",
                "recently",
                "latest",
                "moved",
                "switched",
                "changed",
                "updated",
                "set",
            }
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
            picked = _pick(limit=6, must=["prefer", "favorite", "recommend", "suggest", "like", "enjoy"])
            if not picked:
                picked = _pick(limit=5)
            lines.extend([f"- {x}" for x in picked[:6]])
        else:
            lines.extend([f"- {x}" for x in _pick(limit=6)])

        if not lines:
            insufficient = True
            lines = [f"- {self._normalize_space(str(x.get('text', '')))}" for x in top[:4] if str(x.get("text", "")).strip()]

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

    def prepare_answer_inputs(
        self,
        query: str,
        precomputed_context: Optional[
            Tuple[str, List[Dict[str, object]], List[Dict[str, object]]]
        ],
    ) -> Tuple[
        str,
        List[Dict[str, object]],
        List[Dict[str, object]],
        List[Dict[str, object]],
        str,
        Optional[Dict[str, str]],
        str,
        str,
    ]:
        if precomputed_context is not None:
            context_text, topics, chunks = precomputed_context
        else:
            context_text, topics, chunks = self.m.retrieve_context(query)
        logger.info(f"MemoryManager.chat: retrieved chunks={len(chunks)}")

        if self.m.retrieval_execution_mode in {"model_only", "naive_rag"}:
            evidence_sentences: List[Dict[str, object]] = []
            candidates: List[Dict[str, object]] = []
            fallback_answer = ""
            evidence_candidate = None
            best_evidence = ""
            best_candidate = ""
            self._last_specialist_payload = {}
        else:
            evidence_sentences = self.m.answering.collect_evidence_sentences(query, chunks)
            self._last_evidence_pack = self._build_evidence_pack(
                query=query,
                evidence_sentences=evidence_sentences,
                chunks=chunks,
            )
            candidates = self.m.answering.extract_candidates(query, evidence_sentences)
            self.m.answering.log_decision_snapshot(query, evidence_sentences, candidates)
            fallback_answer = ""
            evidence_candidate = self.m.answering.extract_evidence_candidate(
                query, evidence_sentences, candidates
            )
            if bool(getattr(self.m.answering, "reasoning_fallback_enabled", True)):
                # Base fallback comes from candidate extractor path only.
                fallback_answer = (
                    str((evidence_candidate or {}).get("answer", "")).strip()
                    if evidence_candidate is not None
                    else ""
                )
            best_evidence = (
                str(evidence_sentences[0].get("text", ""))[:160] if evidence_sentences else ""
            )
            best_candidate = str(candidates[0].get("text", "")) if candidates else ""
            graph_context = self.build_graph_context(query=query, chunks=chunks)
            self._last_specialist_payload = self.m.specialist_layer.run(
                query=query,
                graph_context=graph_context,
                evidence_sentences=evidence_sentences,
                candidates=candidates,
                chunks=chunks,
            )
            specialist_fallback = str(
                self._last_specialist_payload.get("fallback_answer", "")
            ).strip()
            if specialist_fallback:
                fallback_answer = specialist_fallback
        return (
            context_text,
            topics,
            chunks,
            evidence_sentences,
            candidates,
            fallback_answer,
            evidence_candidate,
            best_evidence,
            best_candidate,
        )

    def resolve_prompt_fallback(
        self,
        fallback_answer: str,
        evidence_candidate: Optional[Dict[str, str]],
        candidates: List[Dict[str, object]],
        best_evidence: str,
        query_plan: Optional[Dict[str, object]] = None,
    ) -> str:
        plan = dict(query_plan or {})
        answer_type = str(plan.get("answer_type", "")).strip().lower()
        if answer_type == "temporal_comparison":
            return ""
        fallback_candidates: List[str] = []
        if str(fallback_answer or "").strip():
            fallback_candidates.append(str(fallback_answer).strip())
        if evidence_candidate is not None:
            ans = str(evidence_candidate.get("answer", "")).strip()
            if ans:
                fallback_candidates.append(ans)
        if candidates:
            cand_text = str(candidates[0].get("text", "")).strip()
            if cand_text:
                fallback_candidates.append(cand_text)
        # Never use full raw evidence sentence as generic fallback; only allow extractive modes.
        if answer_type in {"extractive", "span"} and best_evidence.strip():
            fallback_candidates.append(best_evidence.strip())

        for cand in fallback_candidates:
            if self._fallback_confident(
                text=cand,
                query_plan=plan,
                evidence_candidate=evidence_candidate,
            ):
                return cand
        return ""

    def build_generation_prompt(
        self,
        *,
        input_text: str,
        retrieved_context_text: str,
        evidence_sentences: List[Dict[str, object]],
        chunks: List[Dict[str, object]],
        candidates: List[Dict[str, object]],
        best_evidence: str,
        fallback_answer: str,
        evidence_candidate: Optional[Dict[str, str]],
    ) -> str:
        execution_mode = str(getattr(self.m, "retrieval_execution_mode", "memslm")).strip().lower()
        graph_context = self.build_graph_context(query=input_text, chunks=chunks)
        retrieved_context_text = str(retrieved_context_text or "").strip()
        evidence_pack_text = self._format_evidence_pack(self._last_evidence_pack)
        query_plan = dict(getattr(self.m, "last_query_plan", {}) or {})

        if execution_mode == "model_only":
            prompt_sections: List[Dict[str, str]] = [
                {
                    "section": "answer_rules",
                    "text": "Return only the final answer.",
                }
            ]
            compact_prompt = self.m.answering.build_answer_prompt(
                input_text=input_text,
                graph_context="",
                query_plan="",
                graph_tool_hints="",
                rag_evidence="",
                fallback_answer="",
            )
            self.m._set_prompt_eval_chunks(prompt_sections)
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
            self.m._set_prompt_eval_chunks(prompt_sections)
            return compact_prompt

        graph_tool_hints = self.build_graph_tool_hints(
            query=input_text,
            graph_context=graph_context,
            evidence_sentences=evidence_sentences,
            candidates=candidates,
            chunks=chunks,
        )

        fallback_text = self.resolve_prompt_fallback(
            fallback_answer=fallback_answer,
            evidence_candidate=evidence_candidate,
            candidates=candidates,
            best_evidence=best_evidence,
            query_plan=query_plan,
        )
        prompt_sections: List[Dict[str, str]] = []
        query_plan_text = ""
        if query_plan:
            query_plan_text = json.dumps(
                {
                    "intent": query_plan.get("intent", ""),
                    "answer_type": query_plan.get("answer_type", ""),
                    "focus_phrases": list(query_plan.get("focus_phrases", []))[:6],
                    "sub_queries": list(query_plan.get("sub_queries", []))[:6],
                },
                ensure_ascii=False,
            )
            prompt_sections.append(
                {
                    "section": "query_plan",
                    "text": query_plan_text,
                }
            )
        if graph_context.strip():
            prompt_sections.append({"section": "graph_evidence", "text": graph_context.strip()})
        if graph_tool_hints.strip():
            prompt_sections.append({"section": "graph_tool_hints", "text": graph_tool_hints.strip()})
        rag_evidence_text = evidence_pack_text or best_evidence.strip()
        if rag_evidence_text:
            prompt_sections.append({"section": "rag_evidence", "text": rag_evidence_text})
        if fallback_text:
            prompt_sections.append({"section": "fallback_answer", "text": fallback_text})
        prompt_sections.append(
            {
                "section": "answer_rules",
                "text": (
                    "Use Graph Evidence first.\n"
                    "If Graph Evidence is weak or empty, use the Fallback Answer as the compact backup clue.\n"
                    "Do not repeat long evidence blocks.\n"
                    "Do not say Not found unless both Graph Evidence and the fallback cues are insufficient.\n"
                    "Keep key qualifiers (for example: each way, round trip, per day).\n"
                    "Return only the final answer."
                    if self.m.answering.answer_context_only
                    else "Return only the final answer."
                ),
            }
        )
        compact_prompt = self.m.answering.build_answer_prompt(
            input_text=input_text,
            graph_context=graph_context,
            query_plan=query_plan_text,
            graph_tool_hints=graph_tool_hints,
            rag_evidence=rag_evidence_text,
            fallback_answer=fallback_text,
        )
        self.m._set_prompt_eval_chunks(prompt_sections)
        return compact_prompt

    def build_graph_tool_hints(
        self,
        *,
        query: str,
        graph_context: str,
        evidence_sentences: List[Dict[str, object]],
        candidates: List[Dict[str, object]],
        chunks: List[Dict[str, object]],
    ) -> str:
        # Kept method name for backward compatibility with prompt-builder call sites.
        # In the current architecture, specialist hints come from the unified specialist layer.
        hints = str(self._last_specialist_payload.get("hints", "")).strip()
        if hints:
            return hints
        payload = self.m.specialist_layer.run(
            query=query,
            graph_context=graph_context,
            evidence_sentences=evidence_sentences,
            candidates=candidates,
            chunks=chunks,
        )
        self._last_specialist_payload = payload
        return str(payload.get("hints", "")).strip()

    def build_graph_context(self, query: str, chunks: List[Dict[str, object]]) -> str:
        if not self.m.graph_refiner_enabled:
            return ""
        if not bool(getattr(self.m, "long_memory_enabled", False)):
            return ""
        if not bool(getattr(self.m, "offline_graph_build_enabled", False)):
            return ""
        # Enforce offline-first long-memory usage:
        # graph extraction is completed before answering; chat only reads stored graph evidence.
        if not self.m.graph_context_from_store_enabled:
            return ""
        snippets: List[str] = []
        engine = getattr(self.m, "graph_query_engine", None)
        if engine is not None:
            try:
                pack = engine.query(
                    query=query,
                    max_items=min(4, max(1, int(getattr(self.m.long_memory, "context_max_items", 4)))),
                )
                raw_snippets = list(pack.get("snippets", [])) if isinstance(pack, dict) else []
                snippets = [str(x).strip() for x in raw_snippets if str(x).strip()]
            except (RuntimeError, ValueError, TypeError):
                snippets = []
        # Safe fallback to legacy snippet generation if graph query yielded nothing.
        if not snippets:
            snippets = self.m.long_memory.build_context_snippets(query)
        if not snippets:
            return ""
        return "[Long Memory Graph]\n" + "\n".join(f"- {line}" for line in snippets[:4])

    def generate_with_fallback(
        self,
        *,
        input_text: str,
        query: str,
        prompt_text: str,
        evidence_sentences: List[Dict[str, object]],
        candidates: List[Dict[str, object]],
        fallback_answer: str,
        evidence_candidate: Optional[Dict[str, str]],
    ) -> Tuple[str, str, str]:
        prompt_messages: List[Dict[str, str]] = [{"role": "user", "content": prompt_text}]
        model_name = str(getattr(self.m.llm, "model_name", "unknown"))
        logger.info(
            "MemoryManager.chat: invoking main LLM "
            f"(model={model_name}, prompt_chars={len(prompt_text)})."
        )
        response = self.m.llm.chat(prompt_messages)
        execution_mode = str(getattr(self.m, "retrieval_execution_mode", "memslm")).strip().lower()
        if execution_mode in {"model_only", "naive_rag"}:
            ai_response = self.m.answering.postprocess_final_answer(
                response, query, evidence_candidate=None
            )
            return ai_response, f"{execution_mode}_direct", ""
        fallback_result = self.m.answering.evaluate_response_fallback(
            response=response,
            evidence_sentences=evidence_sentences,
            candidates=candidates,
            evidence_candidate=evidence_candidate,
            fallback_answer=fallback_answer,
        )
        ai_response = str(fallback_result.get("response", "")).strip()
        fallback_path = str(fallback_result.get("fallback_path", "none"))
        not_found_reason = str(fallback_result.get("not_found_reason", ""))

        normalized_ai_response = ai_response.strip().lower()
        should_retry_second_pass = (
            fallback_path.startswith("retry_due_to_")
            or (
                self.m.answering.second_pass_llm_enabled
                and evidence_sentences
                and normalized_ai_response == "not found in retrieved context."
                and fallback_path in {"fallback_to_not_found", "llm_not_found_accepted"}
            )
        )

        if should_retry_second_pass:
            model_name = str(getattr(self.m.llm, "model_name", "unknown"))
            logger.info(
                "MemoryManager.chat: invoking second-pass LLM "
                f"(model={model_name})."
            )
            second_prompt = self.m.answering.build_second_pass_prompt(
                prompt_text=prompt_text,
                evidence_candidate=evidence_candidate,
            )
            second_response = self.m.llm.chat([{"role": "user", "content": second_prompt}])
            second_result = self.m.answering.evaluate_response_fallback(
                response=second_response,
                evidence_sentences=evidence_sentences,
                candidates=candidates,
                evidence_candidate=evidence_candidate,
                fallback_answer=fallback_answer,
            )
            second_path = str(second_result.get("fallback_path", "none"))
            ai_response = str(second_result.get("response", "")).strip()
            if second_path.startswith("retry_due_to_"):
                ai_response = "Not found in retrieved context."
                second_path = "fallback_to_not_found"
            fallback_path = "second_pass:" + second_path
            not_found_reason = str(second_result.get("not_found_reason", not_found_reason))

        ai_response = self.m.answering.postprocess_final_answer(
            ai_response, query, evidence_candidate=evidence_candidate
        )
        return ai_response, fallback_path, not_found_reason
