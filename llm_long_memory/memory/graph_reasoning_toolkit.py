"""Lightweight graph-tool hint builder for MemSLM answering.

This module does not introduce a new LLM call. Instead, it converts the
retrieved graph/context evidence into compact tool-style hints that the final
model can consume before producing the answer.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from llm_long_memory.memory.answering_temporal import parse_choice_query
from llm_long_memory.memory.counting_resolver import CountingResolver


class GraphReasoningToolkit:
    """Build compact graph-tool hints for count / temporal / preference questions."""

    def __init__(self, manager: Any) -> None:
        self.m = manager
        self.counting: CountingResolver = manager.answering.counting
        self.count_cues = {
            str(x).strip().lower()
            for x in list(getattr(self.counting, "count_query_keywords", []))
            if str(x).strip()
        }
        self.day_count_cues = {
            str(x).strip().lower()
            for x in list(getattr(self.counting, "day_count_keywords", []))
            if str(x).strip()
        }
        self.list_count_cues = {
            str(x).strip().lower()
            for x in list(getattr(self.counting, "list_count_query_keywords", []))
            if str(x).strip()
        }
        self.preference_cues = {
            "prefer",
            "preference",
            "recommend",
            "suggest",
            "resources",
            "resource",
            "advice",
            "tips",
            "help",
            "help me",
            "interested in",
            "would prefer",
            "what should i choose",
            "would like",
            "like to",
            "how should i",
            "what should i",
        }
        self.temporal_cues = {
            "before",
            "after",
            "earlier",
            "later",
            "second",
            "third",
            "which came first",
            "when",
            "between",
            "compare",
            "compared",
            "earliest",
            "latest",
            "more recent",
        }

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(str(text).split()).strip()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", str(text).lower())

    @staticmethod
    def _split_phrases(text: str) -> List[str]:
        raw = str(text or "").strip()
        if not raw:
            return []
        parts = re.split(r"(?:,|;|/|\band\b|\bor\b|\bplus\b|\+)", raw, flags=re.IGNORECASE)
        out: List[str] = []
        for part in parts:
            cleaned = " ".join(str(part).split()).strip(" -:;,.")
            if cleaned:
                out.append(cleaned)
        return out

    @staticmethod
    def _parse_graph_context(graph_context: str) -> List[str]:
        text = str(graph_context or "").strip()
        if not text:
            return []
        lines: List[str] = []
        for raw in re.split(r"\n+", text):
            line = re.sub(r"^\s*[-*•]+\s*", "", str(raw)).strip()
            line = re.sub(r"^\s*\[\d+\]\s*", "", line).strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split("||") if part.strip()]
            for part in parts:
                cleaned = " ".join(part.split()).strip()
                if cleaned:
                    lines.append(cleaned)
        uniq: List[str] = []
        seen: set[str] = set()
        for line in lines:
            key = line.lower()
            if not key or key in seen:
                continue
            seen.add(key)
            uniq.append(line)
        return uniq

    @staticmethod
    def _parse_evidence_lines(items: Sequence[Dict[str, object]], limit: int = 4) -> List[str]:
        ranked = sorted(items, key=lambda x: float(x.get("score", 0.0)), reverse=True)
        lines: List[str] = []
        for item in ranked[: max(0, limit)]:
            text = " ".join(str(item.get("text", "")).split()).strip()
            if not text:
                continue
            lines.append(text[:220])
        return lines

    def _query_focus_phrase(self, query: str) -> str:
        q = " ".join(str(query or "").split()).strip()
        if not q:
            return ""
        def _clean_focus(text: str) -> str:
            cleaned = " ".join(str(text).split()).strip(" ,.;:!?\"'")
            cleaned = re.sub(r"^(?:my|our|your|their|the)\s+", "", cleaned, flags=re.IGNORECASE)
            return cleaned
        patterns = [
            r"\blearn more about\s+(.+?)(?:[?.!]|$)",
            r"\bways to\s+(.+?)(?:[?.!]|$)",
            r"\brecommend(?:\s+some)?\s+(.+?)(?:[?.!]|$)",
            r"\brecommend(?:\s+some)?\s+resources\s+for\s+(.+?)(?:[?.!]|$)",
            r"\bwhat advice do you have for\s+(.+?)(?:[?.!]|$)",
            r"\bwhat advice would you give for\s+(.+?)(?:[?.!]|$)",
            r"\bwhat advice do you have about\s+(.+?)(?:[?.!]|$)",
            r"\bwhat should i\s+(.+?)(?:[?.!]|$)",
            r"\bhow should i\s+(.+?)(?:[?.!]|$)",
            r"\bthinking about ways to\s+(.+?)(?:[?.!]|$)",
        ]
        lowered = q.lower()
        for pat in patterns:
            m = re.search(pat, lowered, flags=re.IGNORECASE)
            if m:
                focus = _clean_focus(str(m.group(1)))
                if focus:
                    return focus
        return ""

    def _extract_resource_phrases(self, lines: Sequence[str]) -> List[str]:
        phrases: List[str] = []
        seen: set[str] = set()
        stop = {
            "are",
            "is",
            "the",
            "this",
            "that",
            "those",
            "these",
            "hey",
            "long",
            "memory",
            "graph",
            "example",
            "script",
            "want",
            "looking",
            "resource",
            "resources",
            "im",
            "i",
            "me",
            "my",
            "you",
            "we",
            "our",
            "your",
            "there",
            "any",
            "concerns",
            "thoughts",
        }
        for line in lines:
            text = self._normalize(line)
            if not text:
                continue
            candidates = []
            candidates.extend(re.findall(r"\b(?:[A-Z][A-Za-z0-9.&'-]*)(?:\s+(?:[A-Z][A-Za-z0-9.&'-]*|[0-9]+))+", text))
            candidates.extend(re.findall(r"\b[A-Z][A-Za-z0-9.&'-]{2,}(?:\s+[A-Z][A-Za-z0-9.&'-]{2,})*\b", text))
            for cand in candidates:
                cleaned = " ".join(str(cand).split()).strip(" ,.;:!?\"'")
                if not cleaned:
                    continue
                if re.search(r"\b(graph|memory|long memory|tool hints?)\b", cleaned.lower()):
                    continue
                tokens = [tok for tok in self._tokenize(cleaned) if tok not in stop]
                if len(tokens) < 2 and not re.search(r"[A-Z]{2,}|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", cleaned):
                    continue
                if all(tok in stop or len(tok) <= 2 for tok in tokens):
                    continue
                key = cleaned.lower()
                if key in seen:
                    continue
                seen.add(key)
                phrases.append(cleaned)
        return phrases

    @staticmethod
    def _extract_spelled_number(text: str) -> int:
        low = str(text or "").lower()
        mapping = {
            "zero": 0,
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "eleven": 11,
            "twelve": 12,
        }
        for word, num in mapping.items():
            if re.search(rf"\b{word}\b", low):
                return num
        if re.search(r"\ba pair of\b", low):
            return 2
        if re.search(r"\ba couple of\b", low):
            return 2
        return -1

    def _extract_count_items(self, query: str, lines: Sequence[str]) -> List[str]:
        focus_tokens = set(self._tokenize(query))
        items: List[str] = []
        seen: set[str] = set()
        for line in lines:
            text = self._normalize(line)
            if not text:
                continue
            text_tokens = set(self._tokenize(text))
            list_entities = self.counting._extract_list_entities(text)
            if focus_tokens and len(focus_tokens.intersection(text_tokens)) == 0:
                if not (self._line_has_list_shape(text) and len(list_entities) >= 2):
                    continue
            for ent in list_entities:
                cleaned = self._normalize(ent)
                if not cleaned:
                    continue
                key = cleaned.lower()
                if key in seen:
                    continue
                seen.add(key)
                items.append(cleaned)
            if len(items) >= 6:
                break
        return items

    @staticmethod
    def _line_has_list_shape(text: str) -> bool:
        low = " ".join(str(text or "").split()).lower()
        return (
            "," in low
            or " and " in low
            or " or " in low
            or " plus " in low
            or "+" in low
        )

    def _extract_temporal_dates(self, lines: Sequence[str]) -> List[str]:
        dates: List[str] = []
        seen: set[str] = set()
        for line in lines:
            text = self._normalize(line)
            if not text:
                continue
            parsed = self.counting._extract_dates_with_context(text=text, session_date="")
            for dt in parsed:
                key = dt.strftime("%Y-%m-%d")
                if key in seen:
                    continue
                seen.add(key)
                dates.append(key)
        return dates

    def _build_temporal_duration_hint(self, query: str, lines: Sequence[str]) -> str:
        delta_days = self._temporal_duration_days(lines)
        if delta_days is None:
            return ""
        return self._format_duration_from_days(query, delta_days)

    @staticmethod
    def _format_duration_from_days(query: str, delta_days: int) -> str:
        q = str(query or "").lower()
        if delta_days <= 0:
            return ""
        if "week" in q:
            weeks = max(1, round(delta_days / 7.0))
            return f"{weeks} week{'s' if weeks != 1 else ''}"
        if "month" in q:
            months = max(1, round(delta_days / 30.0))
            return f"{months} month{'s' if months != 1 else ''}"
        if "year" in q:
            years = max(1, round(delta_days / 365.0))
            return f"{years} year{'s' if years != 1 else ''}"
        return f"{delta_days} days"

    def _temporal_duration_days(self, lines: Sequence[str]) -> Optional[int]:
        dates = self._extract_temporal_dates(lines)
        if len(dates) < 2:
            return None
        from datetime import datetime

        parsed_dates = []
        for d in dates:
            try:
                parsed_dates.append(datetime.strptime(d, "%Y-%m-%d"))
            except ValueError:
                continue
        if len(parsed_dates) < 2:
            return None
        parsed_dates.sort()
        delta_days = abs((parsed_dates[-1] - parsed_dates[0]).days)
        if delta_days <= 0:
            return None
        return delta_days

    def _extract_temporal_anchor_dates(self, query: str, lines: Sequence[str]) -> List[str]:
        q = str(query or "").lower()
        if not lines:
            return []
        anchor_terms = {
            "accept": {"accept", "accepted", "admit", "admitted", "approved", "exchange program", "application"},
            "start": {"start", "started", "attend", "attending", "orientation", "session"},
        }
        selected: List[str] = []
        for line in lines:
            text = self._normalize(line)
            if not text:
                continue
            low = text.lower()
            dates = self._extract_temporal_dates([text])
            if not dates:
                continue
            if any(term in low for term in anchor_terms["accept"]):
                selected.extend(dates[-1:])
                continue
            if any(term in low for term in anchor_terms["start"]):
                selected.extend(dates[:1])
                continue
        if len(selected) >= 2:
            uniq: List[str] = []
            seen: set[str] = set()
            for key in selected:
                if key in seen:
                    continue
                seen.add(key)
                uniq.append(key)
            return sorted(uniq)
        # Fallback to query-guided phrases when the coarse buckets are insufficient.
        if "when" in q and ("accepted" in q or "accept" in q):
            uniq: List[str] = []
            seen: set[str] = set()
            for key in selected:
                if key in seen:
                    continue
                seen.add(key)
                uniq.append(key)
            return uniq
        return []

    def _classify_intent(self, query: str) -> str:
        q = str(query or "").strip().lower()
        if not q:
            return "generic"
        choice_options = parse_choice_query(
            query,
            max_options=4,
            default_target_k=2,
        )
        if re.search(r"\bhow many\s+(?:days?|weeks?|months?|years?)\b", q) or re.search(
            r"\bhow long\b", q
        ):
            return "temporal_count"
        if any(cue in q for cue in self.count_cues) or any(
            phrase in q for phrase in ("how many", "how much", "count of", "number of", "total")
        ):
            return "count"
        if any(cue in q for cue in self.day_count_cues) or any(
            phrase in q for phrase in ("days between", "weeks between", "how long")
        ):
            return "temporal_count"
        if any(cue in q for cue in self.preference_cues):
            return "preference"
        if choice_options and len(choice_options) >= 2:
            return "temporal_compare"
        if "which came first" in q or "which came earlier" in q or "which came later" in q:
            return "temporal_compare"
        return "generic"

    def _select_best_lines(
        self,
        *,
        query: str,
        lines: Sequence[str],
        max_items: int,
        prefer_time: bool = False,
        prefer_number: bool = False,
        prefer_preference: bool = False,
    ) -> List[str]:
        q_tokens = set(self._tokenize(query))
        ranked: List[tuple[float, int, str]] = []
        for idx, line in enumerate(lines):
            l = self._normalize(line)
            if not l:
                continue
            l_tokens = set(self._tokenize(l))
            if not l_tokens:
                continue
            overlap = len(q_tokens.intersection(l_tokens)) / float(max(1, len(q_tokens)))
            score = overlap
            low = l.lower()
            if prefer_time and re.search(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", low):
                score += 0.45
            if prefer_time and re.search(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b", low):
                score += 0.30
            if prefer_number and re.search(r"\b\d+(?:\.\d+)?\b", low):
                score += 0.35
            if prefer_preference and any(cue in low for cue in self.preference_cues):
                score += 0.30
            if any(tok in low for tok in ("state_fact", "value=", "location=", "time=")):
                score += 0.10
            ranked.append((score, idx, l))
        ranked.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        chosen: List[str] = []
        seen: set[str] = set()
        for _score, _idx, line in ranked:
            key = line.lower()
            if key in seen:
                continue
            seen.add(key)
            chosen.append(line)
            if len(chosen) >= max(1, max_items):
                break
        return chosen

    def _build_preference_summary(
        self,
        *,
        query: str,
        focus: str,
        resources: Sequence[str],
        selected: Sequence[str],
    ) -> str:
        parts: List[str] = []
        if focus:
            parts.append(f"Prefer {focus}")
        if resources:
            parts.append("resource suggestions: " + ", ".join(resources[:2]))
        if selected:
            reason = self._normalize(selected[0])
            if reason:
                parts.append(f"reason: {reason}")
        if parts:
            return "; ".join(parts)
        return ""

    def _derive_count_answer_from_lines(
        self,
        *,
        query: str,
        lines: Sequence[str],
    ) -> tuple[str, List[str]]:
        selected = self._select_best_lines(
            query=query,
            lines=lines,
            max_items=5,
            prefer_number=False,
        )
        count_items = self._extract_count_items(query, selected)
        if len(count_items) >= 2:
            return str(len(count_items)), count_items
        for line in selected:
            if not self._line_has_list_shape(line):
                continue
            spelled = self._extract_spelled_number(line)
            if spelled >= 0:
                return str(spelled), count_items
            m = re.search(r"\b(\d+)\b", line)
            if m:
                return str(int(m.group(1))), count_items
        return "", count_items

    def build_tool_hints(
        self,
        *,
        query: str,
        graph_context: str,
        evidence_sentences: Sequence[Dict[str, object]],
        candidates: Sequence[Dict[str, object]],
        chunks: Sequence[Dict[str, object]],
    ) -> str:
        """Return compact graph-tool hints to feed the final LLM prompt."""
        intent = self._classify_intent(query)
        if intent == "generic":
            return ""

        graph_lines = self._parse_graph_context(graph_context)
        evidence_lines = self._parse_evidence_lines(evidence_sentences, limit=4)
        candidate_lines = self._parse_evidence_lines(candidates, limit=3)
        all_lines = graph_lines + evidence_lines + candidate_lines
        evidence_pref_lines = evidence_lines + candidate_lines

        tool_lines: List[str] = [f"intent={intent}"]

        if intent == "count":
            count_answer_lines = evidence_lines + candidate_lines if (evidence_lines or candidate_lines) else all_lines
            count_answer, count_items = self._derive_count_answer_from_lines(
                query=query,
                lines=count_answer_lines,
            )
            if count_items:
                tool_lines.append("count_items=" + " | ".join(count_items[:6]))
                if len(count_items) >= 2:
                    tool_lines.append(f"count_hint={len(count_items)} items")
            if count_answer:
                tool_lines.append(f"count_answer={count_answer}")
            selected = self._select_best_lines(
                query=query,
                lines=count_answer_lines,
                max_items=5,
                prefer_number=False,
                prefer_time=(intent == "temporal_count"),
            )
            if selected:
                tool_lines.append("support=" + " || ".join(selected[:3]))
            elif count_answer:
                tool_lines.append(f"count_hint={count_answer}")
            return "\n".join(tool_lines[:7]).strip()

        if intent == "temporal_count":
            duration_source_lines = self._select_best_lines(
                query=query,
                lines=evidence_lines + candidate_lines if (evidence_lines or candidate_lines) else all_lines,
                max_items=3,
                prefer_time=True,
                prefer_number=True,
            )
            duration_answer_lines = duration_source_lines or (evidence_lines + candidate_lines if (evidence_lines or candidate_lines) else all_lines)
            anchor_dates = self._extract_temporal_anchor_dates(query, duration_answer_lines)
            if len(anchor_dates) >= 2:
                try:
                    from datetime import datetime

                    parsed_dates = [datetime.strptime(d, "%Y-%m-%d") for d in anchor_dates[:2]]
                    delta_days = abs((parsed_dates[1] - parsed_dates[0]).days)
                    duration_hint = self._format_duration_from_days(query, delta_days)
                except ValueError:
                    duration_hint = ""
            else:
                duration_hint = self._build_temporal_duration_hint(query, duration_answer_lines)
            if duration_hint:
                tool_lines.append(f"duration_hint={duration_hint}")
                tool_lines.append(f"duration_answer={duration_hint}")
            dates = anchor_dates or self._extract_temporal_dates(duration_answer_lines)
            if dates:
                tool_lines.append("temporal_points=" + " | ".join(dates[:4]))
            selected = self._select_best_lines(
                query=query,
                lines=duration_answer_lines,
                max_items=3,
                prefer_time=True,
                prefer_number=True,
            )
            if selected:
                tool_lines.append("support=" + " || ".join(selected[:3]))
            return "\n".join(tool_lines[:7]).strip()

        if intent == "temporal_compare":
            options = parse_choice_query(
                query,
                max_options=4,
                default_target_k=2,
            )
            if options:
                tool_lines.append("options=" + " | ".join(options[:4]))
            selected = self._select_best_lines(
                query=query,
                lines=all_lines,
                max_items=3,
                prefer_time=True,
            )
            if selected:
                tool_lines.append("anchors=" + " || ".join(selected[:3]))
            return "\n".join(tool_lines[:5]).strip()

        if intent == "preference":
            selected = self._select_best_lines(
                query=query,
                lines=all_lines,
                max_items=3,
                prefer_preference=True,
            )
            focus = self._query_focus_phrase(query)
            resources = self._extract_resource_phrases(evidence_pref_lines or selected or all_lines)
            if focus:
                tool_lines.append(f"preference_focus={focus}")
            if resources:
                tool_lines.append("resource_hint=" + " | ".join(resources[:3]))
            preference_summary = self._build_preference_summary(
                query=query,
                focus=focus,
                resources=resources,
                selected=selected,
            )
            if preference_summary:
                tool_lines.append("preference_summary=" + preference_summary)
                tool_lines.append("preference_answer=" + preference_summary)
            if selected:
                tool_lines.append("preference_hint=" + " || ".join(selected[:3]))
            else:
                fallback = self._select_best_lines(
                    query=query,
                    lines=all_lines,
                    max_items=2,
                )
                if fallback:
                    if focus:
                        tool_lines.append(f"preference_focus={focus}")
                    resources = self._extract_resource_phrases(fallback)
                    if resources:
                        tool_lines.append("resource_hint=" + " | ".join(resources[:3]))
                        fallback_summary = self._build_preference_summary(
                            query=query,
                            focus=focus,
                            resources=resources,
                            selected=fallback,
                        )
                        if fallback_summary:
                            tool_lines.append("preference_summary=" + fallback_summary)
                            tool_lines.append("preference_answer=" + fallback_summary)
                    tool_lines.append("preference_hint=" + " || ".join(fallback[:2]))
            return "\n".join(tool_lines[:6]).strip()

        return ""

    def build_tool_answer(
        self,
        *,
        query: str,
        graph_context: str,
        evidence_sentences: Sequence[Dict[str, object]],
        candidates: Sequence[Dict[str, object]],
        chunks: Sequence[Dict[str, object]],
    ) -> str:
        """Return a compact tool-side answer clue for the final prompt."""
        intent = self._classify_intent(query)
        if intent == "generic":
            return ""
        graph_lines = self._parse_graph_context(graph_context)
        evidence_lines = self._parse_evidence_lines(evidence_sentences, limit=4)
        candidate_lines = self._parse_evidence_lines(candidates, limit=3)
        all_lines = graph_lines + evidence_lines + candidate_lines
        evidence_pref_lines = evidence_lines + candidate_lines

        if intent == "count":
            answer_lines = evidence_lines + candidate_lines if (evidence_lines or candidate_lines) else all_lines
            count_answer, count_items = self._derive_count_answer_from_lines(
                query=query,
                lines=answer_lines,
            )
            if len(count_items) >= 2:
                return str(len(count_items))
            if count_answer:
                return count_answer
            selected = self._select_best_lines(
                query=query,
                lines=answer_lines,
                max_items=3,
                prefer_number=True,
            )
            if selected:
                for line in selected:
                    if self._line_has_list_shape(line):
                        spelled = self._extract_spelled_number(line)
                        if spelled >= 0:
                            return str(spelled)
                        m = re.search(r"\b(\d+)\b", line)
                        if m:
                            return str(int(m.group(1)))
            return ""

        if intent == "temporal_count":
            answer_lines = self._select_best_lines(
                query=query,
                lines=evidence_lines + candidate_lines if (evidence_lines or candidate_lines) else all_lines,
                max_items=3,
                prefer_time=True,
                prefer_number=True,
            ) or (evidence_lines + candidate_lines if (evidence_lines or candidate_lines) else all_lines)
            anchor_dates = self._extract_temporal_anchor_dates(query, answer_lines)
            if len(anchor_dates) >= 2:
                try:
                    from datetime import datetime

                    parsed_dates = [datetime.strptime(d, "%Y-%m-%d") for d in anchor_dates[:2]]
                    delta_days = abs((parsed_dates[1] - parsed_dates[0]).days)
                    duration_hint = self._format_duration_from_days(query, delta_days)
                    if duration_hint:
                        return duration_hint
                except ValueError:
                    pass
            duration_hint = self._build_temporal_duration_hint(query, answer_lines)
            if duration_hint:
                return duration_hint
            return ""

        if intent == "preference":
            focus = self._query_focus_phrase(query)
            resources = self._extract_resource_phrases(evidence_pref_lines)
            if not resources:
                resources = self._extract_resource_phrases(graph_lines)
            selected = self._select_best_lines(
                query=query,
                lines=all_lines,
                max_items=2,
                prefer_preference=True,
            )
            summary = self._build_preference_summary(
                query=query,
                focus=focus,
                resources=resources,
                selected=selected,
            )
            if summary:
                return summary
            return ""

        if intent == "temporal_compare":
            selected = self._select_best_lines(
                query=query,
                lines=all_lines,
                max_items=2,
                prefer_time=True,
            )
            if selected:
                return selected[0]
        return ""
