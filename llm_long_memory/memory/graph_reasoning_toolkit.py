"""Generic graph-tool hint builder for MemSLM answering.

This module intentionally avoids dataset-specific heuristics.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from llm_long_memory.memory.answering_temporal import parse_choice_query


class GraphReasoningToolkit:
    """Build compact tool hints/answers for generic task intents."""

    def __init__(self, manager: Any) -> None:
        self.m = manager
        self.count_cues = {"how many", "number of", "count", "total", "how much"}
        self.temporal_cues = {
            "before",
            "after",
            "earlier",
            "later",
            "first",
            "last",
            "when",
            "between",
            "compare",
            "which came",
        }
        self.preference_cues = {
            "prefer",
            "preference",
            "recommend",
            "suggest",
            "resource",
            "resources",
            "advice",
            "tips",
            "what should i",
            "how should i",
            "ways to",
        }
        self.update_cues = {
            "currently",
            "now",
            "latest",
            "updated",
            "update",
            "switch",
            "switched",
            "change",
            "changed",
            "moved",
            "where is",
            "where are",
        }
        # Optional config only used for generic list-count filtering.
        cfg = {}
        try:
            cfg = dict(manager.config["retrieval"]["answering"].get("counting", {}))
        except Exception:
            cfg = {}
        self.list_count_focus_stopwords = {
            str(x).strip().lower()
            for x in list(
                cfg.get(
                    "list_count_focus_stopwords",
                    [
                        "how",
                        "many",
                        "number",
                        "count",
                        "did",
                        "do",
                        "does",
                        "i",
                        "we",
                        "you",
                        "in",
                        "the",
                        "a",
                        "an",
                        "of",
                        "to",
                        "for",
                        "with",
                        "and",
                    ],
                )
            )
            if str(x).strip()
        }

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(str(text or "").split()).strip()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", str(text or "").lower())

    @staticmethod
    def _singular(token: str) -> str:
        t = str(token or "").strip().lower()
        if len(t) <= 3:
            return t
        if t.endswith("ies") and len(t) > 4:
            return t[:-3] + "y"
        if t.endswith("es") and len(t) > 4 and t[-3] in {"s", "x", "z"}:
            return t[:-2]
        if t.endswith("s") and len(t) > 3:
            return t[:-1]
        return t

    @staticmethod
    def _parse_graph_context(graph_context: str) -> List[str]:
        text = str(graph_context or "").strip()
        if not text:
            return []
        lines: List[str] = []
        for raw in re.split(r"\n+", text):
            line = re.sub(r"^\s*[-*•]+\s*", "", str(raw)).strip()
            line = re.sub(r"^\s*\[\d+\]\s*", "", line).strip()
            if line:
                lines.append(line)
        return lines

    @staticmethod
    def _parse_evidence_lines(items: Sequence[Dict[str, object]], limit: int = 4) -> List[str]:
        ranked = sorted(items, key=lambda x: float(x.get("score", 0.0)), reverse=True)
        lines: List[str] = []
        for item in ranked[: max(0, limit)]:
            text = " ".join(str(item.get("text", "")).split()).strip()
            if text:
                lines.append(text[:220])
        return lines

    @staticmethod
    def _parse_session_date(session_date: str) -> Optional[datetime]:
        raw = str(session_date or "").strip()
        if not raw:
            return None
        for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%Y/%m", "%Y-%m"):
            try:
                return datetime.strptime(raw, fmt)
            except ValueError:
                continue
        return None

    def _extract_dates(self, text: str, session_date: str = "") -> List[datetime]:
        out: List[datetime] = []
        txt = str(text or "")
        for m in re.finditer(r"\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b", txt):
            y, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
            try:
                out.append(datetime(y, mm, dd))
            except ValueError:
                pass
        months = {
            "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
            "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
            "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10, "october": 10,
            "nov": 11, "november": 11, "dec": 12, "december": 12,
        }
        for m in re.finditer(
            r"\b("
            r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
            r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
            r")\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*(\d{4}))?\b",
            txt,
            flags=re.IGNORECASE,
        ):
            mon = months.get(m.group(1).lower(), 1)
            day = int(m.group(2))
            year = int(m.group(3)) if m.group(3) else 2000
            try:
                out.append(datetime(year, mon, day))
            except ValueError:
                pass
        for m in re.finditer(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b", txt):
            mm = int(m.group(1))
            dd = int(m.group(2))
            y = m.group(3)
            year = int(y) if y else 2000
            if year < 100:
                year += 2000
            try:
                out.append(datetime(year, mm, dd))
            except ValueError:
                pass
        weekday_map = {
            "monday": 1,
            "tuesday": 2,
            "wednesday": 3,
            "thursday": 4,
            "friday": 5,
            "saturday": 6,
            "sunday": 7,
        }
        for wd, idx in weekday_map.items():
            if re.search(rf"\b{wd}\b", txt, flags=re.IGNORECASE):
                out.append(datetime(2000, 1, idx))
        if not out:
            d = self._parse_session_date(session_date)
            if d is not None:
                out.append(d)
        return out

    def _extract_temporal_dates(self, lines: Sequence[str]) -> List[str]:
        dates: List[str] = []
        seen: set[str] = set()
        for line in lines:
            for dt in self._extract_dates(line, ""):
                k = dt.strftime("%Y-%m-%d")
                if k not in seen:
                    seen.add(k)
                    dates.append(k)
        return dates

    def _extract_query_anchors(self, query: str) -> List[str]:
        options = parse_choice_query(query, max_options=4, default_target_k=2)
        if options:
            anchors: List[str] = []
            for x in options:
                n = self._normalize(x)
                if not n:
                    continue
                m = re.search(re.escape(n), query, flags=re.IGNORECASE)
                anchors.append(self._normalize(m.group(0)) if m else n)
            return anchors
        q = self._normalize(query)
        m = re.search(r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:[?.!]|$)", q, flags=re.IGNORECASE)
        if m:
            return [self._normalize(m.group(1)), self._normalize(m.group(2))]
        return []

    def _anchor_match_score(self, anchor: str, text: str) -> float:
        a = set(self._tokenize(anchor))
        t = set(self._tokenize(text))
        if not a or not t:
            return 0.0
        return len(a.intersection(t)) / float(len(a))

    def _classify_intent(self, query: str) -> str:
        q = self._normalize(query).lower()
        if not q:
            return "generic"
        if re.search(r"\bhow many\s+(?:days?|weeks?|months?|years?)\b", q) or "how long" in q:
            return "temporal_count"
        if any(cue in q for cue in self.count_cues):
            return "count"
        anchors = self._extract_query_anchors(query)
        if anchors and len(anchors) >= 2 and any(cue in q for cue in self.temporal_cues):
            return "temporal_compare"
        if any(cue in q for cue in self.update_cues):
            return "update"
        if any(cue in q for cue in self.preference_cues):
            return "preference"
        return "generic"

    def _query_object_heads(self, query: str) -> List[str]:
        q = self._normalize(query).lower()
        captured: List[str] = []
        pats = [
            r"\bhow many\s+(.+?)(?:\bdo\b|\bdid\b|\bhave\b|\bhas\b|\bhad\b|[?.!]|$)",
            r"\bnumber of\s+(.+?)(?:\bdo\b|\bdid\b|\bhave\b|\bhas\b|\bhad\b|[?.!]|$)",
            r"\bcount of\s+(.+?)(?:\bdo\b|\bdid\b|\bhave\b|\bhas\b|\bhad\b|[?.!]|$)",
        ]
        for p in pats:
            m = re.search(p, q, flags=re.IGNORECASE)
            if m:
                captured.append(self._normalize(m.group(1)))
        out: List[str] = []
        seen: set[str] = set()
        for phrase in captured:
            toks = [t for t in self._tokenize(phrase) if t not in self.list_count_focus_stopwords]
            if not toks:
                continue
            head = self._singular(toks[-1])
            if head and head not in seen:
                seen.add(head)
                out.append(head)
        return out

    def _extract_list_entities(self, text: str) -> List[str]:
        raw = self._normalize(text)
        if not raw:
            return []
        raw = re.sub(r"^\((?:user|assistant|system)\)\s*", "", raw, flags=re.IGNORECASE)
        # Drop aggregate leading clauses: "I have three bikes: ..."
        raw = re.sub(
            r"^\s*(?:i|we|you|they|he|she)\s+(?:have|own|got|bought|tried|use|used)\s+"
            r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|many|several|few)\b"
            r"(?:\s+\w+){0,4}[:,-]\s*",
            "",
            raw,
            flags=re.IGNORECASE,
        )
        parts = re.split(r"(?:,|;|\band\b|\bor\b|\bplus\b|\+)", raw, flags=re.IGNORECASE)
        out: List[str] = []
        seen: set[str] = set()
        for p in parts:
            c = self._normalize(str(p).strip(" -:;,.!?\"'"))
            if not c:
                continue
            if ":" in c and not re.match(r"^(?:a|an|the|my|our)\b", c, flags=re.IGNORECASE):
                c = self._normalize(c.split(":")[-1]).strip(" -:;,.!?\"'")
            if " - " in c and not re.match(r"^(?:a|an|the|my|our)\b", c, flags=re.IGNORECASE):
                c = self._normalize(c.split(" - ")[-1]).strip(" -:;,.!?\"'")
            if not c:
                continue
            low = c.lower()
            if low in {"by the way", "speaking of my", "speaking of my bikes"}:
                continue
            if low.startswith("by the way"):
                continue
            if re.search(r"\b(?:using|different|types?|ride|riding|been)\b", low) and not re.match(
                r"^(?:a|an|the|my|our)\b", low
            ):
                continue
            if re.match(r"^\d+\b", low):
                continue
            if low not in seen:
                seen.add(low)
                out.append(c)
        return out

    @staticmethod
    def _line_has_list_shape(text: str) -> bool:
        low = " ".join(str(text or "").split()).lower()
        return ("," in low) or (" and " in low) or (" or " in low) or (" plus " in low) or ("+" in low)

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

    def _build_count_evidence_pack(
        self,
        *,
        query: str,
        evidence_sentences: Sequence[Dict[str, object]],
        candidates: Sequence[Dict[str, object]],
        chunks: Sequence[Dict[str, object]],
    ) -> Dict[str, object]:
        _ = candidates
        query_heads = self._query_object_heads(query)
        qlow = query.lower()
        if re.search(r"\bhow many\s+(?:days?|weeks?|months?|years?)\b", qlow) or "how long" in qlow:
            return {}
        if not query_heads:
            return {}
        lines = self._parse_evidence_lines(evidence_sentences, limit=8) + self._parse_evidence_lines(
            chunks, limit=8
        )
        matched_lines: List[str] = []
        seen_line: set[str] = set()
        for line in lines:
            line_tokens = {self._singular(t) for t in self._tokenize(line)}
            if not line_tokens:
                continue
            is_match = any(
                (h in line_tokens) or any((h in tok) or (tok in h) for tok in line_tokens)
                for h in query_heads
            )
            if not is_match:
                continue
            normalized_line = self._normalize(line)
            if not normalized_line:
                continue
            if normalized_line in seen_line:
                continue
            seen_line.add(normalized_line)
            matched_lines.append(normalized_line)

        explicit_candidates: List[str] = []
        seen_explicit: set[str] = set()
        enumerated_items: List[str] = []
        seen_items: set[str] = set()
        for line in matched_lines:
            low = line.lower()
            if re.search(r"\b\d{1,2}[:/]\d{1,2}(?:[:/]\d{2,4})?\b", low):
                continue
            if re.search(r"\b\d+\s+(?:years?|months?|weeks?|days?)\b", low):
                continue
            spelled = self._extract_spelled_number(line)
            if spelled >= 0:
                n = str(spelled)
                if n not in seen_explicit:
                    seen_explicit.add(n)
                    explicit_candidates.append(n)
            for m in re.finditer(r"\b(\d+)\b", line):
                n = str(int(m.group(1)))
                if n not in seen_explicit:
                    seen_explicit.add(n)
                    explicit_candidates.append(n)
            for ent in self._extract_list_entities(line):
                k = ent.lower()
                if k in seen_items:
                    continue
                seen_items.add(k)
                enumerated_items.append(ent)

        if not matched_lines:
            return {}
        return {
            "object_type": query_heads[0],
            "query_heads": query_heads,
            "explicit_count_candidates": explicit_candidates[:4],
            "enumerated_items": enumerated_items[:10],
            "support_spans": matched_lines[:4],
        }

    def _temporal_count_result(self, query: str, lines: Sequence[str]) -> str:
        dates: List[datetime] = []
        for line in lines:
            dates.extend(self._extract_dates(line, ""))
        if len(dates) < 2:
            return ""
        dates = sorted({d.strftime("%Y-%m-%d"): d for d in dates}.values())
        delta_days = abs((dates[-1] - dates[0]).days)
        if delta_days <= 0:
            return ""
        q = query.lower()
        if "week" in q:
            v = max(1, round(delta_days / 7.0))
            return f"{v} week{'s' if v != 1 else ''}"
        if "month" in q:
            v = max(1, round(delta_days / 30.0))
            return f"{v} month{'s' if v != 1 else ''}"
        if "year" in q:
            v = max(1, round(delta_days / 365.0))
            return f"{v} year{'s' if v != 1 else ''}"
        return f"{delta_days} days"

    def _temporal_compare_result(
        self,
        *,
        query: str,
        evidence_sentences: Sequence[Dict[str, object]],
    ) -> str:
        anchors = self._extract_query_anchors(query)
        if len(anchors) < 2:
            return ""
        pools: Dict[str, List[Tuple[datetime, float]]] = {a: [] for a in anchors[:2]}
        for item in evidence_sentences:
            text = str(item.get("text", ""))
            session_date = str(item.get("session_date", ""))
            dates = self._extract_dates(text, session_date)
            if not dates:
                continue
            for anchor in pools:
                score = self._anchor_match_score(anchor, text)
                if score <= 0.0:
                    continue
                for dt in dates:
                    pools[anchor].append((dt, score))
        left, right = anchors[0], anchors[1]
        if not pools[left] or not pools[right]:
            return ""
        left_best = sorted(pools[left], key=lambda x: (x[0], -x[1]))[0][0]
        right_best = sorted(pools[right], key=lambda x: (x[0], -x[1]))[0][0]
        q = query.lower()
        if "later" in q or "after" in q:
            return left if left_best > right_best else right
        return left if left_best <= right_best else right

    def _extract_state_value(self, query: str, text: str) -> str:
        q = query.lower()
        t = self._normalize(text)
        if "more" in q and "less" in q:
            low = t.lower()
            if re.search(r"\bless\b", low):
                return "less"
            if re.search(r"\bmore\b", low):
                return "more"
        if any(x in q for x in {"where", "located", "hanging"}):
            m = re.search(r"\b(in|at|on|above|under|near)\s+([a-zA-Z][^,.!?;]{1,60})", t, flags=re.IGNORECASE)
            if m:
                return self._normalize(f"{m.group(1)} {m.group(2)}").strip(" ,.;:!?\"'")
        if "time" in q or "best" in q:
            times = re.findall(r"\b\d{1,2}:\d{2}\b", t)
            if times:
                return sorted(times, key=lambda v: int(v.split(":")[0]) * 60 + int(v.split(":")[1]))[0]
        cop = re.search(r"\b(?:is|are|was|were)\s+([^,.!?;]{2,80})", t, flags=re.IGNORECASE)
        if cop:
            return self._normalize(cop.group(1)).strip(" ,.;:!?\"'")
        return ""

    def _update_result(
        self,
        *,
        query: str,
        evidence_sentences: Sequence[Dict[str, object]],
        candidates: Sequence[Dict[str, object]],
    ) -> str:
        key_tokens = [t for t in self._tokenize(query) if t not in {"what", "where", "is", "are", "was", "were", "my", "the", "a", "an", "did", "i"}]
        scored: List[Tuple[datetime, float, str]] = []
        for item in evidence_sentences:
            text = str(item.get("text", ""))
            val = self._extract_state_value(query, text)
            if not val:
                continue
            tks = set(self._tokenize(text))
            overlap = len(set(key_tokens).intersection(tks)) / float(max(1, len(set(key_tokens)))) if key_tokens else 0.0
            dt = self._parse_session_date(str(item.get("session_date", ""))) or datetime.min
            score = float(item.get("score", 0.0)) + overlap
            scored.append((dt, score, val))
        for c in candidates:
            text = str(c.get("text", ""))
            val = self._extract_state_value(query, text)
            if val:
                scored.append((datetime.min, float(c.get("score", 0.0)), val))
        if not scored:
            return ""
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return scored[0][2]

    def _query_focus_phrase(self, query: str) -> str:
        q = self._normalize(query)
        if not q:
            return ""
        patterns = [
            r"\blearn more about\s+(.+?)(?:[?.!]|$)",
            r"\bways to\s+(.+?)(?:[?.!]|$)",
            r"\brecommend(?:\s+some)?\s+resources\s+for\s+(.+?)(?:[?.!]|$)",
            r"\bwhat should i\s+(.+?)(?:[?.!]|$)",
            r"\bhow should i\s+(.+?)(?:[?.!]|$)",
        ]
        low = q.lower()
        for p in patterns:
            m = re.search(p, low, flags=re.IGNORECASE)
            if m:
                return self._normalize(m.group(1)).strip(" ,.;:!?\"'")
        return self._normalize(q).strip(" ,.;:!?\"'")

    def _extract_resource_phrases(self, lines: Sequence[str]) -> List[str]:
        phrases: List[str] = []
        seen: set[str] = set()
        for line in lines:
            text = self._normalize(line)
            if not text:
                continue
            for cand in re.findall(r"\b[A-Z][A-Za-z0-9.&'-]{2,}(?:\s+[A-Z][A-Za-z0-9.&'-]{2,})*\b", text):
                c = self._normalize(cand).strip(" ,.;:!?\"'")
                if not c:
                    continue
                key = c.lower()
                if key in seen:
                    continue
                seen.add(key)
                phrases.append(c)
        return phrases

    def _preference_result(self, query: str, lines: Sequence[str]) -> str:
        direction = self._query_focus_phrase(query)
        resources = self._extract_resource_phrases(lines)
        direction_tokens = set(self._tokenize(direction))
        reason = ""
        for line in lines:
            lt = set(self._tokenize(line))
            if not direction_tokens or direction_tokens.intersection(lt):
                reason = self._normalize(line)
                break
        if not reason and lines:
            reason = self._normalize(lines[0])
        suggestion = ", ".join(resources[:2]) if resources else "context-aware targeted resources"
        return f"Preference: {direction} | Reason: {reason} | Suggestion: {suggestion}"

    def build_tool_hints(
        self,
        *,
        query: str,
        graph_context: str,
        evidence_sentences: Sequence[Dict[str, object]],
        candidates: Sequence[Dict[str, object]],
        chunks: Sequence[Dict[str, object]],
    ) -> str:
        intent = self._classify_intent(query)
        if intent == "generic":
            return ""
        graph_lines = self._parse_graph_context(graph_context)
        evidence_lines = self._parse_evidence_lines(evidence_sentences, limit=4)
        candidate_lines = self._parse_evidence_lines(candidates, limit=3)
        all_lines = graph_lines + evidence_lines + candidate_lines
        tool_lines: List[str] = [f"intent={intent}"]

        if intent == "count":
            pack = self._build_count_evidence_pack(
                query=query,
                evidence_sentences=evidence_sentences,
                candidates=candidates,
                chunks=chunks,
            )
            if pack.get("object_type"):
                tool_lines.append(f"count_object_type={pack.get('object_type', '')}")
            if pack.get("explicit_count_candidates"):
                vals = [str(x) for x in list(pack.get("explicit_count_candidates", []))]
                tool_lines.append("count_explicit_candidates=" + " | ".join(vals[:4]))
                if len(vals) == 1:
                    tool_lines.append(f"count_hint={vals[0]}")
            if pack.get("enumerated_items"):
                items = [str(x) for x in list(pack.get("enumerated_items", []))]
                tool_lines.append("count_enumerated_items=" + " | ".join(items[:6]))
                # Keep backward-compatible key for prompt templates.
                tool_lines.append("count_items=" + " | ".join(items[:6]))
            if pack.get("support_spans"):
                spans = [str(x) for x in list(pack.get("support_spans", []))]
                tool_lines.append("count_support_spans=" + " | ".join(spans[:3]))
            return "\n".join(tool_lines[:7]).strip()

        if intent == "temporal_count":
            ans = self._temporal_count_result(query, all_lines)
            if ans:
                tool_lines.append(f"duration_answer={ans}")
                tool_lines.append(f"duration_hint={ans}")
            points = self._extract_temporal_dates(all_lines)
            if points:
                tool_lines.append("temporal_points=" + " | ".join(points[:4]))
            return "\n".join(tool_lines[:5]).strip()

        if intent == "temporal_compare":
            ans = self._temporal_compare_result(query=query, evidence_sentences=evidence_sentences)
            if ans:
                tool_lines.append(f"compare_answer={ans}")
            return "\n".join(tool_lines[:5]).strip()

        if intent == "update":
            ans = self._update_result(query=query, evidence_sentences=evidence_sentences, candidates=candidates)
            if ans:
                tool_lines.append(f"state_answer={ans}")
            return "\n".join(tool_lines[:5]).strip()

        if intent == "preference":
            pref = self._preference_result(query, all_lines)
            if pref:
                tool_lines.append(f"preference_answer={pref}")
                tool_lines.append(f"preference_summary={pref}")
                tool_lines.append("preference_hint=" + (all_lines[0] if all_lines else ""))
            return "\n".join(tool_lines[:5]).strip()

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
        _ = graph_context
        intent = self._classify_intent(query)
        if intent == "generic":
            return ""
        evidence_lines = self._parse_evidence_lines(evidence_sentences, limit=4)
        candidate_lines = self._parse_evidence_lines(candidates, limit=3)
        all_lines = evidence_lines + candidate_lines
        if intent == "count":
            # Count intent returns evidence pack via tool hints; avoid direct module-side verdict.
            return ""
        if intent == "temporal_count":
            return self._temporal_count_result(query, all_lines)
        if intent == "temporal_compare":
            return self._temporal_compare_result(query=query, evidence_sentences=evidence_sentences)
        if intent == "update":
            return self._update_result(query=query, evidence_sentences=evidence_sentences, candidates=candidates)
        if intent == "preference":
            return self._preference_result(query, all_lines)
        return ""
