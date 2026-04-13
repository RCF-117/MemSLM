"""Soft-quality gate for long-memory event acceptance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from llm_long_memory.memory.long_memory_text_utils import LongMemoryTextUtils


@dataclass(frozen=True)
class LongMemoryGateConfig:
    enabled: bool
    hard_only: bool
    hard_require_action: bool
    hard_require_subject_or_object: bool
    hard_min_confidence: float
    quality_threshold: float
    weight_completeness: float
    weight_grounding: float
    weight_keyword: float
    weight_structure: float
    pronoun_penalty: float
    keyword_empty_score: float
    grounding_missing_score: float


class LongMemoryGate:
    """Compute gate decisions with minimal hard rules + soft quality score."""

    def __init__(
        self,
        *,
        config: LongMemoryGateConfig,
        text_utils: LongMemoryTextUtils,
        subject_pronouns: Iterable[str],
    ) -> None:
        self.config = config
        self.text_utils = text_utils
        self.subject_pronouns = {
            str(x).strip().lower() for x in subject_pronouns if str(x).strip()
        }

    @staticmethod
    def _clip01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def is_pronoun_subject(self, subject: str) -> bool:
        s = self.text_utils.normalize_space(subject).lower()
        return bool(s) and s in self.subject_pronouns

    def span_overlap_ratio(self, span: str, content: str) -> float:
        s = self.text_utils.normalize_space(span).lower()
        c = self.text_utils.normalize_space(content).lower()
        if not s:
            return 0.0
        if s in c:
            return 1.0
        s_tokens = set(self.text_utils.tokenize(s))
        if not s_tokens:
            return 0.0
        c_tokens = set(self.text_utils.tokenize(c))
        return float(len(s_tokens.intersection(c_tokens))) / float(max(1, len(s_tokens)))

    def keyword_noise_ratio(self, keywords: List[str]) -> float:
        if not keywords:
            return 1.0
        noise_terms = self.subject_pronouns.union({"assistant", "user", "system", "im", "ck"})
        bad = 0
        for kw in keywords:
            token = str(kw).strip().lower()
            if (not token) or (token in noise_terms):
                bad += 1
        return float(bad) / float(max(1, len(keywords)))

    def quality_score(
        self,
        *,
        subject: str,
        action: str,
        obj: str,
        time_text: str,
        location_text: str,
        evidence_span: str,
        source_content: str,
        keywords: List[str],
    ) -> float:
        completeness = float(int(bool(subject)) + int(bool(action)) + int(bool(obj))) / 3.0
        if evidence_span:
            grounding = self._clip01(self.span_overlap_ratio(evidence_span, source_content))
        else:
            grounding = self._clip01(self.config.grounding_missing_score)
        if keywords:
            keyword_quality = self._clip01(1.0 - self.keyword_noise_ratio(keywords))
        else:
            keyword_quality = self._clip01(self.config.keyword_empty_score)
        structure_bonus = float(int(bool(time_text)) + int(bool(location_text))) / 2.0

        score = (
            (self.config.weight_completeness * completeness)
            + (self.config.weight_grounding * grounding)
            + (self.config.weight_keyword * keyword_quality)
            + (self.config.weight_structure * structure_bonus)
        )
        if self.is_pronoun_subject(subject):
            score -= self.config.pronoun_penalty
        return self._clip01(score)

    def evaluate(
        self,
        *,
        subject: str,
        action: str,
        obj: str,
        confidence: float,
        time_text: str,
        location_text: str,
        evidence_span: str,
        source_content: str,
        keywords: List[str],
    ) -> Tuple[bool, str, float]:
        if not self.config.enabled:
            return True, "disabled", 1.0

        if confidence < self.config.hard_min_confidence:
            return False, "hard_low_confidence", 0.0
        if self.config.hard_require_action and (not action):
            return False, "hard_missing_action", 0.0
        if self.config.hard_require_subject_or_object and (not subject) and (not obj):
            return False, "hard_missing_subject_object", 0.0

        if self.config.hard_only:
            return True, "accepted_hard_only", 1.0

        score = self.quality_score(
            subject=subject,
            action=action,
            obj=obj,
            time_text=time_text,
            location_text=location_text,
            evidence_span=evidence_span,
            source_content=source_content,
            keywords=keywords,
        )
        if score < self.config.quality_threshold:
            return False, "quality_below_threshold", score
        return True, "accepted", score

    def reject_stats(self, reasons: Dict[str, int]) -> Dict[str, int]:
        return {
            "reject_reason_hard_low_confidence": int(reasons.get("hard_low_confidence", 0)),
            "reject_reason_hard_missing_action": int(reasons.get("hard_missing_action", 0)),
            "reject_reason_hard_missing_subject_object": int(
                reasons.get("hard_missing_subject_object", 0)
            ),
            "reject_reason_quality_below_threshold": int(
                reasons.get("quality_below_threshold", 0)
            ),
        }
