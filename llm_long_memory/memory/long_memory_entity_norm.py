"""Entity normalization helpers for long-memory event extraction."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple


class LongMemoryEntityNormalizer:
    """Normalize noisy/pronoun-heavy fields into stable entity phrases."""

    def __init__(
        self,
        *,
        enabled: bool,
        pronoun_map: Dict[str, str],
        possessive_prefixes: Iterable[str],
    ) -> None:
        self.enabled = bool(enabled)
        self.pronoun_map = {
            str(k).strip().lower(): str(v).strip().lower()
            for k, v in dict(pronoun_map).items()
            if str(k).strip() and str(v).strip()
        }
        self.possessive_prefixes = {
            str(x).strip().lower() for x in possessive_prefixes if str(x).strip()
        }

    @staticmethod
    def _norm_space(text: str) -> str:
        return " ".join(str(text).strip().split())

    def _map_pronoun_subject(self, subject: str) -> str:
        s = self._norm_space(subject).lower()
        if not s:
            return ""
        mapped = self.pronoun_map.get(s, "")
        return mapped or s

    def _strip_possessive_prefix(self, text: str) -> Tuple[str, str]:
        s = self._norm_space(text)
        if not s:
            return "", ""
        parts = s.split(" ", 1)
        head = parts[0].lower()
        tail = parts[1] if len(parts) > 1 else ""
        if head in self.possessive_prefixes and tail:
            return head, self._norm_space(tail)
        return "", s

    def normalize_subject(self, subject: str, role: str) -> str:
        if not self.enabled:
            return self._norm_space(subject)

        subj = self._norm_space(subject)
        if not subj:
            return ""

        mapped = self._map_pronoun_subject(subj)
        if mapped != subj.lower():
            return mapped

        pref, rest = self._strip_possessive_prefix(subj)
        if pref:
            anchor = self.pronoun_map.get(pref, "user" if role == "user" else role)
            return self._norm_space(f"{anchor} {rest}")

        return subj.lower()

    def normalize_object(self, obj: str, role: str) -> str:
        if not self.enabled:
            return self._norm_space(obj)

        text = self._norm_space(obj)
        if not text:
            return ""
        pref, rest = self._strip_possessive_prefix(text)
        if pref:
            anchor = self.pronoun_map.get(pref, "user" if role == "user" else role)
            return self._norm_space(f"{anchor} {rest}")
        return text
