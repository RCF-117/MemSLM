"""Local LLM judge helpers for thesis evaluation."""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional

from llm_long_memory.llm.ollama_client import ollama_generate_with_retry
from llm_long_memory.utils.helpers import load_config


def _extract_json_object(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if not match:
        return {}
    candidate = match.group(0)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


@dataclass
class JudgeResult:
    """Structured verdict returned by the judge model."""

    is_correct: bool
    verdict: str
    reason: str


class LocalLLMJudge:
    """Use a local Ollama model to judge answer correctness semantically."""

    def __init__(
        self,
        *,
        model_name: Optional[str] = None,
        host: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout_sec: Optional[int] = None,
        max_output_tokens: int = 160,
        max_attempts: int = 1,
        backoff_sec: float = 0.0,
        retry_on_timeout: bool = True,
        retry_on_http_502: bool = True,
        retry_on_url_error: bool = False,
    ) -> None:
        llm_cfg = load_config()["llm"]
        selected_model = model_name or str(llm_cfg["default_model"])
        self.model_name = selected_model
        self.host = (host or str(llm_cfg["host"])).rstrip("/")
        self.temperature = (
            float(temperature) if temperature is not None else float(llm_cfg["temperature"])
        )
        self.timeout_sec = int(timeout_sec) if timeout_sec is not None else int(llm_cfg["request_timeout_sec"])
        self.max_output_tokens = int(max_output_tokens)
        retry_cfg = dict(llm_cfg.get("retry", {}))
        self.max_attempts = int(max_attempts or retry_cfg.get("max_attempts", 1))
        self.backoff_sec = float(backoff_sec if backoff_sec is not None else retry_cfg.get("backoff_sec", 0.0))
        self.retry_on_timeout = bool(retry_on_timeout if retry_on_timeout is not None else retry_cfg.get("retry_on_timeout", True))
        self.retry_on_http_502 = bool(retry_on_http_502 if retry_on_http_502 is not None else retry_cfg.get("retry_on_http_502", True))
        self.retry_on_url_error = bool(retry_on_url_error if retry_on_url_error is not None else retry_cfg.get("retry_on_url_error", False))
        judge_cfg = dict(llm_cfg.get("judge", {}))
        self.judge_think = bool(judge_cfg.get("think", False))
        self.judge_response_format = str(judge_cfg.get("response_format", "json")).strip() or "json"
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @staticmethod
    def _normalize_space(text: str) -> str:
        return " ".join(str(text).split())

    def _build_prompt(self, question: str, gold: str, prediction: str) -> str:
        return (
            "You are a strict but fair evaluator for a memory QA benchmark.\n"
            "Judge whether the prediction is semantically correct with respect to the gold answer.\n"
            "Ignore formatting differences, punctuation, and minor paraphrases.\n"
            "Prefer correctness if the prediction captures the same meaning or an explicitly accepted equivalent.\n"
            "Return JSON only with this schema:\n"
            '{"is_correct": true|false, "verdict": "correct|incorrect", "reason": "short reason"}\n'
            "Rules:\n"
            "- If the prediction is an accepted paraphrase or equivalent form, mark correct.\n"
            "- If the prediction is broader, partially correct, or missing the key answer, mark incorrect.\n"
            "- If the gold answer contains multiple acceptable forms, any equivalent form counts as correct.\n"
            "- Keep the reason short and grounded in the comparison.\n\n"
            f"Question: {self._normalize_space(question)}\n"
            f"Gold Answer: {self._normalize_space(gold)}\n"
            f"Prediction: {self._normalize_space(prediction)}\n"
        )

    def judge(self, question: str, gold: str, prediction: str) -> JudgeResult:
        """Return a semantic correctness verdict from the judge model."""
        prompt = self._build_prompt(question, gold, prediction)
        raw = ollama_generate_with_retry(
            host=self.host,
            model=self.model_name,
            prompt=prompt,
            temperature=self.temperature,
            timeout_sec=self.timeout_sec,
            opener=self._opener,
            max_attempts=self.max_attempts,
            backoff_sec=self.backoff_sec,
            retry_on_timeout=self.retry_on_timeout,
            retry_on_http_502=self.retry_on_http_502,
            retry_on_url_error=self.retry_on_url_error,
            max_output_tokens=self.max_output_tokens if self.max_output_tokens > 0 else None,
            think=self.judge_think,
            response_format=self.judge_response_format,
        )
        payload = _extract_json_object(raw)
        verdict = str(payload.get("verdict", "")).strip().lower()
        reason = str(payload.get("reason", "")).strip()
        is_correct = bool(payload.get("is_correct", False))
        if verdict in {"correct", "incorrect"}:
            is_correct = verdict == "correct"
        if not reason:
            reason = raw.strip()[:240]
        return JudgeResult(is_correct=is_correct, verdict=verdict or ("correct" if is_correct else "incorrect"), reason=reason)
