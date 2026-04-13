"""Minimal Ollama client for local conversational models."""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Sequence, Union

from llm_long_memory.utils.helpers import load_config
from llm_long_memory.utils.logger import logger

Message = Dict[str, str]


def ollama_generate_with_retry(
    *,
    host: str,
    model: str,
    prompt: str,
    temperature: float,
    timeout_sec: int,
    opener: urllib.request.OpenerDirector,
    max_attempts: int,
    backoff_sec: float,
    retry_on_timeout: bool,
    retry_on_http_502: bool,
    retry_on_url_error: bool,
    max_output_tokens: Optional[int] = None,
    think: Optional[bool] = None,
    response_format: Optional[str] = None,
) -> str:
    """Call Ollama /api/generate with retry for transient local failures."""
    options: Dict[str, Union[float, int]] = {"temperature": temperature}
    if max_output_tokens is not None and int(max_output_tokens) > 0:
        options["num_predict"] = int(max_output_tokens)
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }
    if think is not None:
        body["think"] = bool(think)
    if response_format is not None and str(response_format).strip():
        body["format"] = str(response_format).strip()
    req = urllib.request.Request(
        url=f"{host}/api/generate",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    attempts = max(1, int(max_attempts))
    wait_sec = max(0.0, float(backoff_sec))
    last_err: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        try:
            with opener.open(req, timeout=timeout_sec) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            if payload.get("error"):
                raise RuntimeError(str(payload["error"]))
            return str(payload.get("response", "")).strip()
        except urllib.error.HTTPError as exc:
            details = ""
            try:
                details = exc.read().decode("utf-8").strip()
            except (OSError, UnicodeDecodeError):
                details = ""
            if exc.code == 404:
                raise RuntimeError(f"Model `{model}` not found. Run `ollama pull {model}`.") from exc
            should_retry = bool(retry_on_http_502 and exc.code == 502 and attempt < attempts)
            if should_retry:
                logger.warn(
                    f"Ollama HTTP {exc.code}; retrying {attempt}/{attempts} after {wait_sec:.1f}s."
                )
                time.sleep(wait_sec)
                continue
            if exc.code == 502:
                raise RuntimeError(
                    "HTTP 502 from Ollama API. This is often a local proxy/host mismatch; "
                    f"try host `{host}`."
                    + (f" Details: {details}" if details else "")
                ) from exc
            raise RuntimeError(f"Ollama HTTP error {exc.code}" + (f": {details}" if details else "")) from exc
        except (TimeoutError, socket.timeout) as exc:
            last_err = exc
            should_retry = bool(retry_on_timeout and attempt < attempts)
            if should_retry:
                logger.warn(
                    f"Ollama timeout; retrying {attempt}/{attempts} after {wait_sec:.1f}s."
                )
                time.sleep(wait_sec)
                continue
            raise RuntimeError(
                f"Ollama request timed out at {host} (timeout={timeout_sec}s)."
            ) from exc
        except urllib.error.URLError as exc:
            last_err = exc
            reason = getattr(exc, "reason", None)
            is_timeout = isinstance(reason, (TimeoutError, socket.timeout))
            should_retry = bool(
                attempt < attempts and ((retry_on_timeout and is_timeout) or retry_on_url_error)
            )
            if should_retry:
                logger.warn(
                    "Ollama URL error; "
                    f"retrying {attempt}/{attempts} after {wait_sec:.1f}s: {exc}"
                )
                time.sleep(wait_sec)
                continue
            raise RuntimeError(
                f"Cannot reach Ollama at {host}. Start with `ollama serve`."
            ) from exc
        except (RuntimeError, ValueError, TypeError) as exc:
            last_err = exc
            raise

    if last_err is not None:
        raise RuntimeError(f"Ollama call failed after {attempts} attempts: {last_err}") from last_err
    raise RuntimeError(f"Ollama call failed after {attempts} attempts.")


class LLM:
    """Simple local LLM wrapper using Ollama HTTP API."""

    def __init__(self, model_name: Optional[str] = None, host: Optional[str] = None) -> None:
        llm_cfg = load_config()["llm"]
        supported_models = set(llm_cfg["supported_models"])
        selected_model = model_name or str(llm_cfg["default_model"])
        if selected_model not in supported_models:
            supported = ", ".join(sorted(supported_models))
            raise ValueError(f"Unsupported model: {selected_model}. Supported: {supported}")
        self.model_name = selected_model
        self.host = (host or str(llm_cfg["host"])).rstrip("/")
        self.temperature = float(llm_cfg["temperature"])
        self.request_timeout_sec = int(llm_cfg["request_timeout_sec"])
        retry_cfg = dict(llm_cfg.get("retry", {}))
        self.retry_max_attempts = int(retry_cfg.get("max_attempts", 1))
        self.retry_backoff_sec = float(retry_cfg.get("backoff_sec", 0.0))
        self.retry_on_timeout = bool(retry_cfg.get("retry_on_timeout", True))
        self.retry_on_http_502 = bool(retry_cfg.get("retry_on_http_502", True))
        self.retry_on_url_error = bool(retry_cfg.get("retry_on_url_error", False))
        self.max_output_tokens = int(llm_cfg.get("max_output_tokens", 0))
        # Force direct local connection; ignore HTTP(S)_PROXY env.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def chat(self, messages: Union[str, Sequence[Message]]) -> str:
        """Generate assistant response from a message list or plain user text."""
        normalized = self._normalize_messages(messages)
        prompt = self._messages_to_prompt(normalized)
        return self._generate(prompt)

    @staticmethod
    def _normalize_messages(messages: Union[str, Sequence[Message]]) -> List[Message]:
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        out: List[Message] = []
        for msg in messages:
            role = str(msg.get("role", "")).strip().lower()
            content = str(msg.get("content", "")).strip()
            if role and content:
                out.append({"role": role, "content": content})
        if not out:
            raise ValueError("messages must include at least one non-empty item.")
        return out

    @staticmethod
    def _messages_to_prompt(messages: Sequence[Message]) -> str:
        lines: List[str] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                lines.append(f"System: {content}")
            elif role == "assistant":
                lines.append(f"Assistant: {content}")
            else:
                lines.append(f"User: {content}")
        lines.append("Assistant:")
        return "\n".join(lines)

    def _generate(self, prompt: str) -> str:
        return ollama_generate_with_retry(
            host=self.host,
            model=self.model_name,
            prompt=prompt,
            temperature=self.temperature,
            timeout_sec=self.request_timeout_sec,
            opener=self._opener,
            max_attempts=self.retry_max_attempts,
            backoff_sec=self.retry_backoff_sec,
            retry_on_timeout=self.retry_on_timeout,
            retry_on_http_502=self.retry_on_http_502,
            retry_on_url_error=self.retry_on_url_error,
            max_output_tokens=self.max_output_tokens if self.max_output_tokens > 0 else None,
        )
