"""Minimal Ollama client for local conversational models."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Sequence, Union

from llm_long_memory.utils.helpers import load_config

Message = Dict[str, str]


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
        body = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        req = urllib.request.Request(
            url=f"{self.host}/api/generate",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener.open(req, timeout=self.request_timeout_sec) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            details = ""
            try:
                details = exc.read().decode("utf-8").strip()
            except (OSError, UnicodeDecodeError):
                details = ""
            if exc.code == 404:
                raise RuntimeError(f"Model `{self.model_name}` not found. Run `ollama pull {self.model_name}`.") from exc
            if exc.code == 502:
                raise RuntimeError(
                    "HTTP 502 from Ollama API. This is often a local proxy/host mismatch; "
                    f"try host `{self.host}`."
                    + (f" Details: {details}" if details else "")
                ) from exc
            raise RuntimeError(f"Ollama HTTP error {exc.code}" + (f": {details}" if details else "")) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.host}. Start with `ollama serve`."
            ) from exc

        if payload.get("error"):
            raise RuntimeError(str(payload["error"]))
        return str(payload.get("response", "")).strip()
