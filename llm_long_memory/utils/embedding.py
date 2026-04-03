"""Local embedding functions using Ollama's embeddings API.

This replaces the earlier hash embedding with a reliable local vector model.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections import OrderedDict
from typing import Any, Dict, Tuple

import numpy as np

from llm_long_memory.utils.helpers import load_config
from llm_long_memory.utils.logger import logger

_EMBED_RUNTIME: Dict[str, Any] | None = None


def _get_runtime() -> Dict[str, Any]:
    global _EMBED_RUNTIME
    if _EMBED_RUNTIME is None:
        cfg = load_config()
        _EMBED_RUNTIME = {
            "embedding_cfg": dict(cfg["embedding"]),
            "llm_cfg": dict(cfg["llm"]),
        }
    return _EMBED_RUNTIME


def embed(text: str, dim: int) -> np.ndarray:
    """Embed text using a local Ollama embedding model.

    The returned vector is L2-normalized. If the embedding model's native
    dimension doesn't match `dim`, it will be truncated/padded only when
    enabled by config.
    """
    runtime = _get_runtime()
    embedding_cfg = runtime["embedding_cfg"]
    llm_cfg = runtime["llm_cfg"]

    model = str(embedding_cfg["model"])
    host = str(llm_cfg["host"]).rstrip("/")
    timeout = int(llm_cfg["embedding_timeout_sec"])
    target_dim = int(dim)
    truncate_or_pad = bool(embedding_cfg["truncate_or_pad"])

    clean = str(text or "").strip()
    if not clean:
        return np.zeros(target_dim, dtype=np.float32)

    state = _EmbeddingState.get(
        host=host,
        model=model,
        cache_size=int(embedding_cfg["cache_size"]),
        timeout_sec=timeout,
    )
    state.request_count += 1
    cached = state.cache_get((clean, target_dim, truncate_or_pad))
    if cached is not None:
        return cached

    payload = {"model": model, "prompt": clean}
    if state.request_count == 1 or (state.request_count % 200 == 0):
        logger.info(
            "Embedding request: "
            f"model={model}, text_chars={len(clean)}, target_dim={target_dim}, "
            f"request_count={state.request_count}."
        )
    req = urllib.request.Request(
        url=f"{host}/api/embeddings",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with state.opener.open(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
        raise RuntimeError(
            f"Ollama embeddings call failed (host={host}, model={model}): {exc}"
        ) from exc

    vec = data.get("embedding")
    if not isinstance(vec, list) or not vec:
        raise RuntimeError(f"Ollama embeddings returned invalid payload keys: {list(data.keys())}")

    arr = np.asarray(vec, dtype=np.float32)
    if arr.ndim != 1:
        arr = arr.reshape(-1).astype(np.float32)

    if arr.size != target_dim:
        if not truncate_or_pad:
            raise ValueError(
                f"Embedding dim mismatch: got={arr.size}, expected={target_dim}. "
                "Set embedding.truncate_or_pad=true or adjust embedding.dim."
            )
        fixed = np.zeros(target_dim, dtype=np.float32)
        limit = min(arr.size, target_dim)
        fixed[:limit] = arr[:limit]
        arr = fixed

    norm = float(np.linalg.norm(arr))
    if norm > 0.0:
        arr = (arr / norm).astype(np.float32)
    else:
        arr = np.zeros(target_dim, dtype=np.float32)

    state.cache_put((clean, target_dim, truncate_or_pad), arr)
    return arr


class _EmbeddingState:
    _instances: Dict[Tuple[str, str], "_EmbeddingState"] = {}

    def __init__(self, host: str, model: str, cache_size: int, timeout_sec: int) -> None:
        self.host = host
        self.model = model
        self.cache_size = max(0, int(cache_size))
        self.timeout_sec = int(timeout_sec)
        self.request_count = 0
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self._cache: "OrderedDict[Tuple[str, int, bool], np.ndarray]" = OrderedDict()

    @classmethod
    def get(cls, host: str, model: str, cache_size: int, timeout_sec: int) -> "_EmbeddingState":
        key = (host, model)
        inst = cls._instances.get(key)
        if inst is None or inst.cache_size != max(0, int(cache_size)) or inst.timeout_sec != int(timeout_sec):
            inst = _EmbeddingState(host, model, cache_size, timeout_sec)
            cls._instances[key] = inst
            logger.info(f"Embedding client initialized: host={host}, model={model}, cache_size={cache_size}.")
        return inst

    def cache_get(self, key: Tuple[str, int, bool]) -> np.ndarray | None:
        if self.cache_size <= 0:
            return None
        hit = self._cache.get(key)
        if hit is None:
            return None
        self._cache.move_to_end(key)
        return hit

    def cache_put(self, key: Tuple[str, int, bool], value: np.ndarray) -> None:
        if self.cache_size <= 0:
            return
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
