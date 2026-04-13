"""Short-term memory module for recent dialogue context."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from llm_long_memory.utils.helpers import load_config
from llm_long_memory.utils.logger import logger


Message = Dict[str, str]


class ShortMemory:
    """FIFO buffer for recent dialogue turns used as immediate LLM context."""

    def __init__(self, max_turns: Optional[int] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize short-term memory with a configurable capacity."""
        cfg = config or load_config()
        configured_size = int(cfg["memory"]["short_memory_size"])
        max_turns = configured_size if max_turns is None else max_turns
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        self.max_turns = max_turns
        self._buffer: List[Message] = []
        logger.info(f"ShortMemory initialized (max_turns={self.max_turns}).")

    def add(self, message: Message) -> None:
        """Add a message to the short-term buffer."""
        self._buffer.append(message)
        role = message.get("role", "unknown")
        logger.debug(f"ShortMemory.add: added role={role}, size={len(self._buffer)}.")

    def get(self) -> List[Message]:
        """Return current short-term context buffer."""
        logger.debug(f"ShortMemory.get: returning {len(self._buffer)} messages.")
        return list(self._buffer)

    def flush_to_mid_memory(self, mid_memory: object) -> List[Message]:
        """Flush oldest messages into mid-memory until within capacity."""
        moved: List[Message] = []
        flushed = 0
        while len(self._buffer) > self.max_turns:
            old_message = self._buffer.pop(0)
            mid_memory.add(old_message)
            moved.append(old_message)
            flushed += 1
        if flushed:
            logger.info(
                f"ShortMemory.flush_to_mid_memory: flushed {flushed} messages to MidMemory."
            )
        else:
            logger.debug("ShortMemory.flush_to_mid_memory: no flush needed.")
        return moved

    def clear(self) -> None:
        """Clear all short-term memory turns."""
        size = len(self._buffer)
        self._buffer.clear()
        logger.info(f"ShortMemory.clear: removed {size} messages.")
