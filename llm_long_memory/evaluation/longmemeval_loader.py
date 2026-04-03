"""Streaming loader for LongMemEval evaluation instances.

This module preserves LongMemEval's instance-level structure:
one yielded item == one evaluation instance (question + haystack sessions).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List

from llm_long_memory.utils.helpers import load_config


Turn = Dict[str, Any]
Session = List[Turn]
EvalInstance = Dict[str, Any]


def _normalize_turn(turn: Any) -> Turn | None:
    """Normalize one turn dict into {'role','content',...} format."""
    if not isinstance(turn, dict):
        return None
    content = turn.get("content")
    if content is None:
        content = turn.get("text")
    if content is None:
        content = turn.get("utterance")
    text = str(content or "").strip()
    if not text:
        return None

    role = str(turn.get("role") or turn.get("speaker") or "user").strip().lower()
    if not role:
        role = "user"
    if role not in {"user", "assistant", "system"}:
        role = "user"

    normalized: Turn = {"role": role, "content": text}
    if bool(turn.get("has_answer", False)):
        normalized["has_answer"] = True
    return normalized


def _normalize_session(session: Any) -> Session:
    """Normalize one session into a list of turns."""
    if not isinstance(session, list):
        return []
    out: Session = []
    for turn in session:
        normalized = _normalize_turn(turn)
        if normalized is not None:
            out.append(normalized)
    return out


def _normalize_instance(item: Any) -> EvalInstance | None:
    """Normalize one raw JSON object into LongMemEval-like instance shape.

    Returned dict always includes:
    question_id, question_type, question, answer, question_date,
    haystack_session_ids, haystack_dates, haystack_sessions, answer_session_ids
    """
    if isinstance(item, list):
        # Compatibility fallback: treat list-of-turns as one session instance.
        session = _normalize_session(item)
        if not session:
            return None
        return {
            "question_id": "",
            "question_type": "",
            "question": "",
            "answer": "",
            "question_date": "",
            "haystack_session_ids": [],
            "haystack_dates": [],
            "haystack_sessions": [session],
            "answer_session_ids": [],
        }

    if not isinstance(item, dict):
        return None

    raw_sessions = item.get("haystack_sessions")
    sessions: List[Session] = []
    if isinstance(raw_sessions, list):
        for raw_session in raw_sessions:
            session = _normalize_session(raw_session)
            if session:
                sessions.append(session)

    # Compatibility fallback for non-LongMemEval data.
    if not sessions:
        fallback_messages = item.get("messages")
        fallback_history = item.get("history")
        for candidate in (fallback_messages, fallback_history):
            session = _normalize_session(candidate)
            if session:
                sessions.append(session)

    if not sessions:
        return None

    return {
        "question_id": str(item.get("question_id", "")),
        "question_type": str(item.get("question_type", "")),
        "question": str(item.get("question", "")),
        "answer": str(item.get("answer", "")),
        "question_date": str(item.get("question_date", "")),
        "haystack_session_ids": list(item.get("haystack_session_ids", [])),
        "haystack_dates": list(item.get("haystack_dates", [])),
        "haystack_sessions": sessions,
        "answer_session_ids": list(item.get("answer_session_ids", [])),
    }


def load_stream(path: str) -> Generator[EvalInstance, None, None]:
    """Yield one normalized LongMemEval instance at a time."""
    stream_read_size = int(load_config()["dataset"]["stream_read_size"])
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as file:
        first_char = file.read(1)
        file.seek(0)

        if first_char == "[":
            decoder = json.JSONDecoder()
            buffer = ""
            while True:
                chunk = file.read(stream_read_size)
                if not chunk:
                    break
                buffer += chunk
                while True:
                    buffer = buffer.lstrip()
                    if not buffer:
                        break
                    if buffer[0] in "[,":
                        buffer = buffer[1:]
                        continue
                    if buffer[0] == "]":
                        return
                    try:
                        item, index = decoder.raw_decode(buffer)
                    except json.JSONDecodeError:
                        break
                    buffer = buffer[index:]
                    instance = _normalize_instance(item)
                    if instance is not None:
                        yield instance
            return

        for line in file:
            text = line.strip()
            if not text:
                continue
            item = json.loads(text)
            instance = _normalize_instance(item)
            if instance is not None:
                yield instance


def iter_history_messages(instance: EvalInstance) -> Iterable[Turn]:
    """Yield history turns in dataset order (session order, then turn order)."""
    sessions = instance.get("haystack_sessions", [])
    session_ids = list(instance.get("haystack_session_ids", []))
    session_dates = list(instance.get("haystack_dates", []))
    if not isinstance(sessions, list):
        return
    for session_index, session in enumerate(sessions):
        sid = str(session_ids[session_index]) if session_index < len(session_ids) else ""
        sdate = str(session_dates[session_index]) if session_index < len(session_dates) else ""
        if not isinstance(session, list):
            continue
        for turn_index, turn in enumerate(session):
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role", "user"))
            content = str(turn.get("content", "")).strip()
            if not content:
                continue
            out: Turn = {
                "role": role,
                "content": content,
                "session_id": sid,
                "session_date": sdate,
                "turn_index": int(turn_index),
            }
            if bool(turn.get("has_answer", False)):
                out["has_answer"] = True
            yield out
