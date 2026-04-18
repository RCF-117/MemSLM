"""Streaming loader for LongMemEval and LoCoMo evaluation instances.

The public interface is intentionally stable:
- ``load_stream(path)`` yields normalized eval instances
- ``iter_history_messages(instance)`` yields normalized history turns

For LongMemEval:
- one raw item -> one eval instance

For LoCoMo:
- one raw sample -> multiple eval instances (one per QA pair)
"""

from __future__ import annotations

import json
import re
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


def _is_locomo_sample(item: Any) -> bool:
    """Return True if item matches LoCoMo sample shape."""
    return (
        isinstance(item, dict)
        and isinstance(item.get("conversation"), dict)
        and isinstance(item.get("qa"), list)
    )


def _session_index_from_dia_id(dia_id: str) -> int | None:
    """Extract session index from dia id like 'D12:3'."""
    match = re.fullmatch(r"[Dd](\d+):\d+", str(dia_id).strip())
    if not match:
        return None
    return int(match.group(1))


def _normalize_locomo_instances(
    item: Dict[str, Any],
    *,
    drop_empty_answers: bool,
    drop_categories: set[str],
    max_qas_per_sample: int,
) -> List[EvalInstance]:
    """Normalize one LoCoMo sample into LongMemEval-like eval instances."""
    conversation = item.get("conversation", {})
    if not isinstance(conversation, dict):
        return []

    speaker_a = str(conversation.get("speaker_a", "")).strip()
    speaker_b = str(conversation.get("speaker_b", "")).strip()

    session_keys = [
        key
        for key in conversation.keys()
        if re.fullmatch(r"session_\d+", str(key))
    ]
    session_keys.sort(key=lambda key: int(str(key).split("_")[1]))

    session_ids: List[str] = []
    session_dates: List[str] = []
    sessions: List[Session] = []
    dia_to_session_id: Dict[str, str] = {}

    for session_key in session_keys:
        raw_session = conversation.get(session_key)
        if not isinstance(raw_session, list):
            continue

        normalized_session: Session = []
        sid = str(session_key)
        sdate = str(conversation.get(f"{session_key}_date_time", "")).strip()

        for turn in raw_session:
            if not isinstance(turn, dict):
                continue
            text = str(turn.get("text", "")).strip()
            if not text:
                continue
            speaker = str(turn.get("speaker", "")).strip()
            role = "assistant"
            if speaker and speaker_a and speaker == speaker_a:
                role = "user"
            elif speaker and speaker_b and speaker == speaker_b:
                role = "assistant"
            elif str(turn.get("role", "")).strip().lower() in {"user", "assistant", "system"}:
                role = str(turn.get("role")).strip().lower()

            normalized: Turn = {"role": role, "content": text}
            dia_id = str(turn.get("dia_id", "")).strip()
            if dia_id:
                normalized["dia_id"] = dia_id
                dia_to_session_id[dia_id] = sid
            normalized_session.append(normalized)

        if normalized_session:
            session_ids.append(sid)
            session_dates.append(sdate)
            sessions.append(normalized_session)

    if not sessions:
        return []

    qa_list = item.get("qa", [])
    if not isinstance(qa_list, list):
        return []

    sample_id = str(item.get("sample_id", "")).strip()
    out: List[EvalInstance] = []
    for index, qa in enumerate(qa_list):
        if not isinstance(qa, dict):
            continue
        question = str(qa.get("question", "")).strip()
        answer = str(qa.get("answer", "")).strip()
        if not question:
            continue
        category = str(qa.get("category", "")).strip()
        if category and (category in drop_categories):
            continue
        if drop_empty_answers and (not answer):
            continue

        evidence = qa.get("evidence", [])
        evidence_ids = {str(x).strip() for x in evidence if str(x).strip()}

        tagged_sessions: List[Session] = []
        for session in sessions:
            tagged: Session = []
            for turn in session:
                copied = dict(turn)
                if str(copied.get("dia_id", "")).strip() in evidence_ids:
                    copied["has_answer"] = True
                tagged.append(copied)
            tagged_sessions.append(tagged)

        answer_session_ids = sorted(
            {
                dia_to_session_id[eid]
                for eid in evidence_ids
                if eid in dia_to_session_id
            }
        )

        qid = str(qa.get("question_id", "")).strip()
        if not qid:
            qid = f"{sample_id}_qa_{index}"
        qtype = f"locomo_category_{category}" if category else "locomo"
        question_date = str(qa.get("question_date", "")).strip()

        out.append(
            {
                "question_id": qid,
                "question_type": qtype,
                "question": question,
                "answer": answer,
                "question_date": question_date,
                "haystack_session_ids": list(session_ids),
                "haystack_dates": list(session_dates),
                "haystack_sessions": tagged_sessions,
                "answer_session_ids": answer_session_ids,
            }
        )
        if max_qas_per_sample > 0 and len(out) >= max_qas_per_sample:
            break
    return out


def load_stream(path: str) -> Generator[EvalInstance, None, None]:
    """Yield one normalized eval instance at a time.

    Supports:
    - LongMemEval JSON / JSONL
    - LoCoMo JSON (one sample expanded into multiple QA instances)
    """
    cfg = load_config()
    stream_read_size = int(cfg["dataset"]["stream_read_size"])
    eval_cfg = dict(cfg.get("evaluation", {}))
    locomo_cfg = dict(eval_cfg.get("locomo", {}))
    drop_empty_answers = bool(locomo_cfg.get("drop_empty_answers", True))
    drop_categories = {
        str(x).strip()
        for x in list(locomo_cfg.get("drop_categories", []))
        if str(x).strip()
    }
    max_qas_per_sample = int(locomo_cfg.get("max_qas_per_sample", 0))
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
                    if _is_locomo_sample(item):
                        for locomo_instance in _normalize_locomo_instances(
                            item,
                            drop_empty_answers=drop_empty_answers,
                            drop_categories=drop_categories,
                            max_qas_per_sample=max_qas_per_sample,
                        ):
                            yield locomo_instance
                    else:
                        instance = _normalize_instance(item)
                        if instance is not None:
                            yield instance
            return

        for line in file:
            text = line.strip()
            if not text:
                continue
            item = json.loads(text)
            if _is_locomo_sample(item):
                for locomo_instance in _normalize_locomo_instances(
                    item,
                    drop_empty_answers=drop_empty_answers,
                    drop_categories=drop_categories,
                    max_qas_per_sample=max_qas_per_sample,
                ):
                    yield locomo_instance
            else:
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
