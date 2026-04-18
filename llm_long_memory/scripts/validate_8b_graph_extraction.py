"""Standalone probe: can 8B extract a human-readable graph from raw dialogs?

This script intentionally bypasses the project long-memory graph pipeline.
It sends the first N ragdebug instances to a single extraction prompt and
evaluates whether the returned graph is grounded and non-noisy.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from llm_long_memory.llm.ollama_client import ollama_generate_with_retry
    from llm_long_memory.utils.helpers import load_config, resolve_project_path
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from llm_long_memory.llm.ollama_client import ollama_generate_with_retry
    from llm_long_memory.utils.helpers import load_config, resolve_project_path


@dataclass
class ProbeConfig:
    model: str
    host: str
    temperature: float
    timeout_sec: int
    max_attempts: int
    backoff_sec: float
    retry_on_timeout: bool
    retry_on_http_502: bool
    retry_on_url_error: bool
    max_output_tokens: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate 8B graph extraction quality.")
    parser.add_argument(
        "--config",
        default="llm_long_memory/config/config.yaml",
        help="Config path.",
    )
    parser.add_argument(
        "--dataset",
        default="",
        help="Dataset path. Default uses dataset.eval_splits.ragdebug10.",
    )
    parser.add_argument(
        "--model",
        default="qwen3:8b",
        help="Extractor model.",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=2,
        help="How many instances from dataset head to probe.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/thesis_reports_debug_analysis",
        help="Output directory for probe artifacts.",
    )
    return parser.parse_args()


def _extract_first_json_block(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return "{}"
    if value.startswith("{") and value.endswith("}"):
        return value
    start = value.find("{")
    if start < 0:
        return "{}"
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(value)):
        ch = value[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return value[start : idx + 1]
    return "{}"


def _safe_json(text: str) -> Dict[str, Any]:
    block = _extract_first_json_block(text)
    try:
        obj = json.loads(block)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {"entities": [], "events": [], "relations": []}
    if not isinstance(obj, dict):
        return {"entities": [], "events": [], "relations": []}
    return obj


def _unwrap_graph_payload(obj: Dict[str, Any]) -> Dict[str, Any]:
    if any(k in obj for k in ("entities", "events", "relations")):
        return obj
    maybe_response = obj.get("response")
    if isinstance(maybe_response, str) and maybe_response.strip():
        nested = _safe_json(maybe_response)
        if any(k in nested for k in ("entities", "events", "relations")):
            return nested
    maybe_data = obj.get("data")
    if isinstance(maybe_data, dict) and any(k in maybe_data for k in ("entities", "events", "relations")):
        return maybe_data
    return {"entities": [], "events": [], "relations": []}


def _normalize_items(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, dict):
        return [raw]
    if not isinstance(raw, list):
        return []
    return [x for x in raw if isinstance(x, dict)]


def normalize_graph(obj: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    entities = _normalize_items(obj.get("entities"))
    events = _normalize_items(obj.get("events"))
    relations = _normalize_items(obj.get("relations"))
    ent_out: List[Dict[str, Any]] = []
    for i, item in enumerate(entities, start=1):
        ent_out.append(
            {
                "id": str(item.get("id", f"E{i}")).strip() or f"E{i}",
                "name": str(item.get("name", "")).strip(),
                "type": str(item.get("type", "")).strip().lower(),
                "evidence_spans": [
                    str(x).strip()
                    for x in list(item.get("evidence_spans", []))
                    if str(x).strip()
                ],
            }
        )
    evt_out: List[Dict[str, Any]] = []
    for i, item in enumerate(events, start=1):
        evt_out.append(
            {
                "id": str(item.get("id", f"V{i}")).strip() or f"V{i}",
                "summary": str(item.get("summary", "")).strip(),
                "time": str(item.get("time", "")).strip(),
                "location": str(item.get("location", "")).strip(),
                "participants": [
                    str(x).strip() for x in list(item.get("participants", [])) if str(x).strip()
                ],
                "evidence_spans": [
                    str(x).strip()
                    for x in list(item.get("evidence_spans", []))
                    if str(x).strip()
                ],
            }
        )
    rel_out: List[Dict[str, Any]] = []
    for i, item in enumerate(relations, start=1):
        rel_out.append(
            {
                "id": str(item.get("id", f"R{i}")).strip() or f"R{i}",
                "source_id": str(item.get("source_id", "")).strip(),
                "target_id": str(item.get("target_id", "")).strip(),
                "type": str(item.get("type", "")).strip().lower(),
                "evidence_span": str(item.get("evidence_span", "")).strip(),
            }
        )
    return {"entities": ent_out, "events": evt_out, "relations": rel_out}


def flatten_session_messages(instance: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    sessions = instance.get("haystack_sessions") or []
    for s_idx, session in enumerate(sessions, start=1):
        if not isinstance(session, list):
            continue
        for t_idx, msg in enumerate(session, start=1):
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "")).strip().lower() or "user"
            content = str(msg.get("content", "")).strip()
            if not content:
                continue
            out.append(
                {
                    "session_idx": s_idx,
                    "turn_idx": t_idx,
                    "role": role,
                    "content": content,
                    "has_answer": bool(msg.get("has_answer", False)),
                }
            )
    return out


def render_dialog(messages: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for msg in messages:
        sid = int(msg["session_idx"])
        tid = int(msg["turn_idx"])
        role = str(msg["role"]).upper()
        content = str(msg["content"]).strip()
        lines.append(f"[S{sid:02d} T{tid:02d} {role}] {content}")
    return "\n".join(lines)


def build_probe_prompt(question: str, dialog_text: str) -> str:
    return (
        "You are a graph extraction engine for memory reasoning.\n"
        "Extract a grounded, human-readable knowledge graph from the dialogue.\n"
        "Return JSON only.\n"
        'Schema: {"entities":[{"id":"","name":"","type":"","evidence_spans":[]}],'
        '"events":[{"id":"","summary":"","time":"","location":"","participants":[],"evidence_spans":[]}],'
        '"relations":[{"id":"","source_id":"","target_id":"","type":"","evidence_span":""}]}\n'
        "Rules:\n"
        "- Use evidence_spans as exact quotes from dialogue, no paraphrase.\n"
        "- Prefer user facts and objective updates; avoid generic assistant advice.\n"
        "- No meta nodes like 'user', 'assistant', 'the conversation'.\n"
        "- Keep relations factual: updates, before, after, owns, located_in, prefers, related_to.\n"
        "- If uncertain, output fewer but high-precision nodes.\n"
        f"Question: {question}\n"
        "Dialogue:\n"
        f"{dialog_text}\n"
    )


def build_retry_prompt(question: str, dialog_text: str) -> str:
    return (
        "Return JSON only. Do NOT answer as assistant. Do NOT include explanation text.\n"
        'Required schema: {"entities":[{"id":"","name":"","type":"","evidence_spans":[]}],'
        '"events":[{"id":"","summary":"","time":"","location":"","participants":[],"evidence_spans":[]}],'
        '"relations":[{"id":"","source_id":"","target_id":"","type":"","evidence_span":""}]}\n'
        "Constraints:\n"
        "- At least 3 entities, 2 events, 2 relations when evidence exists.\n"
        "- Every entity/event/relation must include evidence from dialogue.\n"
        "- Exact quotes only in evidence spans.\n"
        "- Exclude generic assistant suggestions and meta statements.\n"
        f"Question: {question}\n"
        "Evidence dialogue:\n"
        f"{dialog_text}\n"
    )


def run_extract(prompt: str, cfg: ProbeConfig) -> str:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return ollama_generate_with_retry(
        host=cfg.host,
        model=cfg.model,
        prompt=prompt,
        temperature=cfg.temperature,
        timeout_sec=cfg.timeout_sec,
        opener=opener,
        max_attempts=cfg.max_attempts,
        backoff_sec=cfg.backoff_sec,
        retry_on_timeout=cfg.retry_on_timeout,
        retry_on_http_502=cfg.retry_on_http_502,
        retry_on_url_error=cfg.retry_on_url_error,
        max_output_tokens=cfg.max_output_tokens,
        think=False,
        response_format="json",
    )


def build_fact_pack(messages: List[Dict[str, Any]], max_items: int = 24) -> List[Dict[str, Any]]:
    if not messages:
        return []
    keep: List[Dict[str, Any]] = []
    selected_idx = set()
    for idx, msg in enumerate(messages):
        if bool(msg.get("has_answer", False)):
            for j in (idx - 1, idx, idx + 1):
                if 0 <= j < len(messages):
                    selected_idx.add(j)
    if not selected_idx:
        for idx, msg in enumerate(messages):
            if str(msg.get("role", "")).lower() == "user":
                selected_idx.add(idx)
    for idx in sorted(selected_idx):
        keep.append(messages[idx])
        if len(keep) >= max_items:
            break
    return keep


def graph_is_empty(graph: Dict[str, List[Dict[str, Any]]]) -> bool:
    return (
        len(graph.get("entities", [])) == 0
        and len(graph.get("events", [])) == 0
        and len(graph.get("relations", [])) == 0
    )


def _span_hit_ratio(spans: List[str], corpus: str) -> float:
    if not spans:
        return 0.0
    hit = 0
    corpus_low = corpus.lower()
    for span in spans:
        s = str(span).strip()
        if not s:
            continue
        if s.lower() in corpus_low:
            hit += 1
    return float(hit) / float(max(1, len(spans)))


def _is_meta_text(text: str) -> bool:
    low = str(text or "").strip().lower()
    if not low:
        return True
    meta_patterns = [
        r"^user\b",
        r"^assistant\b",
        r"\bi can help\b",
        r"\bi suggest\b",
        r"\bgeneral advice\b",
        r"\bthe conversation\b",
        r"\bthis chat\b",
    ]
    return any(re.search(pat, low) for pat in meta_patterns)


def evaluate_graph(graph: Dict[str, List[Dict[str, Any]]], corpus: str) -> Dict[str, Any]:
    entities = graph["entities"]
    events = graph["events"]
    relations = graph["relations"]

    entity_ev = [s for e in entities for s in e.get("evidence_spans", [])]
    event_ev = [s for e in events for s in e.get("evidence_spans", [])]
    rel_ev = [str(r.get("evidence_span", "")).strip() for r in relations if str(r.get("evidence_span", "")).strip()]

    entity_ground = _span_hit_ratio(entity_ev, corpus)
    event_ground = _span_hit_ratio(event_ev, corpus)
    relation_ground = _span_hit_ratio(rel_ev, corpus)

    meta_entity_ratio = (
        float(sum(1 for e in entities if _is_meta_text(e.get("name", ""))))
        / float(max(1, len(entities)))
    )
    meta_event_ratio = (
        float(sum(1 for e in events if _is_meta_text(e.get("summary", ""))))
        / float(max(1, len(events)))
    )

    node_ids = {str(e["id"]) for e in entities} | {str(v["id"]) for v in events}
    degree = {nid: 0 for nid in node_ids}
    for rel in relations:
        sid = str(rel.get("source_id", "")).strip()
        tid = str(rel.get("target_id", "")).strip()
        if sid in degree:
            degree[sid] += 1
        if tid in degree:
            degree[tid] += 1
    for evt in events:
        vid = str(evt.get("id", "")).strip()
        for pid in evt.get("participants", []):
            p = str(pid).strip()
            if vid in degree:
                degree[vid] += 1
            if p in degree:
                degree[p] += 1
    isolated = sum(1 for d in degree.values() if d == 0)

    quality_score = (
        0.35 * entity_ground
        + 0.30 * event_ground
        + 0.20 * relation_ground
        + 0.10 * (1.0 - meta_entity_ratio)
        + 0.05 * (1.0 - meta_event_ratio)
    )
    return {
        "entity_count": len(entities),
        "event_count": len(events),
        "relation_count": len(relations),
        "entity_grounding_ratio": round(entity_ground, 4),
        "event_grounding_ratio": round(event_ground, 4),
        "relation_grounding_ratio": round(relation_ground, 4),
        "meta_entity_ratio": round(meta_entity_ratio, 4),
        "meta_event_ratio": round(meta_event_ratio, 4),
        "isolated_node_ratio": round(float(isolated) / float(max(1, len(node_ids))), 4),
        "overall_quality_score": round(quality_score, 4),
    }


def to_visual_graph(graph: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    for entity in graph["entities"]:
        nodes.append(
            {
                "id": entity["id"],
                "label": entity.get("name", ""),
                "type": f"entity:{entity.get('type', '') or 'unknown'}",
            }
        )
    for event in graph["events"]:
        nodes.append(
            {
                "id": event["id"],
                "label": event.get("summary", ""),
                "type": "event",
            }
        )
        for pid in event.get("participants", []):
            p = str(pid).strip()
            if p:
                edges.append(
                    {
                        "source": event["id"],
                        "target": p,
                        "label": "involves",
                    }
                )
    for rel in graph["relations"]:
        edges.append(
            {
                "source": rel.get("source_id", ""),
                "target": rel.get("target_id", ""),
                "label": rel.get("type", ""),
            }
        )
    return {"nodes": nodes, "edges": edges}


def probe_one(instance: Dict[str, Any], cfg: ProbeConfig) -> Dict[str, Any]:
    question = str(instance.get("question", "")).strip()
    qid = str(instance.get("question_id", "")).strip()
    qtype = str(instance.get("question_type", "")).strip()
    messages = flatten_session_messages(instance)
    dialog_text = render_dialog(messages)
    prompt = build_probe_prompt(question=question, dialog_text=dialog_text)
    raw = run_extract(prompt=prompt, cfg=cfg)
    parsed = _unwrap_graph_payload(_safe_json(raw))
    graph = normalize_graph(parsed)
    used_retry = False
    if graph_is_empty(graph):
        used_retry = True
        pack_messages = build_fact_pack(messages, max_items=24)
        pack_text = render_dialog(pack_messages)
        retry_prompt = build_retry_prompt(question=question, dialog_text=pack_text)
        raw_retry = run_extract(prompt=retry_prompt, cfg=cfg)
        parsed_retry = _unwrap_graph_payload(_safe_json(raw_retry))
        graph_retry = normalize_graph(parsed_retry)
        if not graph_is_empty(graph_retry):
            raw = raw_retry
            graph = graph_retry
    metrics = evaluate_graph(graph, corpus=dialog_text)
    visual = to_visual_graph(graph)
    return {
        "question_id": qid,
        "question_type": qtype,
        "question": question,
        "answer": str(instance.get("answer", "")).strip(),
        "message_count": len(messages),
        "metrics": metrics,
        "graph": graph,
        "visual_graph": visual,
        "raw_model_output": raw,
        "used_retry": used_retry,
    }


def build_markdown(run_id: str, model: str, dataset_name: str, cases: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append(f"# 8B Graph Extraction Probe")
    lines.append("")
    lines.append(f"- run_id: `{run_id}`")
    lines.append(f"- model: `{model}`")
    lines.append(f"- dataset: `{dataset_name}`")
    lines.append(f"- instances: `{len(cases)}`")
    lines.append("")
    if cases:
        avg_quality = sum(float(c["metrics"]["overall_quality_score"]) for c in cases) / float(len(cases))
        avg_meta = sum(float(c["metrics"]["meta_event_ratio"]) for c in cases) / float(len(cases))
        lines.append("## Aggregate")
        lines.append("")
        lines.append(f"- avg_overall_quality_score: `{avg_quality:.4f}`")
        lines.append(f"- avg_meta_event_ratio: `{avg_meta:.4f}`")
        lines.append("")
    lines.append("## Per Case")
    lines.append("")
    for idx, case in enumerate(cases, start=1):
        m = case["metrics"]
        lines.append(f"### Case {idx}: `{case['question_id']}` ({case['question_type']})")
        lines.append(f"- question: {case['question']}")
        lines.append(f"- quality_score: `{m['overall_quality_score']}`")
        lines.append(
            "- counts: "
            f"entity={m['entity_count']}, event={m['event_count']}, relation={m['relation_count']}"
        )
        lines.append(
            "- grounding: "
            f"entity={m['entity_grounding_ratio']}, event={m['event_grounding_ratio']}, relation={m['relation_grounding_ratio']}"
        )
        lines.append(
            "- noise: "
            f"meta_entity={m['meta_entity_ratio']}, meta_event={m['meta_event_ratio']}, isolated={m['isolated_node_ratio']}"
        )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def build_probe_config(config: Dict[str, Any], model: str) -> ProbeConfig:
    llm_cfg = dict(config["llm"])
    retry_cfg = dict(llm_cfg.get("retry", {}))
    return ProbeConfig(
        model=model,
        host=str(llm_cfg["host"]).rstrip("/"),
        temperature=float(llm_cfg.get("temperature", 0.0)),
        timeout_sec=int(llm_cfg.get("request_timeout_sec", 120)),
        max_attempts=int(retry_cfg.get("max_attempts", 1)),
        backoff_sec=float(retry_cfg.get("backoff_sec", 0.0)),
        retry_on_timeout=bool(retry_cfg.get("retry_on_timeout", True)),
        retry_on_http_502=bool(retry_cfg.get("retry_on_http_502", True)),
        retry_on_url_error=bool(retry_cfg.get("retry_on_url_error", False)),
        max_output_tokens=384,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    split_map = dict(config["dataset"].get("eval_splits", {}))
    dataset_path = (
        resolve_project_path(args.dataset)
        if str(args.dataset).strip()
        else resolve_project_path(str(split_map.get("ragdebug10", "")))
    )
    if (not dataset_path) or (not Path(dataset_path).exists()):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    data = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Dataset must be a non-empty list.")
    head_n = max(1, int(args.max_instances))
    samples = data[:head_n]
    cfg = build_probe_config(config=config, model=str(args.model).strip() or "qwen3:8b")
    run_id = datetime.now().strftime("graph_probe_%Y%m%d_%H%M%S")
    out_dir = resolve_project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for idx, instance in enumerate(samples, start=1):
        qid = str(instance.get("question_id", "")).strip()
        print(f"[probe {idx}/{len(samples)}] question_id={qid}")
        results.append(probe_one(instance=instance, cfg=cfg))

    dataset_name = Path(dataset_path).name
    payload = {
        "run_id": run_id,
        "model": cfg.model,
        "dataset": dataset_name,
        "instance_count": len(results),
        "cases": results,
    }
    json_path = out_dir / f"{run_id}__{dataset_name}__model-{cfg.model.replace(':', '_')}.json"
    md_path = out_dir / f"{run_id}__{dataset_name}__model-{cfg.model.replace(':', '_')}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown(run_id, cfg.model, dataset_name, results), encoding="utf-8")

    print(f"probe_json: {json_path}")
    print(f"probe_md: {md_path}")


if __name__ == "__main__":
    main()
