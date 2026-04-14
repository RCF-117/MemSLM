"""Event-node graph construction helper for long memory."""

from __future__ import annotations

from typing import Callable, Dict, List


def upsert_event_nodes(
    *,
    store: object,
    event_id: str,
    subject: str,
    predicate: str,
    value: str,
    time_text: str,
    location_text: str,
    keywords: List[str],
    raw_span: str,
    context_max_chars_per_item: int,
    node_keyword_limit: int,
    current_step: int,
    normalize_space_fn: Callable[[str], str],
    stable_id_fn: Callable[[str, str], str],
    safe_embed_fn: Callable[[str], object],
) -> None:
    """Write a minimal fact-centric node graph for one event."""
    store.clear_event_node_edges_for_event(event_id)
    node_ids: Dict[str, str] = {}
    keyword_node_ids: List[str] = []

    def add_node(kind: str, text: str, is_core: bool) -> str:
        value_text = normalize_space_fn(text)
        if not value_text:
            return ""
        node_id = stable_id_fn("node", f"{event_id}|{kind}|{value_text}")
        store.upsert_event_node(
            node_id=node_id,
            event_id=event_id,
            node_kind=kind,
            node_text=value_text,
            is_core=is_core,
            node_embedding=safe_embed_fn(value_text),
            current_step=current_step,
        )
        return node_id

    node_ids["subject"] = add_node("subject", subject, True)
    node_ids["predicate"] = add_node("predicate", predicate, True)
    node_ids["value"] = add_node("value", value, True)
    node_ids["time"] = add_node("time", time_text, False)
    node_ids["location"] = add_node("location", location_text, False)
    node_ids["evidence"] = add_node("evidence", raw_span[:context_max_chars_per_item], False)

    for keyword in keywords[: max(0, int(node_keyword_limit))]:
        keyword_id = add_node("keyword", keyword, False)
        if keyword_id:
            keyword_node_ids.append(keyword_id)

    def add_node_edge(src_kind: str, dst_kind: str, relation: str, weight: float) -> None:
        src = str(node_ids.get(src_kind, "")).strip()
        dst = str(node_ids.get(dst_kind, "")).strip()
        if (not src) or (not dst):
            return
        edge_id = stable_id_fn("node_edge", f"{event_id}|{src}|{dst}|{relation}")
        store.upsert_event_node_edge(
            node_edge_id=edge_id,
            event_id=event_id,
            from_node_id=src,
            to_node_id=dst,
            relation=relation,
            weight=weight,
            current_step=current_step,
        )

    add_node_edge("predicate", "value", "has_value", 1.0)
    add_node_edge("predicate", "time", "has_time", 0.9)
    add_node_edge("predicate", "location", "has_location", 0.9)
    add_node_edge("predicate", "evidence", "supported_by", 0.8)

    predicate_node = str(node_ids.get("predicate", "")).strip()
    if predicate_node:
        for keyword_id in keyword_node_ids:
            edge_id = stable_id_fn("node_edge", f"{event_id}|{predicate_node}|{keyword_id}|has_keyword")
            store.upsert_event_node_edge(
                node_edge_id=edge_id,
                event_id=event_id,
                from_node_id=predicate_node,
                to_node_id=keyword_id,
                relation="has_keyword",
                weight=0.4,
                current_step=current_step,
            )
