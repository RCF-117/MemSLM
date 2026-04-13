"""Event-node graph construction helper for long memory."""

from __future__ import annotations

from typing import Callable, Dict, List


def upsert_event_nodes(
    *,
    store: object,
    event_id: str,
    subject: str,
    action: str,
    obj: str,
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
    store.clear_event_node_edges_for_event(event_id)
    node_ids: Dict[str, str] = {}
    keyword_node_ids: List[str] = []

    def add_node(kind: str, text: str, is_core: bool) -> str:
        val = normalize_space_fn(text)
        if not val:
            return ""
        node_id = stable_id_fn("node", f"{event_id}|{kind}|{val}")
        store.upsert_event_node(
            node_id=node_id,
            event_id=event_id,
            node_kind=kind,
            node_text=val,
            is_core=is_core,
            node_embedding=safe_embed_fn(val),
            current_step=current_step,
        )
        return node_id

    node_ids["subject"] = add_node("subject", subject, True)
    node_ids["action"] = add_node("action", action, True)
    node_ids["object"] = add_node("object", obj, True)
    node_ids["time"] = add_node("time", time_text, True)
    node_ids["location"] = add_node("location", location_text, True)
    node_ids["evidence"] = add_node("evidence", raw_span[:context_max_chars_per_item], False)
    for kw in keywords[: max(0, int(node_keyword_limit))]:
        kid = add_node("keyword", kw, False)
        if kid:
            keyword_node_ids.append(kid)

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

    add_node_edge("subject", "action", "agent_of", 1.0)
    add_node_edge("action", "object", "acts_on", 1.0)
    add_node_edge("action", "time", "happens_at", 0.9)
    add_node_edge("action", "location", "happens_in", 0.9)
    add_node_edge("action", "evidence", "grounded_by", 0.8)

    action_node = str(node_ids.get("action", "")).strip()
    if action_node:
        for kid in keyword_node_ids:
            edge_id = stable_id_fn("node_edge", f"{event_id}|{action_node}|{kid}|has_keyword")
            store.upsert_event_node_edge(
                node_edge_id=edge_id,
                event_id=event_id,
                from_node_id=action_node,
                to_node_id=kid,
                relation="has_keyword",
                weight=0.5,
                current_step=current_step,
            )

