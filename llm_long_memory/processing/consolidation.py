"""Graph consolidation utilities for long-memory maintenance."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import networkx as nx


def _node_signature(attrs: Dict[str, Any]) -> Tuple[str, str]:
    node_type = str(attrs.get("type", "unknown")).strip().lower()
    text = str(attrs.get("text", "")).strip().lower()
    return node_type, text


def merge_duplicate_nodes(graph: nx.DiGraph, max_evidence_per_edge: int) -> int:
    """Merge duplicate nodes that share the same (type, normalized text)."""
    by_sig: Dict[Tuple[str, str], str] = {}
    merged = 0

    for node_id, attrs in list(graph.nodes(data=True)):
        sig = _node_signature(attrs)
        canonical = by_sig.get(sig)
        if canonical is None:
            by_sig[sig] = str(node_id)
            continue
        if canonical == node_id:
            continue

        for src, _, edge_attrs in list(graph.in_edges(node_id, data=True)):
            rel = str(edge_attrs.get("relation", "related_to"))
            _merge_edge(graph, str(src), str(canonical), rel, edge_attrs, max_evidence_per_edge)
        for _, dst, edge_attrs in list(graph.out_edges(node_id, data=True)):
            rel = str(edge_attrs.get("relation", "related_to"))
            _merge_edge(graph, str(canonical), str(dst), rel, edge_attrs, max_evidence_per_edge)

        graph.remove_node(node_id)
        merged += 1

    return merged


def _merge_edge(
    graph: nx.DiGraph,
    src: str,
    dst: str,
    relation: str,
    attrs: Dict[str, Any],
    max_evidence_per_edge: int,
) -> None:
    if graph.has_edge(src, dst):
        existing = graph[src][dst]
        existing_rel = str(existing.get("relation", "related_to"))
        if existing_rel != relation:
            relation = existing_rel
        old_weight = float(existing.get("weight", 1.0))
        new_weight = float(attrs.get("weight", 1.0))
        existing["weight"] = old_weight + new_weight
        evidence = list(existing.get("evidence", [])) + list(attrs.get("evidence", []))
        existing["evidence"] = evidence[-max_evidence_per_edge:]
    else:
        graph.add_edge(
            src,
            dst,
            relation=relation,
            weight=float(attrs.get("weight", 1.0)),
            evidence=list(attrs.get("evidence", []))[-max_evidence_per_edge:],
        )


def prune_isolated_event_nodes(graph: nx.DiGraph) -> int:
    """Drop event nodes that became isolated after merges."""
    to_remove = []
    for node_id, attrs in graph.nodes(data=True):
        node_type = str(attrs.get("type", "")).strip().lower()
        if node_type != "event":
            continue
        if graph.degree(node_id) == 0:
            to_remove.append(node_id)
    for node_id in to_remove:
        graph.remove_node(node_id)
    return len(to_remove)
