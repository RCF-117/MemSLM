"""Export long-memory graph artifacts and an HTML visualization preview."""

from __future__ import annotations

import argparse
import html
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import networkx as nx

from llm_long_memory.utils.helpers import resolve_project_path


def _safe_text(value: Any, limit: int = 180) -> str:
    text = " ".join(str(value or "").split()).strip()
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def _load_rows(conn: sqlite3.Connection, sql: str, params: Sequence[Any] = ()) -> List[sqlite3.Row]:
    return conn.execute(sql, tuple(params)).fetchall()


def _build_event_graph(
    event_rows: Sequence[sqlite3.Row],
    edge_rows: Sequence[sqlite3.Row],
) -> nx.DiGraph:
    graph = nx.DiGraph()
    for row in event_rows:
        event_id = str(row["event_id"])
        keywords = row["keywords"] or "[]"
        try:
            keyword_list = json.loads(keywords)
        except Exception:
            keyword_list = []
        graph.add_node(
            event_id,
            label=_safe_text(row["skeleton_text"], 80),
            title=_safe_text(row["skeleton_text"], 400),
            fact_key=str(row["fact_key"] or ""),
            fact_type=str(row["fact_type"] or ""),
            role=str(row["role"] or ""),
            status=str(row["status"] or ""),
            is_latest=int(row["is_latest"] or 0),
            salience=float(row["salience"] or 0.0),
            last_seen_step=int(row["last_seen_step"] or 0),
            extract_confidence=float(row["extract_confidence"] or 0.0),
            raw_span=_safe_text(row["raw_span"], 220),
            keywords=",".join(str(x) for x in keyword_list[:12]),
        )
    for row in edge_rows:
        from_event = str(row["from_event_id"] or "")
        to_event = str(row["to_event_id"] or "")
        if (not from_event) or (not to_event):
            continue
        if from_event not in graph or to_event not in graph:
            continue
        graph.add_edge(
            from_event,
            to_event,
            label=str(row["relation"] or ""),
            weight=float(row["weight"] or 0.0),
            title=f"{row['relation']} ({row['weight']})",
            created_step=int(row["created_step"] or 0),
        )
    return graph


def _build_node_graph(
    node_rows: Sequence[sqlite3.Row],
    edge_rows: Sequence[sqlite3.Row],
) -> nx.DiGraph:
    graph = nx.DiGraph()
    for row in node_rows:
        node_id = str(row["node_id"])
        graph.add_node(
            node_id,
            label=_safe_text(row["node_text"], 60),
            title=_safe_text(row["node_text"], 260),
            event_id=str(row["event_id"] or ""),
            node_kind=str(row["node_kind"] or ""),
            is_core=int(row["is_core"] or 0),
            created_step=int(row["created_step"] or 0),
        )
    for row in edge_rows:
        from_node = str(row["from_node_id"] or "")
        to_node = str(row["to_node_id"] or "")
        if (not from_node) or (not to_node):
            continue
        if from_node not in graph or to_node not in graph:
            continue
        graph.add_edge(
            from_node,
            to_node,
            label=str(row["relation"] or ""),
            weight=float(row["weight"] or 0.0),
            created_step=int(row["created_step"] or 0),
        )
    return graph


def _graph_to_json(graph: nx.DiGraph) -> Dict[str, Any]:
    nodes = []
    for node_id, attrs in graph.nodes(data=True):
        payload = {"id": node_id}
        payload.update(attrs)
        nodes.append(payload)
    edges = []
    for src, dst, attrs in graph.edges(data=True):
        payload = {"from": src, "to": dst}
        payload.update(attrs)
        edges.append(payload)
    return {"nodes": nodes, "edges": edges}


def _write_html_preview(
    *,
    out_path: Path,
    title: str,
    graph_json: Dict[str, Any],
    note: str,
) -> None:
    nodes_json = json.dumps(graph_json["nodes"], ensure_ascii=False).replace("</", "<\\/")
    edges_json = json.dumps(graph_json["edges"], ensure_ascii=False).replace("</", "<\\/")
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #0f172a; color: #e2e8f0; }}
    header {{ padding: 16px 20px; background: #111827; border-bottom: 1px solid #334155; }}
    header h1 {{ margin: 0 0 6px 0; font-size: 18px; }}
    header p {{ margin: 0; color: #94a3b8; font-size: 13px; }}
    #network {{ width: 100vw; height: calc(100vh - 94px); }}
    .legend {{ padding: 10px 20px 0; color: #cbd5e1; font-size: 12px; }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <p>{html.escape(note)}</p>
  </header>
  <div class="legend">Blue: latest / active facts. Hover nodes for details. Open GraphML in Gephi or Cytoscape for a richer view.</div>
  <div id="network"></div>
  <script>
    const nodes = new vis.DataSet({nodes_json});
    const edges = new vis.DataSet({edges_json});
    const container = document.getElementById('network');
    const data = {{ nodes, edges }};
    const options = {{
      nodes: {{
        shape: 'dot',
        size: 14,
        font: {{ color: '#e2e8f0', size: 12 }},
        borderWidth: 1
      }},
      edges: {{
        arrows: 'to',
        color: {{ color: '#64748b', highlight: '#93c5fd' }},
        font: {{ align: 'middle', color: '#cbd5e1', size: 10 }}
      }},
      physics: {{
        stabilization: true,
        barnesHut: {{
          gravitationalConstant: -2200,
          springLength: 140,
          springConstant: 0.04
        }}
      }},
      interaction: {{ hover: true, navigationButtons: true, keyboard: true }}
    }};
    new vis.Network(container, data, options);
  </script>
</body>
</html>
"""
    out_path.write_text(html_text, encoding="utf-8")


def export_graph(
    *,
    db_path: str,
    output_dir: str,
    active_only: bool = False,
    event_limit: int = 0,
    preview_limit: int = 250,
) -> Dict[str, Any]:
    db_file = resolve_project_path(db_path)
    out_root = resolve_project_path(output_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / f"graph_export_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    try:
        event_sql = """
            SELECT event_id, fact_key, fact_type, skeleton_text, keywords, role, status, is_latest,
                   salience, first_seen_step, last_seen_step, extract_confidence, raw_span
            FROM events
        """
        if active_only:
            event_sql += " WHERE is_latest=1"
        event_sql += " ORDER BY is_latest DESC, last_seen_step DESC, salience DESC"
        if int(event_limit) > 0:
            event_sql += " LIMIT ?"
            event_rows = _load_rows(conn, event_sql, (int(event_limit),))
        else:
            event_rows = _load_rows(conn, event_sql)

        event_ids = [str(row["event_id"]) for row in event_rows]
        edge_rows = []
        node_rows = []
        node_edge_rows = []
        if event_ids:
            marks = ",".join("?" for _ in event_ids)
            edge_rows = _load_rows(
                conn,
                f"""
                SELECT edge_id, from_event_id, to_event_id, relation, weight, created_step
                FROM edges
                WHERE from_event_id IN ({marks}) OR to_event_id IN ({marks})
                """,
                tuple(event_ids) + tuple(event_ids),
            )
            node_rows = _load_rows(
                conn,
                f"""
                SELECT node_id, event_id, node_kind, node_text, is_core, created_step
                FROM event_nodes
                WHERE event_id IN ({marks})
                """,
                tuple(event_ids),
            )
            node_ids = [str(row["node_id"]) for row in node_rows]
            if node_ids:
                node_marks = ",".join("?" for _ in node_ids)
                node_edge_rows = _load_rows(
                    conn,
                    f"""
                    SELECT node_edge_id, event_id, from_node_id, to_node_id, relation, weight, created_step
                    FROM event_node_edges
                    WHERE event_id IN ({marks})
                    """,
                    tuple(event_ids),
                )

        event_graph = _build_event_graph(event_rows, edge_rows)
        node_graph = _build_node_graph(node_rows, node_edge_rows)
        event_json = _graph_to_json(event_graph)
        node_json = _graph_to_json(node_graph)

        summary = {
            "db_path": str(db_file),
            "exported_at": datetime.now().isoformat(timespec="seconds"),
            "active_only": bool(active_only),
            "event_limit": int(event_limit),
            "preview_limit": int(preview_limit),
            "event_nodes": event_graph.number_of_nodes(),
            "event_edges": event_graph.number_of_edges(),
            "node_nodes": node_graph.number_of_nodes(),
            "node_edges": node_graph.number_of_edges(),
        }

        event_graphml = out_dir / "event_graph.graphml"
        node_graphml = out_dir / "node_graph.graphml"
        nx.write_graphml(event_graph, event_graphml)
        nx.write_graphml(node_graph, node_graphml)
        (out_dir / "event_graph.json").write_text(
            json.dumps({"summary": summary, **event_json}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (out_dir / "node_graph.json").write_text(
            json.dumps({"summary": summary, **node_json}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        preview_graph = {
            "nodes": event_json["nodes"][: max(1, int(preview_limit))],
            "edges": [
                edge
                for edge in event_json["edges"]
                if edge["from"] in {node["id"] for node in event_json["nodes"][: max(1, int(preview_limit))]}
                and edge["to"] in {node["id"] for node in event_json["nodes"][: max(1, int(preview_limit))]}
            ],
        }
        _write_html_preview(
            out_path=out_dir / "event_graph_preview.html",
            title="Long Memory Event Graph Preview",
            graph_json=preview_graph,
            note=f"Exported from {db_file.name}. Open the GraphML in Gephi/Cytoscape for the full graph.",
        )
        node_preview_graph = {
            "nodes": node_json["nodes"][: max(1, int(preview_limit))],
            "edges": [
                edge
                for edge in node_json["edges"]
                if edge["from"] in {node["id"] for node in node_json["nodes"][: max(1, int(preview_limit))]}
                and edge["to"] in {node["id"] for node in node_json["nodes"][: max(1, int(preview_limit))]}
            ],
        }
        _write_html_preview(
            out_path=out_dir / "node_graph_preview.html",
            title="Long Memory Node Graph Preview",
            graph_json=node_preview_graph,
            note=f"Exported from {db_file.name}. This view shows the intra-event node structure.",
        )

        print(f"output_dir: {out_dir}")
        print(f"event_graphml: {event_graphml}")
        print(f"node_graphml: {node_graphml}")
        print(f"event_graph_json: {out_dir / 'event_graph.json'}")
        print(f"node_graph_json: {out_dir / 'node_graph.json'}")
        print(f"preview_html: {out_dir / 'event_graph_preview.html'}")
        print(f"node_preview_html: {out_dir / 'node_graph_preview.html'}")
        return {"summary": summary, "output_dir": str(out_dir)}
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export long-memory graph artifacts.")
    parser.add_argument(
        "--db-path",
        default="data/processed/long_memory.db",
        help="Long-memory SQLite database path.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/graphs",
        help="Base directory for exported graph artifacts.",
    )
    parser.add_argument(
        "--active-only",
        action="store_true",
        help="Export only latest/active events.",
    )
    parser.add_argument(
        "--event-limit",
        type=int,
        default=0,
        help="Optional cap on exported events. 0 means no cap.",
    )
    parser.add_argument(
        "--preview-limit",
        type=int,
        default=250,
        help="Limit for the HTML preview graph.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_graph(
        db_path=args.db_path,
        output_dir=args.output_dir,
        active_only=bool(args.active_only),
        event_limit=int(args.event_limit),
        preview_limit=int(args.preview_limit),
    )


if __name__ == "__main__":
    main()
