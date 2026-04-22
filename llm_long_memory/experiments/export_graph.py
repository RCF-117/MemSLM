"""Export active light-graph artifacts and HTML previews from audit results."""

from __future__ import annotations

import argparse
import html
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

from llm_long_memory.memory.evidence_light_graph import EvidenceLightGraph
from llm_long_memory.utils.helpers import resolve_project_path


_HTML_TEMPLATE = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{title}</title>
  <script src=\"https://unpkg.com/vis-network/standalone/umd/vis-network.min.js\"></script>
  <style>
    body {{
      margin: 0;
      font-family: \"Helvetica Neue\", Helvetica, Arial, sans-serif;
      color: #111827;
      background: #fbfcfe;
    }}
    header {{
      padding: 16px 20px;
      background: #ffffff;
      border-bottom: 1px solid #d9e1ea;
    }}
    header h1 {{ margin: 0 0 6px 0; font-size: 18px; font-weight: 600; letter-spacing: 0.01em; }}
    header p {{ margin: 0; color: #526273; font-size: 13px; }}
    .meta {{ padding: 12px 20px 0; color: #6b7a8c; font-size: 12px; }}
    .legend {{
      padding: 10px 20px 0;
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      color: #536273;
      font-size: 11px;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .legend-swatch {{
      width: 10px;
      height: 10px;
      border-radius: 999px;
      border: 1px solid rgba(17, 24, 39, 0.10);
      display: inline-block;
    }}
    #network {{ width: 100vw; height: calc(100vh - 186px); background: #fbfcfe; }}
    pre {{ margin: 0; white-space: pre-wrap; }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <p>{note}</p>
  </header>
  <div class=\"meta\">{meta}</div>
  <div class=\"legend\">{legend_html}</div>
  <div id=\"network\"></div>
  <script>
    const nodes = new vis.DataSet({nodes_json});
    const edges = new vis.DataSet({edges_json});
    const network = new vis.Network(document.getElementById('network'), {{nodes, edges}}, {{
      nodes: {{
        font: {{ color: '#111827', size: 11, face: 'Helvetica Neue, Helvetica, Arial, sans-serif' }},
        borderWidth: 1,
        shadow: {{ enabled: false }}
      }},
      edges: {{
        arrows: 'to',
        color: {{ color: 'rgba(105, 122, 138, 0.32)', highlight: '#355c7d' }},
        font: {{ color: '#5b6b7b', size: 9 }},
        smooth: {{ type: 'continuous', roundness: 0.10 }}
      }},
      physics: false,
      interaction: {{ hover: true, navigationButtons: true, keyboard: true }}
    }});
    setTimeout(function () {{
      network.fit({{ animation: false, minZoomLevel: 0.08, maxZoomLevel: 1.2 }});
    }}, 40);
    window.__lightGraph = network;
  </script>
</body>
</html>
"""


def _safe_text(value: Any, limit: int = 180) -> str:
    text = " ".join(str(value or "").split()).strip()
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def _hex_to_rgba(value: str, alpha: float) -> str:
    raw = str(value or "").strip().lstrip("#")
    if len(raw) != 6:
        return f"rgba(148, 163, 184, {alpha})"
    try:
        red = int(raw[0:2], 16)
        green = int(raw[2:4], 16)
        blue = int(raw[4:6], 16)
    except ValueError:
        return f"rgba(148, 163, 184, {alpha})"
    return f"rgba({red}, {green}, {blue}, {alpha})"


def _compact_node_label(label: Any, *, node_type: str, cluster_size: int) -> str:
    text = " ".join(str(label or "").split()).strip()
    kind = str(node_type or "").strip().lower()
    if not text:
        return ""
    if kind == "query":
        return text
    if kind == "entity":
        return _safe_text(text, 14 if cluster_size <= 8 else 10) if len(text) <= (16 if cluster_size <= 8 else 12) else ""
    if kind in {"state", "event", "fact"}:
        return ""
    return ""



def _load_rows(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return [dict(item) for item in payload.get("rows", [])]
    raise ValueError("Audit JSON must be a row list or an object with a 'rows' list.")



def _iter_graph_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        graph = dict(row.get("evidence_light_graph", {}) or {})
        if not graph:
            continue
        if not list(graph.get("nodes", [])):
            continue
        out.append(row)
    return out



def _write_graph_html(
    *,
    out_path: Path,
    title: str,
    note: str,
    meta: str,
    legend_html: str,
    vis_nodes: List[Dict[str, Any]],
    vis_edges: List[Dict[str, Any]],
) -> None:
    out_path.write_text(
        _HTML_TEMPLATE.format(
            title=html.escape(title),
            note=html.escape(note),
            meta=meta,
            legend_html=legend_html,
            nodes_json=json.dumps(vis_nodes, ensure_ascii=False).replace("</", "<\\/"),
            edges_json=json.dumps(vis_edges, ensure_ascii=False).replace("</", "<\\/"),
        ),
        encoding="utf-8",
    )


def _node_fill_and_stroke(node: Dict[str, Any]) -> Dict[str, str]:
    color = node.get("color")
    if isinstance(color, dict):
        return {
            "fill": str(color.get("background", "#ffffff") or "#ffffff"),
            "stroke": str(color.get("border", "#94a3b8") or "#94a3b8"),
        }
    if isinstance(color, str) and color.strip():
        return {"fill": color, "stroke": color}
    return {"fill": "#ffffff", "stroke": "#94a3b8"}


def _svg_node_bounds(node: Dict[str, Any]) -> Dict[str, float]:
    x = float(node.get("x", 0.0) or 0.0)
    y = float(node.get("y", 0.0) or 0.0)
    size = float(node.get("size", 10.0) or 10.0)
    shape = str(node.get("shape", "dot") or "dot").strip().lower()
    label = str(node.get("label", "") or "")
    font = dict(node.get("font", {}) or {})
    font_size = float(font.get("size", 10.0) or 10.0)

    if shape == "box":
        width = max(30.0, (len(label) * font_size * 0.58) + 12.0)
        height = max(18.0, font_size * 1.8)
        return {"left": x - (width / 2.0), "right": x + (width / 2.0), "top": y - (height / 2.0), "bottom": y + (height / 2.0)}
    if shape == "ellipse":
        rx = max(10.0, size * 1.55)
        ry = max(8.0, size * 1.05)
        return {"left": x - rx, "right": x + rx, "top": y - ry, "bottom": y + ry}
    radius = max(4.0, size)
    return {"left": x - radius, "right": x + radius, "top": y - radius, "bottom": y + radius}


def _svg_escape(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


def _write_graph_svg(
    *,
    out_path: Path,
    title: str,
    note: str,
    meta: str,
    vis_nodes: List[Dict[str, Any]],
    vis_edges: List[Dict[str, Any]],
) -> None:
    if not vis_nodes:
        out_path.write_text("", encoding="utf-8")
        return

    node_bounds = [_svg_node_bounds(node) for node in vis_nodes]
    left = min(bound["left"] for bound in node_bounds)
    right = max(bound["right"] for bound in node_bounds)
    top = min(bound["top"] for bound in node_bounds)
    bottom = max(bound["bottom"] for bound in node_bounds)
    margin = 72.0
    width = max(1200.0, (right - left) + (margin * 2.0))
    height = max(900.0, (bottom - top) + (margin * 2.0) + 64.0)
    offset_x = margin - left
    offset_y = margin - top + 46.0

    node_lookup = {str(node.get("id", "")): dict(node) for node in vis_nodes}
    lines: List[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(round(width))}" height="{int(round(height))}" viewBox="0 0 {int(round(width))} {int(round(height))}">')
    lines.append("<defs>")
    lines.append('<marker id="arrowhead" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto" markerUnits="strokeWidth">')
    lines.append('<path d="M 0 0 L 8 3 L 0 6 z" fill="#7b8794" />')
    lines.append("</marker>")
    lines.append("</defs>")
    lines.append(f'<rect x="0" y="0" width="{int(round(width))}" height="{int(round(height))}" fill="#fbfcfe" />')
    lines.append(f'<text x="24" y="28" font-family="Helvetica Neue, Helvetica, Arial, sans-serif" font-size="18" font-weight="600" fill="#111827">{_svg_escape(title)}</text>')
    lines.append(f'<text x="24" y="48" font-family="Helvetica Neue, Helvetica, Arial, sans-serif" font-size="11" fill="#5b6b7b">{_svg_escape(note)}</text>')
    lines.append(f'<text x="24" y="65" font-family="Helvetica Neue, Helvetica, Arial, sans-serif" font-size="10" fill="#7b8794">{_svg_escape(meta)}</text>')
    legend_x = 24.0
    legend_y = 86.0
    for item in _legend_items():
        color = _answer_type_family_color(item["family"])
        lines.append(f'<circle cx="{legend_x:.2f}" cy="{legend_y:.2f}" r="4.5" fill="{color}" stroke="rgba(17,24,39,0.12)" stroke-width="0.6" />')
        lines.append(
            f'<text x="{legend_x + 10.0:.2f}" y="{legend_y + 3.2:.2f}" font-family="Helvetica Neue, Helvetica, Arial, sans-serif" '
            f'font-size="10" fill="#5b6b7b">{_svg_escape(item["label"])}</text>'
        )
        legend_x += 92.0

    for edge in vis_edges:
        src = node_lookup.get(str(edge.get("from", "")))
        dst = node_lookup.get(str(edge.get("to", "")))
        if not src or not dst:
            continue
        x1 = float(src.get("x", 0.0) or 0.0) + offset_x
        y1 = float(src.get("y", 0.0) or 0.0) + offset_y
        x2 = float(dst.get("x", 0.0) or 0.0) + offset_x
        y2 = float(dst.get("y", 0.0) or 0.0) + offset_y
        stroke = "rgba(105, 122, 138, 0.30)"
        title_text = str(edge.get("title", "") or "")
        lines.append('<g>')
        if title_text:
            lines.append(f"<title>{_svg_escape(title_text)}</title>")
        lines.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{stroke}" stroke-width="1.05" marker-end="url(#arrowhead)" />'
        )
        lines.append("</g>")

    for node in vis_nodes:
        x = float(node.get("x", 0.0) or 0.0) + offset_x
        y = float(node.get("y", 0.0) or 0.0) + offset_y
        size = float(node.get("size", 10.0) or 10.0)
        shape = str(node.get("shape", "dot") or "dot").strip().lower()
        label = str(node.get("label", "") or "")
        title_text = str(node.get("title", "") or "")
        font = dict(node.get("font", {}) or {})
        font_size = float(font.get("size", 10.0) or 10.0)
        font_color = str(font.get("color", "#334155") or "#334155")
        style = _node_fill_and_stroke(node)

        lines.append("<g>")
        if title_text:
            lines.append(f"<title>{_svg_escape(title_text)}</title>")

        if shape == "box":
            width_px = max(30.0, (len(label) * font_size * 0.58) + 12.0)
            height_px = max(18.0, font_size * 1.8)
            lines.append(
                f'<rect x="{x - (width_px / 2.0):.2f}" y="{y - (height_px / 2.0):.2f}" '
                f'width="{width_px:.2f}" height="{height_px:.2f}" rx="4" ry="4" '
                f'fill="{style["fill"]}" stroke="{style["stroke"]}" stroke-width="0.8" />'
            )
        elif shape == "ellipse":
            rx = max(10.0, size * 1.55)
            ry = max(8.0, size * 1.05)
            lines.append(
                f'<ellipse cx="{x:.2f}" cy="{y:.2f}" rx="{rx:.2f}" ry="{ry:.2f}" '
                f'fill="{style["fill"]}" stroke="{style["stroke"]}" stroke-width="0.9" />'
            )
        else:
            radius = max(4.0, size)
            lines.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" '
                f'fill="{style["fill"]}" stroke="{style["stroke"]}" stroke-width="0.9" />'
            )

        if label:
            lines.append(
                f'<text x="{x:.2f}" y="{y + (font_size * 0.33):.2f}" '
                f'font-family="Helvetica Neue, Helvetica, Arial, sans-serif" '
                f'font-size="{font_size:.1f}" fill="{font_color}" text-anchor="middle">{_svg_escape(label)}</text>'
            )
        lines.append("</g>")

    lines.append("</svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _combined_cluster_positions(node_count: int, *, center_x: int, center_y: int) -> List[Dict[str, int]]:
    if node_count <= 0:
        return []
    positions: List[Dict[str, int]] = [{"x": center_x, "y": center_y}]
    if node_count == 1:
        return positions
    remaining = node_count - 1
    placed = 0
    ring_index = 0
    while placed < remaining:
        ring_index += 1
        ring_capacity = 7 + ((ring_index - 1) * 3)
        ring_radius = 96 + ((ring_index - 1) * 70)
        take = min(ring_capacity, remaining - placed)
        for offset in range(take):
            angle = (2.0 * math.pi * float(offset)) / float(max(1, take))
            x = int(round(center_x + (math.cos(angle) * ring_radius)))
            y = int(round(center_y + (math.sin(angle) * ring_radius)))
            positions.append({"x": x, "y": y})
        placed += take
    return positions


def _cluster_extent(node_count: int) -> int:
    if node_count <= 1:
        return 96
    local_positions = _combined_cluster_positions(node_count, center_x=0, center_y=0)
    max_radius = 0.0
    for pos in local_positions:
        max_radius = max(max_radius, math.sqrt(float(pos["x"] ** 2 + pos["y"] ** 2)))
    return int(round(max_radius + 84.0))


def _ellipse_circumference(radius_x: float, radius_y: float) -> float:
    if radius_x <= 0.0 or radius_y <= 0.0:
        return 0.0
    h = ((radius_x - radius_y) ** 2.0) / max((radius_x + radius_y) ** 2.0, 1e-6)
    return math.pi * (radius_x + radius_y) * (1.0 + ((3.0 * h) / (10.0 + math.sqrt(max(4.0 - (3.0 * h), 1e-6)))))


def _band_capacities(total_clusters: int) -> List[int]:
    if total_clusters <= 0:
        return []
    if total_clusters <= 5:
        return [total_clusters]
    if total_clusters <= 12:
        first = min(6, total_clusters)
        return [first, total_clusters - first]
    first = min(6, total_clusters)
    second = min(8, total_clusters - first)
    third = total_clusters - first - second
    return [count for count in [first, second, third] if count > 0]


def _answer_type_family(answer_type: Any) -> str:
    text = str(answer_type or "").strip().lower()
    if text.startswith("temporal"):
        return "temporal"
    if text.startswith("count"):
        return "count"
    if text.startswith("update"):
        return "update"
    if text.startswith("preference"):
        return "preference"
    if text.startswith("factoid"):
        return "factoid"
    return "other"


def _answer_type_family_order(family: str) -> int:
    order = {
        "count": 0,
        "temporal": 1,
        "update": 2,
        "factoid": 3,
        "preference": 4,
        "other": 5,
    }
    return int(order.get(str(family or "").strip().lower(), 99))


def _answer_type_anchor(family: str) -> Dict[str, int]:
    anchors = {
        "count": {"x": -1160, "y": -120},
        "temporal": {"x": -60, "y": -760},
        "update": {"x": -760, "y": 620},
        "factoid": {"x": 1140, "y": -10},
        "preference": {"x": 520, "y": 700},
        "other": {"x": 0, "y": 0},
    }
    return dict(anchors.get(str(family or "").strip().lower(), anchors["other"]))


def _answer_type_family_color(family: str) -> str:
    palette = {
        "count": "#86a97e",
        "temporal": "#7f9fb7",
        "update": "#c39a7d",
        "factoid": "#7e8e9f",
        "preference": "#b39ab9",
        "other": "#9aa3ad",
    }
    return str(palette.get(str(family or "").strip().lower(), palette["other"]))


def _legend_items() -> List[Dict[str, str]]:
    return [
        {"label": "Count", "family": "count"},
        {"label": "Temporal", "family": "temporal"},
        {"label": "Update", "family": "update"},
        {"label": "Factoid", "family": "factoid"},
        {"label": "Preference", "family": "preference"},
    ]


def _build_html_legend() -> str:
    parts: List[str] = []
    for item in _legend_items():
        color = _answer_type_family_color(item["family"])
        parts.append(
            '<span class="legend-item">'
            f'<span class="legend-swatch" style="background:{html.escape(color)}"></span>'
            f'{html.escape(item["label"])}'
            "</span>"
        )
    return "".join(parts)


def _build_elliptical_band_centers(node_counts: List[int]) -> List[Dict[str, int]]:
    if not node_counts:
        return []
    centers: List[Dict[str, int]] = []
    if len(node_counts) == 1:
        return [{"x": 0, "y": 0}]

    centers.append({"x": 0, "y": 0})
    remaining_counts = list(node_counts[1:])
    capacities = _band_capacities(len(remaining_counts))
    cursor = 0
    previous_radius_x = 0.0
    previous_radius_y = 0.0
    previous_extent = float(_cluster_extent(node_counts[0]))

    for band_index, capacity in enumerate(capacities):
        band_counts = remaining_counts[cursor : cursor + capacity]
        cursor += capacity
        if not band_counts:
            continue
        band_extents = [_cluster_extent(count) for count in band_counts]
        max_extent = float(max(band_extents))
        required_span = sum((float(extent) * 2.45) for extent in band_extents)

        if band_index == 0:
            radius_x = max(560.0, previous_extent + max_extent + 250.0)
            radius_y = max(320.0, radius_x * 0.56)
        else:
            radius_x = previous_radius_x + (previous_extent * 1.05) + (max_extent * 0.86) + 235.0
            radius_y = previous_radius_y + (previous_extent * 0.68) + (max_extent * 0.54) + 120.0

        while _ellipse_circumference(radius_x, radius_y) * 0.82 < required_span:
            radius_x += 80.0
            radius_y += 42.0

        total_weight = sum((float(extent) * 2.45) for extent in band_extents)
        angle_cursor = (-math.pi / 2.0) + (band_index * 0.22)
        for extent in band_extents:
            arc = (float(extent) * 2.45) / max(total_weight, 1.0)
            mid_angle = angle_cursor + (arc * math.pi)
            x = int(round(math.cos(mid_angle) * radius_x))
            y = int(round(math.sin(mid_angle) * radius_y))
            centers.append({"x": x, "y": y})
            angle_cursor += arc * (2.0 * math.pi)

        previous_radius_x = radius_x
        previous_radius_y = radius_y
        previous_extent = max_extent
    return centers


def _build_grouped_family_centers(prepared_rows: List[Dict[str, Any]]) -> List[Dict[str, int]]:
    if not prepared_rows:
        return []
    centers: List[Dict[str, int]] = []
    cursor = 0
    while cursor < len(prepared_rows):
        family = str(prepared_rows[cursor].get("answer_family", "other"))
        group: List[Dict[str, Any]] = []
        probe = cursor
        while probe < len(prepared_rows) and str(prepared_rows[probe].get("answer_family", "other")) == family:
            group.append(prepared_rows[probe])
            probe += 1
        local_centers = _build_elliptical_band_centers([int(item.get("node_count", 0) or 0) for item in group])
        anchor = _answer_type_anchor(family)
        for local in local_centers:
            centers.append(
                {
                    "x": int(anchor["x"] + local["x"]),
                    "y": int(anchor["y"] + local["y"]),
                }
            )
        cursor = probe
    return centers


def _build_combined_graph_payload(
    *,
    rows: List[Dict[str, Any]],
    builder: EvidenceLightGraph,
    title_prefix: str,
) -> Dict[str, Any]:
    combined_nodes: List[Dict[str, Any]] = []
    combined_edges: List[Dict[str, Any]] = []
    graph_rows: List[Dict[str, Any]] = []
    prepared_rows: List[Dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        qid = str(row.get("question_id") or row.get("id") or f"row_{index}")
        graph = dict(row.get("evidence_light_graph", {}) or {})
        viz = builder.build_visualization_bundle(graph)
        base_nodes = list(viz.get("vis_nodes", []))
        base_edges = list(viz.get("vis_edges", []))
        if not base_nodes:
            continue
        prepared_rows.append(
            {
                "row": row,
                "question_id": qid,
                "viz": viz,
                "base_nodes": base_nodes,
                "base_edges": base_edges,
                "node_count": len(base_nodes),
                "answer_family": _answer_type_family(viz.get("answer_type", "")),
            }
        )

    prepared_rows.sort(
        key=lambda item: (
            _answer_type_family_order(item.get("answer_family", "other")),
            -int(item["node_count"]),
            str(item["question_id"]),
        )
    )
    centers = _build_grouped_family_centers(prepared_rows)

    for index, item in enumerate(prepared_rows):
        qid = str(item["question_id"])
        viz = dict(item["viz"])
        base_nodes = list(item["base_nodes"])
        base_edges = list(item["base_edges"])
        center = centers[index] if index < len(centers) else {"x": 0, "y": 0}
        center_x = int(center["x"])
        center_y = int(center["y"])
        positions = _combined_cluster_positions(
            len(base_nodes),
            center_x=center_x,
            center_y=center_y,
        )
        cluster_color = _answer_type_family_color(str(item.get("answer_family", "other")))
        local_node_ids: List[str] = []
        local_edge_ids: List[str] = []
        ordered_nodes = sorted(
            base_nodes,
            key=lambda item: (0 if str(item.get("type", "")) == "query" else 1, str(item.get("id", ""))),
        )
        node_id_map = {
            str(node.get("id", "")): f"{qid}::{str(node.get('id', ''))}"
            for node in ordered_nodes
        }

        halo_id = f"{qid}::cluster_halo"
        halo_size = max(70, min(132, 58 + (len(base_nodes) * 5)))
        combined_nodes.append(
            {
                "id": halo_id,
                "label": "",
                "shape": "dot",
                "size": halo_size,
                "x": center_x,
                "y": center_y,
                "fixed": {"x": True, "y": True},
                "physics": False,
                "borderWidth": 0,
                "color": {
                    "background": _hex_to_rgba(cluster_color, 0.035),
                    "border": _hex_to_rgba(cluster_color, 0.08),
                    "highlight": {"background": _hex_to_rgba(cluster_color, 0.045), "border": _hex_to_rgba(cluster_color, 0.10)},
                    "hover": {"background": _hex_to_rgba(cluster_color, 0.045), "border": _hex_to_rgba(cluster_color, 0.10)},
                },
                "font": {"size": 1, "color": "rgba(0,0,0,0)"},
                "meta": {"question_id": qid, "role": "cluster_halo"},
            }
        )

        for node_index, (node, pos) in enumerate(zip(ordered_nodes, positions)):
            prefixed_id = node_id_map[str(node.get("id", ""))]
            local_node_ids.append(prefixed_id)
            node_type = str(node.get("type", "")).strip().lower()
            label = str(node.get("label", ""))
            if node_type == "query":
                label = qid
            else:
                label = _compact_node_label(
                    label,
                    node_type=node_type,
                    cluster_size=len(base_nodes),
                )
            combined_nodes.append(
                {
                    **dict(node),
                    "id": prefixed_id,
                    "label": label,
                    "title": f"{qid} | {str(node.get('title', ''))}".strip(" |"),
                    "x": pos["x"],
                    "y": pos["y"],
                    "fixed": {"x": True, "y": True},
                    "physics": False,
                    "shape": "dot" if node_type == "query" else node.get("shape", "dot"),
                    "size": 11 if node_type == "query" else node.get("size", 10),
                    "borderWidth": 0.9 if node_type == "query" else node.get("borderWidth", 1),
                    "font": {
                        "size": 9 if node_type == "query" else 10,
                        "color": "#475569" if node_type == "query" else "#334155",
                        "face": "Helvetica Neue, Helvetica, Arial, sans-serif",
                    },
                    "color": {
                        "background": "#ffffff",
                        "border": _hex_to_rgba(cluster_color, 0.62 if node_type == "query" else 0.48),
                        "highlight": {"background": "#ffffff", "border": _hex_to_rgba(cluster_color, 0.74 if node_type == "query" else 0.66)},
                        "hover": {"background": "#ffffff", "border": _hex_to_rgba(cluster_color, 0.74 if node_type == "query" else 0.66)},
                    },
                    "meta": {
                        "question_id": qid,
                        "answer_type": str(viz.get("answer_type", "")),
                        "cluster_index": index,
                        "cluster_node_index": node_index,
                    },
                }
            )

        for edge in base_edges:
            src = node_id_map.get(str(edge.get("from", "")))
            dst = node_id_map.get(str(edge.get("to", "")))
            if not src or not dst:
                continue
            edge_id = f"{qid}::{str(edge.get('id', ''))}"
            local_edge_ids.append(edge_id)
            edge_kind = str(edge.get("label", "")).strip().lower()
            combined_edges.append(
                {
                    **dict(edge),
                    "id": edge_id,
                    "from": src,
                    "to": dst,
                    "label": "",
                    "title": f"{qid} | {str(edge.get('title', ''))}".strip(" |"),
                }
            )

        graph_rows.append(
            {
                "question_id": qid,
                "query": str(viz.get("query", "")),
                "answer_type": str(viz.get("answer_type", "")),
                "stats": dict(viz.get("stats", {}) or {}),
                "node_ids": local_node_ids,
                "edge_ids": local_edge_ids,
                "center": {"x": center_x, "y": center_y},
            }
        )

    return {
        "title": f"{title_prefix} Combined Light Graph",
        "note": "All question-scoped light graphs are composed into one publication-style canvas.",
        "meta": {
            "graph_count": len(graph_rows),
            "node_count": len(combined_nodes),
            "edge_count": len(combined_edges),
            "layout": "answer_type_grouped_elliptical_bands",
        },
        "vis_nodes": combined_nodes,
        "vis_edges": combined_edges,
        "graphs": graph_rows,
    }


def export_graph(
    *,
    audit_json_path: str,
    output_dir: str,
    artifact_prefix: str = "",
    max_graphs: int = 0,
) -> Dict[str, Any]:
    """Export one combined light-graph JSON/HTML overview from an audit artifact."""

    audit_file = resolve_project_path(audit_json_path)
    out_root = resolve_project_path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    prefix = str(artifact_prefix or "").strip() or datetime.now().strftime("light_graph_export_%Y%m%d_%H%M%S")
    prefix = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in prefix)

    rows = _iter_graph_rows(_load_rows(audit_file))
    if int(max_graphs) > 0:
        rows = rows[: int(max_graphs)]

    builder = EvidenceLightGraph({})
    manifest_rows: List[Dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        qid = str(row.get("question_id") or row.get("id") or f"row_{index}")
        graph = dict(row.get("evidence_light_graph", {}) or {})
        viz = builder.build_visualization_bundle(graph)
        stats = dict(viz.get("stats", {}) or {})
        manifest_rows.append(
            {
                "question_id": qid,
                "query": str(viz.get("query", "")),
                "answer_type": str(viz.get("answer_type", "")),
                "stats": stats,
            }
        )

    combined_payload: Dict[str, Any] = {}
    combined_json_path = ""
    combined_html_path = ""
    combined_svg_path = ""
    if rows:
        combined_payload = _build_combined_graph_payload(
            rows=rows,
            builder=builder,
            title_prefix=prefix,
        )
        combined_json = out_root / f"{prefix}__combined.json"
        combined_html = out_root / f"{prefix}__combined.html"
        combined_svg = out_root / f"{prefix}__combined.svg"
        combined_json.write_text(
            json.dumps(combined_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        combined_meta = (
            f"graphs={int(combined_payload.get('meta', {}).get('graph_count', 0) or 0)} | "
            f"nodes={int(combined_payload.get('meta', {}).get('node_count', 0) or 0)} | "
            f"edges={int(combined_payload.get('meta', {}).get('edge_count', 0) or 0)} | "
            f"layout={str(combined_payload.get('meta', {}).get('layout', ''))}"
        )
        _write_graph_html(
            out_path=combined_html,
            title=str(combined_payload.get("title", "")),
            note=str(combined_payload.get("note", "")),
            meta=combined_meta,
            legend_html=_build_html_legend(),
            vis_nodes=list(combined_payload.get("vis_nodes", [])),
            vis_edges=list(combined_payload.get("vis_edges", [])),
        )
        _write_graph_svg(
            out_path=combined_svg,
            title=str(combined_payload.get("title", "")),
            note=str(combined_payload.get("note", "")),
            meta=combined_meta,
            vis_nodes=list(combined_payload.get("vis_nodes", [])),
            vis_edges=list(combined_payload.get("vis_edges", [])),
        )
        combined_json_path = str(combined_json)
        combined_html_path = str(combined_html)
        combined_svg_path = str(combined_svg)

    manifest = {
        "audit_json_path": str(audit_file),
        "output_dir": str(out_root),
        "artifact_prefix": prefix,
        "graph_count": len(manifest_rows),
        "graphs": manifest_rows,
        "combined_graph": {
            "json_path": combined_json_path,
            "html_path": combined_html_path,
            "svg_path": combined_svg_path,
            "graph_count": int(combined_payload.get("meta", {}).get("graph_count", 0) or 0),
            "node_count": int(combined_payload.get("meta", {}).get("node_count", 0) or 0),
            "edge_count": int(combined_payload.get("meta", {}).get("edge_count", 0) or 0),
        },
    }
    manifest_path = out_root / f"{prefix}__manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export active light-graph visualization artifacts.")
    parser.add_argument("--audit-json", required=True, help="Audit JSON containing evidence_light_graph rows.")
    parser.add_argument("--output-dir", required=True, help="Directory for HTML/JSON graph exports.")
    parser.add_argument("--artifact-prefix", default="", help="Optional output prefix.")
    parser.add_argument("--max-graphs", type=int, default=0, help="Optional cap on exported graphs.")
    return parser.parse_args()



def main() -> None:
    args = _parse_args()
    result = export_graph(
        audit_json_path=args.audit_json,
        output_dir=args.output_dir,
        artifact_prefix=args.artifact_prefix,
        max_graphs=args.max_graphs,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
