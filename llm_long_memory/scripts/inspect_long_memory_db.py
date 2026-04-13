"""Inspect long-memory SQLite quality quickly."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import List

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from llm_long_memory.utils.helpers import resolve_project_path


def _noise_ratio(keywords: List[str]) -> float:
    if not keywords:
        return 0.0
    noise = {"assistant", "user", "system", "im", "i", "you", "we", "ck"}
    bad = sum(1 for k in keywords if str(k).strip().lower() in noise)
    return float(bad) / float(len(keywords))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect long-memory DB")
    parser.add_argument("--db", default="data/processed/long_memory.db")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    db_path = resolve_project_path(args.db)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    c_events = int(conn.execute("select count(*) from events").fetchone()[0])
    c_details = int(conn.execute("select count(*) from details").fetchone()[0])
    c_edges = int(conn.execute("select count(*) from edges").fetchone()[0])

    print("=== LONG MEMORY DB INSPECT ===")
    print(f"db={db_path}")
    print(f"events={c_events} details={c_details} edges={c_edges}")

    rows = conn.execute(
        """
        select event_id, fact_key, extract_confidence, skeleton_text, keywords, raw_span, last_seen_step
        from events
        order by last_seen_step desc
        limit ?
        """,
        (args.limit,),
    ).fetchall()

    print("\n--- latest events ---")
    for idx, row in enumerate(rows, start=1):
        try:
            kws = json.loads(str(row["keywords"] or "[]"))
            if not isinstance(kws, list):
                kws = []
        except (TypeError, ValueError, json.JSONDecodeError):
            kws = []
        nr = _noise_ratio([str(x) for x in kws])
        print(f"[{idx}] id={row['event_id']} conf={float(row['extract_confidence'] or 0.0):.2f} noise_ratio={nr:.2f}")
        print(f"    fact_key={str(row['fact_key'])[:120]}")
        print(f"    skeleton={str(row['skeleton_text'])[:220]}")
        print(f"    raw_span={str(row['raw_span'])[:220]}")
        print(f"    keywords={kws[:12]}")

    conn.close()


if __name__ == "__main__":
    main()
