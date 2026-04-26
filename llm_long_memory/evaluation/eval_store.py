"""Evaluation persistence store backed by SQLite."""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Any, Sequence


class EvalStore:
    """Encapsulates eval run/result persistence for MidMemory."""

    def __init__(self, conn: sqlite3.Connection, eval_cfg: dict[str, Any]) -> None:
        self.conn = conn
        self.save_to_db = bool(eval_cfg["save_to_db"])
        self.run_table = self._sanitize_identifier(str(eval_cfg["run_table"]))
        self.result_table = self._sanitize_identifier(str(eval_cfg["result_table"]))
        self.group_table = self._sanitize_identifier(str(eval_cfg["group_table"]))
        self.thesis_run_table = self._sanitize_identifier(
            str(eval_cfg.get("thesis_run_table", "thesis_eval_runs"))
        )
        self.thesis_type_table = self._sanitize_identifier(
            str(eval_cfg.get("thesis_type_table", "thesis_eval_type_metrics"))
        )
        self.thesis_mode_table = self._sanitize_identifier(
            str(eval_cfg.get("thesis_mode_table", "thesis_mode_runs"))
        )

    @staticmethod
    def _sanitize_identifier(name: str) -> str:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"Invalid SQL identifier in config: {name}")
        return name

    def create_tables(self) -> None:
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.run_table}(
              run_id TEXT PRIMARY KEY,
              dataset_path TEXT,
              started_at TEXT,
              finished_at TEXT,
              total INTEGER,
              matched INTEGER,
              accuracy REAL,
              final_answer_acc REAL,
              retrieval_answer_span_hit_rate REAL,
              retrieval_support_sentence_hit_rate REAL,
              retrieval_evidence_hit_rate REAL,
              graph_answer_span_hit_rate REAL,
              graph_support_sentence_hit_rate REAL,
              graph_ingest_accept_rate REAL,
              avg_answer_token_density REAL,
              avg_noise_density REAL,
              avg_latency_sec REAL,
              isolated INTEGER
            )
            """
        )
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.result_table}(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id TEXT,
              question_id TEXT,
              question_type TEXT,
              question TEXT,
              expected_answer TEXT,
              prediction TEXT,
              is_match INTEGER,
              evidence_hit INTEGER,
              evidence_recall REAL,
              answer_span_hit INTEGER,
              support_sentence_hit INTEGER,
              graph_answer_span_hit INTEGER,
              graph_support_sentence_hit INTEGER,
              answer_token_density REAL,
              noise_density REAL,
              latency_sec REAL,
              retrieved_session_ids TEXT
            )
            """
        )
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.group_table}(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id TEXT,
              group_key TEXT,
              total INTEGER,
              matched INTEGER,
              accuracy REAL
            )
            """
        )
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.thesis_run_table}(
              run_id TEXT PRIMARY KEY,
              final_answer_acc REAL,
              retrieval_answer_span_hit_rate REAL,
              retrieval_support_sentence_hit_rate REAL,
              graph_answer_span_hit_rate REAL,
              graph_support_sentence_hit_rate REAL,
              graph_ingest_accept_rate REAL,
              avg_answer_token_density REAL,
              avg_noise_density REAL,
              avg_latency_sec REAL
            )
            """
        )
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.thesis_type_table}(
              run_id TEXT,
              question_type TEXT,
              type_answer_acc REAL,
              type_answer_token_density REAL,
              type_noise_density REAL,
              type_latency_sec REAL,
              PRIMARY KEY(run_id, question_type)
            )
            """
        )
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.thesis_mode_table}(
              dataset_name TEXT,
              mode TEXT,
              run_id TEXT,
              model_name TEXT,
              judge_model TEXT,
              created_at TEXT,
              PRIMARY KEY(dataset_name, mode, run_id)
            )
            """
        )
        self.conn.commit()

    def ensure_schema_compat(self) -> None:
        run_cols = self.conn.execute(f"PRAGMA table_info({self.run_table})").fetchall()
        run_names = {str(row["name"]) for row in run_cols}
        if "retrieval_answer_span_hit_rate" not in run_names:
            self.conn.execute(
                f"ALTER TABLE {self.run_table} ADD COLUMN retrieval_answer_span_hit_rate REAL"
            )
        if "retrieval_support_sentence_hit_rate" not in run_names:
            self.conn.execute(
                f"ALTER TABLE {self.run_table} ADD COLUMN retrieval_support_sentence_hit_rate REAL"
            )
        if "retrieval_evidence_hit_rate" not in run_names:
            self.conn.execute(
                f"ALTER TABLE {self.run_table} ADD COLUMN retrieval_evidence_hit_rate REAL"
            )
        if "final_answer_acc" not in run_names:
            self.conn.execute(f"ALTER TABLE {self.run_table} ADD COLUMN final_answer_acc REAL")
        if "graph_answer_span_hit_rate" not in run_names:
            self.conn.execute(f"ALTER TABLE {self.run_table} ADD COLUMN graph_answer_span_hit_rate REAL")
        if "graph_support_sentence_hit_rate" not in run_names:
            self.conn.execute(
                f"ALTER TABLE {self.run_table} ADD COLUMN graph_support_sentence_hit_rate REAL"
            )
        if "graph_ingest_accept_rate" not in run_names:
            self.conn.execute(f"ALTER TABLE {self.run_table} ADD COLUMN graph_ingest_accept_rate REAL")
        if "avg_answer_token_density" not in run_names:
            self.conn.execute(f"ALTER TABLE {self.run_table} ADD COLUMN avg_answer_token_density REAL")
        if "avg_noise_density" not in run_names:
            self.conn.execute(f"ALTER TABLE {self.run_table} ADD COLUMN avg_noise_density REAL")
        if "avg_latency_sec" not in run_names:
            self.conn.execute(f"ALTER TABLE {self.run_table} ADD COLUMN avg_latency_sec REAL")
        thesis_run_cols = self.conn.execute(f"PRAGMA table_info({self.thesis_run_table})").fetchall()
        thesis_run_names = {str(row["name"]) for row in thesis_run_cols}
        thesis_run_expected = {
            "run_id",
            "final_answer_acc",
            "retrieval_answer_span_hit_rate",
            "retrieval_support_sentence_hit_rate",
            "graph_answer_span_hit_rate",
            "graph_support_sentence_hit_rate",
            "graph_ingest_accept_rate",
            "avg_answer_token_density",
            "avg_noise_density",
            "avg_latency_sec",
        }
        if not thesis_run_expected.issubset(thesis_run_names):
            if "avg_answer_token_density" not in thesis_run_names:
                self.conn.execute(
                    f"ALTER TABLE {self.thesis_run_table} ADD COLUMN avg_answer_token_density REAL"
                )
            if "avg_noise_density" not in thesis_run_names:
                self.conn.execute(
                    f"ALTER TABLE {self.thesis_run_table} ADD COLUMN avg_noise_density REAL"
                )
        thesis_type_cols = self.conn.execute(f"PRAGMA table_info({self.thesis_type_table})").fetchall()
        thesis_type_names = {str(row["name"]) for row in thesis_type_cols}
        thesis_type_expected = {
            "run_id",
            "question_type",
            "type_answer_acc",
            "type_answer_token_density",
            "type_noise_density",
            "type_latency_sec",
        }
        if not thesis_type_expected.issubset(thesis_type_names):
            if "type_answer_token_density" not in thesis_type_names:
                self.conn.execute(
                    f"ALTER TABLE {self.thesis_type_table} ADD COLUMN type_answer_token_density REAL"
                )
            if "type_noise_density" not in thesis_type_names:
                self.conn.execute(
                    f"ALTER TABLE {self.thesis_type_table} ADD COLUMN type_noise_density REAL"
                )
        thesis_mode_cols = self.conn.execute(f"PRAGMA table_info({self.thesis_mode_table})").fetchall()
        thesis_mode_names = {str(row["name"]) for row in thesis_mode_cols}
        thesis_mode_expected = {
            "dataset_name",
            "mode",
            "run_id",
            "model_name",
            "judge_model",
            "created_at",
        }
        if not thesis_mode_expected.issubset(thesis_mode_names):
            pass

        cols = self.conn.execute(f"PRAGMA table_info({self.result_table})").fetchall()
        names = {str(row["name"]) for row in cols}
        if "evidence_hit" not in names:
            self.conn.execute(f"ALTER TABLE {self.result_table} ADD COLUMN evidence_hit INTEGER")
        if "evidence_recall" not in names:
            self.conn.execute(f"ALTER TABLE {self.result_table} ADD COLUMN evidence_recall REAL")
        if "answer_span_hit" not in names:
            self.conn.execute(f"ALTER TABLE {self.result_table} ADD COLUMN answer_span_hit INTEGER")
        if "support_sentence_hit" not in names:
            self.conn.execute(f"ALTER TABLE {self.result_table} ADD COLUMN support_sentence_hit INTEGER")
        if "retrieved_session_ids" not in names:
            self.conn.execute(f"ALTER TABLE {self.result_table} ADD COLUMN retrieved_session_ids TEXT")
        if "graph_answer_span_hit" not in names:
            self.conn.execute(f"ALTER TABLE {self.result_table} ADD COLUMN graph_answer_span_hit INTEGER")
        if "graph_support_sentence_hit" not in names:
            self.conn.execute(
                f"ALTER TABLE {self.result_table} ADD COLUMN graph_support_sentence_hit INTEGER"
            )
        if "answer_token_density" not in names:
            self.conn.execute(f"ALTER TABLE {self.result_table} ADD COLUMN answer_token_density REAL")
        if "noise_density" not in names:
            self.conn.execute(f"ALTER TABLE {self.result_table} ADD COLUMN noise_density REAL")
        if "latency_sec" not in names:
            self.conn.execute(f"ALTER TABLE {self.result_table} ADD COLUMN latency_sec REAL")
        self.conn.commit()

    def run_exists(self, run_id: str) -> bool:
        row = self.conn.execute(
            f"SELECT 1 FROM {self.run_table} WHERE run_id = ? LIMIT 1",
            (run_id,),
        ).fetchone()
        return row is not None

    def get_existing_question_ids(self, run_id: str) -> set[str]:
        rows = self.conn.execute(
            f"SELECT question_id FROM {self.result_table} WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        return {str(row["question_id"] or "").strip() for row in rows if str(row["question_id"] or "").strip()}

    def get_eval_result_rows(self, run_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            f"SELECT * FROM {self.result_table} WHERE run_id = ? ORDER BY id ASC",
            (run_id,),
        ).fetchall()

    def delete_group_results(self, run_id: str) -> None:
        self.conn.execute(f"DELETE FROM {self.group_table} WHERE run_id = ?", (run_id,))

    def log_eval_run_start(
        self, run_id: str, dataset_path: str, isolated: bool, commit: bool = True
    ) -> None:
        if not self.save_to_db:
            return
        self.conn.execute(
            f"""
            INSERT OR REPLACE INTO {self.run_table}
            (run_id, dataset_path, started_at, finished_at, total, matched, accuracy,
             final_answer_acc,
             retrieval_answer_span_hit_rate, retrieval_support_sentence_hit_rate, retrieval_evidence_hit_rate,
             graph_answer_span_hit_rate, graph_support_sentence_hit_rate,
             graph_ingest_accept_rate, avg_answer_token_density, avg_noise_density, avg_latency_sec,
             isolated)
            VALUES(?, ?, datetime('now'), NULL, 0, 0, 0.0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, ?)
            """,
            (run_id, dataset_path, int(isolated)),
        )
        if commit:
            self.conn.commit()

    def log_eval_result(
        self,
        run_id: str,
        question_id: str,
        question_type: str,
        question: str,
        expected_answer: str,
        prediction: str,
        is_match: bool,
        evidence_hit: bool | None = None,
        evidence_recall: float | None = None,
        answer_span_hit: bool | None = None,
        support_sentence_hit: bool | None = None,
        graph_answer_span_hit: bool | None = None,
        graph_support_sentence_hit: bool | None = None,
        answer_token_density: float | None = None,
        noise_density: float | None = None,
        latency_sec: float | None = None,
        retrieved_session_ids: Sequence[str] | None = None,
        commit: bool = True,
    ) -> None:
        if not self.save_to_db:
            return
        session_ids_json = json.dumps(list(retrieved_session_ids or []))
        self.conn.execute(
            f"""
            INSERT INTO {self.result_table}
            (
              run_id, question_id, question_type, question,
              expected_answer, prediction, is_match,
              evidence_hit, evidence_recall, answer_span_hit, support_sentence_hit,
              graph_answer_span_hit, graph_support_sentence_hit, answer_token_density, noise_density, latency_sec,
              retrieved_session_ids
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                question_id,
                question_type,
                question,
                expected_answer,
                prediction,
                int(is_match),
                (int(bool(evidence_hit)) if evidence_hit is not None else None),
                (float(evidence_recall) if evidence_recall is not None else None),
                (int(bool(answer_span_hit)) if answer_span_hit is not None else None),
                (int(bool(support_sentence_hit)) if support_sentence_hit is not None else None),
                (int(bool(graph_answer_span_hit)) if graph_answer_span_hit is not None else None),
                (
                    int(bool(graph_support_sentence_hit))
                    if graph_support_sentence_hit is not None
                    else None
                ),
                (float(answer_token_density) if answer_token_density is not None else None),
                (float(noise_density) if noise_density is not None else None),
                (float(latency_sec) if latency_sec is not None else None),
                session_ids_json,
            ),
        )
        if commit:
            self.conn.commit()

    def log_eval_group_result(
        self,
        run_id: str,
        group_key: str,
        total: int,
        matched: int,
        accuracy: float,
        commit: bool = True,
    ) -> None:
        if not self.save_to_db:
            return
        self.conn.execute(
            f"""
            INSERT INTO {self.group_table}
            (run_id, group_key, total, matched, accuracy)
            VALUES(?, ?, ?, ?, ?)
            """,
            (run_id, group_key, int(total), int(matched), float(accuracy)),
        )
        if commit:
            self.conn.commit()

    def log_thesis_run_metrics(
        self,
        run_id: str,
        *,
        final_answer_acc: float,
        retrieval_answer_span_hit_rate: float | None,
        retrieval_support_sentence_hit_rate: float | None,
        graph_answer_span_hit_rate: float | None,
        graph_support_sentence_hit_rate: float | None,
        graph_ingest_accept_rate: float | None,
        avg_answer_token_density: float | None,
        avg_noise_density: float | None,
        avg_latency_sec: float | None,
        commit: bool = True,
    ) -> None:
        if not self.save_to_db:
            return
        self.conn.execute(
            f"""
            INSERT OR REPLACE INTO {self.thesis_run_table}
            (
              run_id,
              final_answer_acc,
              retrieval_answer_span_hit_rate,
              retrieval_support_sentence_hit_rate,
              graph_answer_span_hit_rate,
              graph_support_sentence_hit_rate,
              graph_ingest_accept_rate,
              avg_answer_token_density,
              avg_noise_density,
              avg_latency_sec
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                float(final_answer_acc),
                float(retrieval_answer_span_hit_rate) if retrieval_answer_span_hit_rate is not None else None,
                float(retrieval_support_sentence_hit_rate) if retrieval_support_sentence_hit_rate is not None else None,
                float(graph_answer_span_hit_rate) if graph_answer_span_hit_rate is not None else None,
                float(graph_support_sentence_hit_rate) if graph_support_sentence_hit_rate is not None else None,
                float(graph_ingest_accept_rate) if graph_ingest_accept_rate is not None else None,
                float(avg_answer_token_density) if avg_answer_token_density is not None else None,
                float(avg_noise_density) if avg_noise_density is not None else None,
                float(avg_latency_sec) if avg_latency_sec is not None else None,
            ),
        )
        if commit:
            self.conn.commit()

    def log_thesis_type_metric(
        self,
        run_id: str,
        question_type: str,
        *,
        type_answer_acc: float,
        type_answer_token_density: float | None,
        type_noise_density: float | None,
        type_latency_sec: float,
        commit: bool = True,
    ) -> None:
        if not self.save_to_db:
            return
        self.conn.execute(
            f"""
            INSERT OR REPLACE INTO {self.thesis_type_table}
            (run_id, question_type, type_answer_acc, type_answer_token_density, type_noise_density, type_latency_sec)
            VALUES(?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                question_type,
                float(type_answer_acc),
                float(type_answer_token_density) if type_answer_token_density is not None else None,
                float(type_noise_density) if type_noise_density is not None else None,
                float(type_latency_sec),
            ),
        )
        if commit:
            self.conn.commit()

    def log_thesis_mode_run(
        self,
        *,
        dataset_name: str,
        mode: str,
        run_id: str,
        model_name: str,
        judge_model: str = "",
        commit: bool = True,
    ) -> None:
        if not self.save_to_db:
            return
        self.conn.execute(
            f"""
            INSERT OR REPLACE INTO {self.thesis_mode_table}
            (dataset_name, mode, run_id, model_name, judge_model, created_at)
            VALUES(?, ?, ?, ?, ?, datetime('now'))
            """,
            (
                str(dataset_name),
                str(mode),
                str(run_id),
                str(model_name),
                str(judge_model),
            ),
        )
        if commit:
            self.conn.commit()

    def get_latest_thesis_mode_run(
        self,
        *,
        dataset_name: str,
        mode: str,
    ) -> str | None:
        row = self.conn.execute(
            f"""
            SELECT run_id
            FROM {self.thesis_mode_table}
            WHERE dataset_name = ? AND mode = ?
            ORDER BY datetime(created_at) DESC, run_id DESC
            LIMIT 1
            """,
            (str(dataset_name), str(mode)),
        ).fetchone()
        if row is None:
            return None
        return str(row["run_id"])

    def log_eval_run_finish(
        self,
        run_id: str,
        total: int,
        matched: int,
        accuracy: float,
        retrieval_answer_span_hit_rate: float | None = None,
        retrieval_support_sentence_hit_rate: float | None = None,
        retrieval_evidence_hit_rate: float | None = None,
        graph_answer_span_hit_rate: float | None = None,
        graph_support_sentence_hit_rate: float | None = None,
        graph_ingest_accept_rate: float | None = None,
        avg_answer_token_density: float | None = None,
        avg_noise_density: float | None = None,
        avg_latency_sec: float | None = None,
        commit: bool = True,
    ) -> None:
        if not self.save_to_db:
            return
        self.conn.execute(
            f"""
            UPDATE {self.run_table}
            SET finished_at = datetime('now'),
                total = ?,
                matched = ?,
                accuracy = ?,
                final_answer_acc = ?,
                retrieval_answer_span_hit_rate = ?,
                retrieval_support_sentence_hit_rate = ?,
                retrieval_evidence_hit_rate = ?,
                graph_answer_span_hit_rate = ?,
                graph_support_sentence_hit_rate = ?,
                graph_ingest_accept_rate = ?,
                avg_answer_token_density = ?,
                avg_noise_density = ?,
                avg_latency_sec = ?
            WHERE run_id = ?
            """,
            (
                int(total),
                int(matched),
                float(accuracy),
                float(accuracy),
                (float(retrieval_answer_span_hit_rate) if retrieval_answer_span_hit_rate is not None else None),
                (
                    float(retrieval_support_sentence_hit_rate)
                    if retrieval_support_sentence_hit_rate is not None
                    else None
                ),
                (float(retrieval_evidence_hit_rate) if retrieval_evidence_hit_rate is not None else None),
                (float(graph_answer_span_hit_rate) if graph_answer_span_hit_rate is not None else None),
                (
                    float(graph_support_sentence_hit_rate)
                    if graph_support_sentence_hit_rate is not None
                    else None
                ),
                (float(graph_ingest_accept_rate) if graph_ingest_accept_rate is not None else None),
                (float(avg_answer_token_density) if avg_answer_token_density is not None else None),
                (float(avg_noise_density) if avg_noise_density is not None else None),
                (float(avg_latency_sec) if avg_latency_sec is not None else None),
                run_id,
            ),
        )
        if commit:
            self.conn.commit()
