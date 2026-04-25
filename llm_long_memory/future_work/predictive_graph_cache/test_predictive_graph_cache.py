"""Tests for predictive graph cache storage, compatibility, and eviction."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from llm_long_memory.future_work.predictive_graph_cache.predictive_graph_cache import (
    PredictiveGraphCacheEntry,
    PredictiveGraphCacheIndex,
    PredictiveGraphCacheMaintenance,
    PredictiveQueryGenerator,
    PredictiveGraphCacheStore,
)


class _FakeMidMemory:
    def __init__(self) -> None:
        self.embedding_dim = 4
        self._chunks = {
            1: {"chunk_id": 1, "text": "(user) Current location is Boston.", "session_id": "s1"},
            2: {"chunk_id": 2, "text": "(user) It lasted one week.", "session_id": "s1"},
        }

    def get_existing_chunk_ids(self, chunk_ids):
        return [int(cid) for cid in chunk_ids if int(cid) in self._chunks]

    def get_chunks_by_ids(self, chunk_ids):
        return [self._chunks[int(cid)] for cid in chunk_ids if int(cid) in self._chunks]

    def get_current_step(self):
        return 2


class _FakeManager:
    def __init__(self) -> None:
        self.mid_memory = _FakeMidMemory()
        self.predictive_graph_cache_metrics = {}
        self.llm = type("FakeLLMObj", (), {"host": "http://127.0.0.1:11434", "model_name": "fake-model"})()
        self.config = {"llm": {"host": "http://127.0.0.1:11434"}}

    @staticmethod
    def _chunks_to_context_text(chunks):
        return "\n\n".join(f"[Chunk {idx}]\n{item['text']}" for idx, item in enumerate(chunks, start=1))

    def _build_query_plan(self, query):
        low = str(query).lower()
        if "how long" in low or "weeks" in low:
            return {
                "answer_type": "temporal_count",
                "entities": ["trip"],
                "focus_phrases": ["how long"],
                "time_terms": ["week"],
                "need_latest_state": False,
            }
        if "now" in low or "current" in low:
            return {
                "answer_type": "update",
                "entities": ["painting"],
                "focus_phrases": ["painting location"],
                "time_terms": [],
                "need_latest_state": True,
            }
        return {
            "answer_type": "count",
            "entities": ["items"],
            "focus_phrases": ["items count"],
            "time_terms": [],
            "need_latest_state": False,
        }


class TestPredictiveGraphCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = f"{self.tmp.name}/predictive_cache.sqlite"
        self.store = PredictiveGraphCacheStore(self.db_path, embedding_dim=4)
        self.manager = _FakeManager()
        self.index = PredictiveGraphCacheIndex(
            self.manager,
            self.store,
            {
                "top_k": 5,
                "semantic_similarity_min": 0.10,
                "entity_overlap_min": 0.34,
                "focus_overlap_min": 0.34,
                "quality_min_top_evidence_score": 0.30,
                "quality_min_selected_evidence": 1,
                "quality_min_selected_claims": 1,
                "quality_min_structural_edges": 1,
            },
        )

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def _entry(self, **overrides):
        now = datetime.now(timezone.utc)
        base = PredictiveGraphCacheEntry(
            cache_id="cache-1",
            window_id="s1:1-2",
            anticipated_query="How many items do I currently have?",
            anticipated_query_embedding=[1.0, 0.0, 0.0, 0.0],
            answer_type="count",
            query_plan_summary={
                "focus_phrases": ["items count"],
                "entities": ["items"],
                "state_keys": ["count"],
                "temporal_anchors": [],
                "time_range": "",
                "need_latest_state": False,
                "target_object": "items",
            },
            retrieval_snapshot={
                "retrieved_session_ids": ["s1"],
                "source_chunk_ids": [1],
                "build_timestamp": now.isoformat(),
                "source_version": 2,
            },
            filtered_pack={"core_evidence": [{"text": "I have four items.", "score": 0.9}]},
            claim_result={"claims": [{"subject": "items", "predicate": "count", "value": "4"}]},
            light_graph={"nodes": [], "edges": [{"subject": "items", "predicate": "count", "value": "4"}]},
            toolkit_payload={},
            quality_signals={
                "top_evidence_score": 0.9,
                "selected_evidence_count": 1,
                "selected_claim_count": 1,
                "graph_structural_edge_count": 1,
                "graph_total_latency_sec": 3.5,
                "toolkit_activated": False,
            },
            cache_stats={
                "hit_count": 0,
                "last_hit_time": "",
                "last_build_time": now.isoformat(),
                "source_version": 2,
            },
            expiry_metadata={
                "ttl_deadline": (now + timedelta(hours=24)).isoformat(),
                "invalidated": False,
                "invalidation_reason": "",
            },
        )
        for key, value in overrides.items():
            setattr(base, key, value)
        return base

    def test_schema_round_trip(self):
        entry = self._entry()
        self.store.upsert(entry)
        rows = self.store.fetch_all()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].cache_id, entry.cache_id)
        self.assertEqual(rows[0].answer_type, "count")
        self.assertEqual(rows[0].retrieval_snapshot["source_chunk_ids"], [1])

    def test_compatibility_rejects_count_vs_temporal_count(self):
        self.store.upsert(self._entry())
        with patch(
            "llm_long_memory.future_work.predictive_graph_cache.predictive_graph_cache.embed",
            return_value=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ):
            hit = self.index.lookup("How many weeks did the trip last?")
        self.assertIsNone(hit)

    def test_compatibility_rejects_current_vs_past_mismatch(self):
        entry = self._entry(
            answer_type="update",
            anticipated_query="Where was the painting before?",
            anticipated_query_embedding=[1.0, 0.0, 0.0, 0.0],
            query_plan_summary={
                "focus_phrases": ["painting location"],
                "entities": ["painting"],
                "state_keys": ["location"],
                "temporal_anchors": [],
                "time_range": "",
                "need_latest_state": False,
                "target_object": "painting",
            },
        )
        self.store.upsert(entry)
        with patch(
            "llm_long_memory.future_work.predictive_graph_cache.predictive_graph_cache.embed",
            return_value=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ):
            hit = self.index.lookup("Where is the painting now?")
        self.assertIsNone(hit)

    def test_compatibility_rejects_expired_entry(self):
        now = datetime.now(timezone.utc)
        entry = self._entry(
            expiry_metadata={
                "ttl_deadline": (now - timedelta(hours=1)).isoformat(),
                "invalidated": False,
                "invalidation_reason": "",
            }
        )
        self.store.upsert(entry)
        with patch(
            "llm_long_memory.future_work.predictive_graph_cache.predictive_graph_cache.embed",
            return_value=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ):
            hit = self.index.lookup("How many items do I have?")
        self.assertIsNone(hit)

    def test_maintenance_prunes_redundant_window_entries(self):
        first = self._entry(cache_id="cache-1")
        second = self._entry(cache_id="cache-2")
        second.cache_stats["hit_count"] = 2
        third = self._entry(cache_id="cache-3")
        third.quality_signals["top_evidence_score"] = 0.2
        self.store.upsert(first)
        self.store.upsert(second)
        self.store.upsert(third)
        worker = PredictiveGraphCacheMaintenance(
            self.store,
            {"max_entries_per_window": 2, "low_value_max_age_hours": 336},
        )
        result = worker.run()
        self.assertEqual(result["deleted_entries"], 1)
        ids = {row.cache_id for row in self.store.fetch_all()}
        self.assertNotIn("cache-3", ids)

    def test_generator_uses_type_conditioned_structured_plans(self):
        manager = _FakeManager()
        generator = PredictiveQueryGenerator(
            manager,
            {
                "enabled": True,
                "max_queries_per_window": 3,
                "max_candidates_per_type": 2,
                "target_answer_types": ["update", "count", "temporal"],
            },
        )

        def _fake_generate(*, prompt, **kwargs):
            if "answer_type must be exactly: update" in prompt:
                return """
                {"plans":[
                  {"target_object":"painting","state_key":"location","temporal_mode":"current","answer_shape":"entity","entities":["painting"],"focus_phrases":["painting location"]},
                  {"target_object":"best practices for moving paintings","state_key":"location","temporal_mode":"current","answer_shape":"entity","entities":["painting"],"focus_phrases":["painting advice"]}
                ]}
                """
            if "answer_type must be exactly: count" in prompt:
                return """
                {"plans":[
                  {"target_object":"projects","state_key":"count","temporal_mode":"current","answer_shape":"number","entities":["projects"],"focus_phrases":["projects count"]}
                ]}
                """
            return """
            {"plans":[
              {"target_object":"exchange program acceptance","state_key":"acceptance time","temporal_mode":"duration","answer_shape":"duration","entities":["exchange program"],"focus_phrases":["acceptance duration"]}
            ]}
            """

        with patch(
            "llm_long_memory.future_work.predictive_graph_cache.predictive_graph_cache.ollama_generate_with_retry",
            side_effect=_fake_generate,
        ):
            out = generator.generate_for_window(
                window_id="s1:1-2",
                chunks=[
                    {"text": "I moved the painting to my bedroom."},
                    {"text": "I am leading three projects right now."},
                    {"text": "I was accepted to the exchange program one week before orientation."},
                ],
            )
        self.assertEqual(len(out), 3)
        by_type = {item["answer_type"]: item["query"] for item in out}
        self.assertEqual(set(by_type.keys()), {"update", "count", "temporal"})
        self.assertEqual(by_type["update"], "What is the current location of painting?")
        self.assertEqual(by_type["count"], "How many projects do I currently have?")
        self.assertEqual(by_type["temporal"], "How long did exchange program acceptance last?")


if __name__ == "__main__":
    unittest.main()
