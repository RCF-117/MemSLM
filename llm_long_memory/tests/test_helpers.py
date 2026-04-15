"""Tests for config loading helpers."""

from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from llm_long_memory.utils.helpers import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class TestHelpers(unittest.TestCase):
    def test_load_config_returns_dict(self) -> None:
        cfg = load_config(str(CONFIG_PATH))
        self.assertIsInstance(cfg, dict)
        self.assertIn("memory", cfg)
        self.assertIn("retrieval", cfg)

    def test_load_config_deepcopy(self) -> None:
        cfg1 = load_config(str(CONFIG_PATH))
        cfg2 = load_config(str(CONFIG_PATH))
        self.assertIsNot(cfg1, cfg2)
        cfg1["memory"]["short_memory_size"] = 999
        self.assertNotEqual(cfg1["memory"]["short_memory_size"], cfg2["memory"]["short_memory_size"])

    def test_missing_required_section_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "bad.yaml"
            config_path.write_text("memory: {}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(str(config_path))

    def test_minimal_valid_config_loads(self) -> None:
        content = textwrap.dedent(
            """
            memory:
              short_memory_size: 10
              mid_memory:
                max_size: 10
            retrieval:
              top_k: 5
              chunks_per_topic: 2
              hybrid_alpha: 0.6
              keyword_weight: 0.2
              answering:
                context_only: true
                log_decision_details: false
                evidence_top_n_chunks: 5
                evidence_top_n_sentences: 5
                evidence_sentence_max_chars: 200
                candidate_top_n: 3
                llm_fallback_to_top_candidate: true
                fallback_min_score: 0.5
                response_evidence_min_token_overlap: 0.5
                response_evidence_min_shared_tokens: 2
                span_min_tokens: 1
                span_max_tokens: 3
                span_top_n_per_sentence: 5
                intent_extraction:
                  enabled: true
                  time_keywords: []
                  number_keywords: []
                  location_keywords: []
                  name_keywords: []
                  time_regexes: []
                  number_regexes: []
                  capitalized_phrase_max_tokens: 3
                candidate_scoring:
                  min_score: 0.2
                  reject_tokens: []
            embedding:
              dim: 8
              model: dummy
            logging:
              log_file: logs/test.log
              level: INFO
              console_enabled: false
            llm:
              default_model: qwen3:8b
              supported_models: [qwen3:8b]
              host: http://127.0.0.1:11434
              temperature: 0.2
              request_timeout_sec: 10
            dataset:
              stream_mode: true
            evaluation:
              save_to_db: false
              run_table: eval_runs
              result_table: eval_results
              group_table: eval_group_results
            """
        ).strip()
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "ok.yaml"
            config_path.write_text(content + "\n", encoding="utf-8")
            cfg = load_config(str(config_path))
            self.assertEqual(cfg["embedding"]["dim"], 8)


if __name__ == "__main__":
    unittest.main()
