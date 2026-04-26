"""Tests for evaluation runtime metric helpers."""

from __future__ import annotations

import unittest
from pathlib import Path

from llm_long_memory.evaluation.metrics_runtime import (
    compute_answer_token_density,
    compute_noise_density,
)
from llm_long_memory.utils.helpers import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class TestMetricsRuntime(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_config(str(CONFIG_PATH))["evaluation"]

    def test_answer_token_density_is_high_when_prompt_is_answer_heavy(self) -> None:
        chunks = [
            {"text": "The answer is my bedroom."},
            {"text": "my bedroom"},
        ]
        density = compute_answer_token_density("my bedroom", chunks, self.cfg)
        noise = compute_noise_density("my bedroom", chunks, self.cfg)
        self.assertGreater(density, 0.45)
        self.assertLess(noise, 0.55)

    def test_noise_density_is_high_when_answer_tokens_are_sparse(self) -> None:
        chunks = [
            {
                "text": (
                    "The retrieved evidence contains several unrelated details about the trip, the weather, "
                    "the planning steps, and various notes before mentioning my bedroom once."
                )
            }
        ]
        density = compute_answer_token_density("my bedroom", chunks, self.cfg)
        noise = compute_noise_density("my bedroom", chunks, self.cfg)
        self.assertLess(density, 0.2)
        self.assertGreater(noise, 0.8)


if __name__ == "__main__":
    unittest.main()
