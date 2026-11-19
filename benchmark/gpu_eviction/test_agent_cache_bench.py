#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only unit tests: run with python -m unittest discover -s benchmark/gpu_eviction."""
import unittest
from unittest.mock import patch

import agent_cache_bench as bench


class BenchTests(unittest.TestCase):
    def test_summary_missing_cached_tokens(self):
        rows = [bench.Sample('cold', 0, 0, 100, 30, 2, 300, None, 'ok')]
        result = bench.summary(rows)['cold']
        self.assertIsNone(result['cached_tokens_mean'])
        self.assertEqual(result['ttft_p50_ms'], 30)

    def test_agent_history_grows_and_preserves_generation(self):
        import argparse
        args = argparse.Namespace(endpoint='http://localhost:8000', model='mock',
                                  sessions=2, turns=2, concurrency=2, prefix_words=6,
                                  output_tokens=4, timeout=10, api_key='', think_seconds=0)
        prompts = []

        def mock_completion(url, model, prompt, output_tokens, timeout, api_key,
                            phase, session, turn):
            prompts.append((session, turn, prompt))
            return bench.Sample(phase, session, turn, 10, 5, 4, 20, None, 'ok'), 'generated'

        with patch.object(bench, 'completion', side_effect=mock_completion):
            samples, _ = bench.run_agents(args)
        self.assertEqual(len(samples), 4)
        later = [p for _, turn, p in prompts if turn == 2]
        self.assertEqual(len(later), 2)
        self.assertTrue(all('Assistant:generated' in p for p in later))

    def test_eviction_reports_unverified_without_usage(self):
        import argparse
        args = argparse.Namespace(endpoint='http://localhost:8000', model='mock',
                                  prefix_words=6, output_tokens=4, timeout=10,
                                  api_key='', pressure_prompts=1, settle_seconds=0,
                                  min_cached_tokens=128)
        with patch.object(bench, 'completion', return_value=(
                bench.Sample('test', 0, 0, 10, 5, 4, 20, None, 'ok'), 'ok')):
            samples, evidence = bench.run_eviction(args)
        self.assertEqual(len(samples), 4)
        self.assertEqual(evidence['gpu_eviction_verification'], 'unverified')


if __name__ == '__main__':
    unittest.main()
