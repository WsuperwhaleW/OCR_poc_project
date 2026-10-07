"""The live /metrics monitor: what one scrape says, what two say, and the gate."""
import unittest
from unittest.mock import MagicMock, patch

import app
import monitor

SCRAPE = (
    'vllm:num_requests_running{engine="0"} 3\n'
    'vllm:num_requests_running{engine="1"} 1\n'
    'vllm:num_requests_waiting{engine="0"} 2\n'
    'vllm:kv_cache_usage_perc{engine="0"} 0.6\n'
    'vllm:kv_cache_usage_perc{engine="1"} 0.2\n'
    'vllm:cache_config_info{block_size="16",num_gpu_blocks="2000",engine="0"} 1\n'
    'vllm:cache_config_info{block_size="16",num_gpu_blocks="2000",engine="1"} 1\n'
    'vllm:generation_tokens_total 1000\n'
    'vllm:prompt_tokens_total 5000\n'
    'vllm:prefix_cache_queries_total 4000\n'
    'vllm:prefix_cache_hits_total 1000\n')


def _res(text, code=200):
    r = MagicMock(status_code=code)
    r.text = text
    return r


class SnapshotTests(unittest.TestCase):
    def test_engines_are_summed_for_counts_and_averaged_for_percentages(self):
        s = monitor.snapshot(monitor.parse_samples(SCRAPE))
        self.assertEqual(s["running"], 4)
        self.assertAlmostEqual(s["kv_usage"], 0.4)
        self.assertEqual(s["kv_tokens_total"], 64000)
        self.assertEqual(s["kv_tokens_free"], 38400)

    def test_v0_names_and_a_missing_block_count(self):
        s = monitor.snapshot(monitor.parse_samples(
            'vllm:gpu_cache_usage_perc 0.25\nvllm:gpu_prefix_cache_hit_rate 0.7\n'))
        self.assertEqual(s["kv_free"], 0.75)
        self.assertIsNone(s["kv_tokens_free"])
        self.assertEqual(monitor.lifetime_hit_rate(s), 0.7)

    def test_rates_and_a_restart(self):
        a = monitor.snapshot(monitor.parse_samples(SCRAPE))
        b = dict(a, generation_tokens=1500, prefix_queries=4100, prefix_hits=1080)
        r = monitor.rates(a, b, 5.0)
        self.assertEqual(r["generation_tps"], 100)
        self.assertEqual(r["prefix_hit_rate"], 0.8)
        self.assertIsNone(monitor.rates(b, a, 5.0)["generation_tps"])

    def test_observe_needs_two_scrapes_close_together(self):
        s = monitor.snapshot(monitor.parse_samples(SCRAPE))
        url = "http://test-observe"
        self.assertIsNone(monitor.observe(url, s, now=100.0))
        self.assertIsNotNone(monitor.observe(url, s, now=102.0))
        self.assertIsNone(monitor.observe(url, s, now=400.0))     # too far apart

    def test_gpu_from_either_exporter(self):
        g = monitor.gpu_memory(monitor.parse_samples(
            'nvidia_smi_memory_used_bytes{uuid="a"} 1073741824\n'
            'nvidia_smi_memory_total_bytes{uuid="a"} 4294967296\n'))
        self.assertEqual(g["free_mb"], 3072)
        self.assertIsNone(monitor.gpu_memory(monitor.parse_samples(SCRAPE)))


class RouteTests(unittest.TestCase):
    def setUp(self):
        app._monitor_cache.clear()

    def test_a_server_that_is_not_vllm_is_refused_and_not_asked(self):
        with patch.object(app.backends, "known", return_value={"kind": "ollama", "url": "u"}), \
             patch.object(app.backends, "extract_separate", return_value=False), \
             patch.object(app.requests, "get") as get:
            res = app.app.test_client().get("/api/monitor")
        self.assertEqual(res.status_code, 409)
        get.assert_not_called()

    def test_vllm_with_a_gpu_exporter(self):
        info = {"kind": "vllm", "url": "http://gpu-box:8000", "model": "m"}
        exporter = ('DCGM_FI_DEV_FB_USED{gpu="0"} 70000\n'
                    'DCGM_FI_DEV_FB_FREE{gpu="0"} 11920\n')

        def get(url, timeout=None):
            return _res(exporter if "9400" in url else SCRAPE)

        with patch.object(app.backends, "known", return_value=info), \
             patch.object(app.backends, "active_url", return_value=info["url"]), \
             patch.object(app.backends, "extract_separate", return_value=False), \
             patch.object(app.requests, "get", side_effect=get):
            d = app.app.test_client().get(
                "/api/monitor?gpu=http://gpu-box:9400/metrics").get_json()
        self.assertEqual(d["snapshot"]["running"], 4)
        self.assertEqual(d["gpu"]["free_mb"], 11920)
        self.assertEqual(d["gpu"]["source"], "DCGM")

    def test_remote_vllm_with_no_exporter_says_why_vram_is_unknown(self):
        info = {"kind": "vllm", "url": "http://gpu-box:8000"}
        with patch.object(app.backends, "known", return_value=info), \
             patch.object(app.backends, "active_url", return_value=info["url"]), \
             patch.object(app.backends, "extract_separate", return_value=False), \
             patch.object(app, "MONITOR_GPU_URL", ""), \
             patch.object(app.requests, "get", return_value=_res(SCRAPE)):
            d = app.app.test_client().get("/api/monitor").get_json()
        self.assertIsNone(d["gpu"])
        self.assertIn("vLLM port", d["gpu_reason"])

    def test_a_second_call_inside_the_min_interval_is_not_a_second_scrape(self):
        info = {'kind': 'vllm', 'url': 'http://gpu-box:8000'}
        with patch.object(app.backends, 'known', return_value=info), \
             patch.object(app.backends, 'active_url', return_value=info['url']), \
             patch.object(app.backends, 'extract_separate', return_value=False), \
             patch.object(app, 'MONITOR_GPU_URL', ''), \
             patch.object(app, 'MONITOR_MIN_INTERVAL', 60.0), \
             patch.object(app.requests, 'get', return_value=_res(SCRAPE)) as get:
            c = app.app.test_client()
            first = c.get('/api/monitor').get_json()
            second = c.get('/api/monitor').get_json()
        self.assertEqual(get.call_count, 1)
        self.assertFalse(first['cached'])
        self.assertTrue(second['cached'])
        self.assertEqual(first['at'], second['at'])
        self.assertEqual(second['min_interval'], 60.0)

    def test_a_failed_scrape_is_not_cached(self):
        info = {'kind': 'vllm', 'url': 'http://gpu-box:8000'}
        with patch.object(app.backends, 'known', return_value=info), \
             patch.object(app.backends, 'active_url', return_value=info['url']), \
             patch.object(app.backends, 'extract_separate', return_value=False), \
             patch.object(app, 'MONITOR_MIN_INTERVAL', 60.0), \
             patch.object(app.requests, 'get', return_value=_res('', 503)) as get:
            c = app.app.test_client()
            self.assertEqual(c.get('/api/monitor').status_code, 502)
            self.assertEqual(c.get('/api/monitor').status_code, 502)
        self.assertEqual(get.call_count, 2)


if __name__ == "__main__":
    unittest.main()
