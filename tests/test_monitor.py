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

    def test_image_cache_batch_size_and_fit(self):
        # Lines exactly as a vLLM 0.30 server exposes them, buckets included.
        text = (
            'vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.4507119763362455\n'
            'vllm:cache_config_info{block_size="16",engine="0",'
            'kv_cache_max_concurrency="14.197265625",num_gpu_blocks="14538"} 1.0\n'
            'vllm:mm_cache_queries_total{engine="0",model_name="m"} 2457.0\n'
            'vllm:mm_cache_hits_total{engine="0",model_name="m"} 263.0\n'
            'vllm:generation_tokens_total{engine="0",model_name="m"} 3.119415e+06\n'
            'vllm:iteration_tokens_total_bucket{engine="0",le="1.0",model_name="m"} 369686.0\n'
            'vllm:iteration_tokens_total_count{engine="0",model_name="m"} 563248.0\n'
            'vllm:iteration_tokens_total_sum{engine="0",model_name="m"} 1.1735316e+07\n'
            'vllm:iteration_tokens_total_created{engine="0",model_name="m"} 1.79e+09\n'
            'vllm:request_prompt_tokens_count{engine="0",model_name="m"} 2420.0\n'
            'vllm:request_prompt_tokens_sum{engine="0",model_name="m"} 9.40437e+06\n'
            'vllm:request_generation_tokens_sum{engine="0",model_name="m"} 3.087733e+06\n')
        s = monitor.snapshot(monitor.parse_samples(text))
        self.assertEqual(monitor.lifetime_mm_hit_rate(s), 0.107)
        self.assertEqual(monitor.lifetime_batch(s), 5.54)
        f = monitor.fit(s)
        self.assertEqual((f["per_request"], f["requests"], f["more_now"],
                          f["at_full_context"], f["source"]),
                         (5162, 45.1, 24.8, 14.2, "lifetime"))

        later = dict(s, mm_hits=s["mm_hits"] + 5, mm_queries=s["mm_queries"] + 10,
                     steps=s["steps"] + 100, step_tokens=s["step_tokens"] + 2000,
                     generation_tokens=s["generation_tokens"] + 800,
                     req_count=s["req_count"] + 3,
                     req_prompt_tokens=s["req_prompt_tokens"] + 12000,
                     req_output_tokens=s["req_output_tokens"] + 3000)
        r = monitor.rates(s, later, 2.0)
        self.assertEqual((r["mm_hit_rate"], r["batch"], r["tokens_per_step"],
                          r["tokens_per_request"]), (0.5, 8.0, 20.0, 5000))
        self.assertEqual(monitor.fit(later, r)["source"], "window")
        # A quiet window: nothing to hit, no batch, and fit falls back to lifetime.
        quiet = monitor.rates(s, dict(s), 2.0)
        self.assertIsNone(quiet["mm_hit_rate"])
        self.assertIsNone(quiet["batch"])
        self.assertEqual(monitor.fit(s, quiet)["source"], "lifetime")

    def test_fit_needs_the_cache_size_and_a_finished_request(self):
        self.assertIsNone(monitor.fit(monitor.snapshot(monitor.parse_samples(
            'vllm:request_prompt_tokens_count 1\nvllm:request_prompt_tokens_sum 10\n'
            'vllm:request_generation_tokens_sum 10\n'))))
        self.assertIsNone(monitor.fit(monitor.snapshot(monitor.parse_samples(SCRAPE))))

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

    def test_server_extract_watches_the_extraction_server(self):
        read = {"kind": "vllm", "url": "http://read:8000"}
        extract = {"kind": "vllm", "url": "http://extract:8000"}
        asked = []

        def get(url, timeout=None):
            asked.append(url)
            return _res(SCRAPE)

        with patch.object(app.backends, "known",
                          side_effect=lambda u=None: extract if "extract" in u else read),              patch.object(app.backends, "active_url", return_value=read["url"]),              patch.object(app.backends, "extract_url", return_value=extract["url"]),              patch.object(app.backends, "extract_separate", return_value=True),              patch.object(app, "MONITOR_GPU_URL", ""),              patch.object(app.requests, "get", side_effect=get):
            c = app.app.test_client()
            ocr = c.get("/api/monitor?server=ocr").get_json()
            ext = c.get("/api/monitor?server=extract").get_json()
            bad = c.get("/api/monitor?server=gpu")
        self.assertEqual(ocr["url"], read["url"])
        self.assertEqual(ext["url"], extract["url"])
        self.assertEqual(asked, ["http://read:8000/metrics", "http://extract:8000/metrics"])
        self.assertEqual(bad.status_code, 400)


if __name__ == "__main__":
    unittest.main()
