"""The user page's Stress test pill: every running stress run, any workspace."""
import json
import unittest

import app


def line(event):
    return json.dumps(event) + "\n"


class StressStatusTests(unittest.TestCase):
    def tearDown(self):
        with app._stress_live_lock:
            app._stress_live.clear()

    def test_status_follows_the_events_the_run_sends(self):
        job = app._StressBackground({"secret": "body"}, "full")
        job.emit(line({"event": "plan", "mode": "full", "concurrency": 4, "jobs": 20,
                       "warmup": 1, "reader": "typhoon", "extractor": "gemma",
                       "server": {"url": "http://gpu:8000", "kind": "vllm"}}))
        job.emit(line({"event": "progress", "phase": "run", "total": 21, "done": 5,
                       "failed": 1}))
        job.emit(line({"event": "alert", "message": "ignored"}))
        status = job.public()
        self.assertEqual(status["phase"], "run")
        self.assertEqual((status["done"], status["total"], status["failed"]), (5, 21, 1))
        self.assertEqual(status["concurrency"], 4)
        self.assertEqual(status["server"], "http://gpu:8000")
        self.assertNotIn("request", status)
        self.assertNotIn("secret", json.dumps(status))
        job.emit(line({"event": "draining", "in_flight": 2}))
        self.assertEqual(job.public()["phase"], "draining")
        self.assertEqual(len(job.lines), 4)

    def test_route_lists_running_runs_only(self):
        client = app.app.test_client()
        self.assertEqual(client.get("/api/user/stress").get_json(),
                         {"running": False, "runs": []})
        running, done = app._StressBackground({}, "ocr"), app._StressBackground({}, "extract")
        done.finish()
        with app._stress_live_lock:
            app._stress_live.extend([running, done])
        body = client.get("/api/user/stress").get_json()
        self.assertTrue(body["running"])
        self.assertEqual([r["id"] for r in body["runs"]], [running.id])

    def test_background_run_registers_and_unregisters(self):
        seen = []

        def fake_generate(cfg, cancel):
            seen.append([r["id"] for r in app.stress_runs_live()])
            yield line({"event": "done"})

        job = app._StressBackground({}, "ocr")
        original = app._stress_generate
        app._stress_generate = fake_generate
        try:
            import threading
            app._stress_background_run(job, {}, threading.Event())
        finally:
            app._stress_generate = original
        self.assertEqual(seen, [[job.id]])
        self.assertEqual(app.stress_runs_live(), [])


if __name__ == "__main__":
    unittest.main()
