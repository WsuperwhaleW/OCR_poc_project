"""A stress run outlives the page that started it (2026-10-27).

The run is stubbed at `_stress_run`, held mid-run by an event, so the test can
hang up on the stream while the run is still going and check that it finishes
anyway, saves to History, and can be followed again from any line.
"""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import app
import config
import runhistory


class StressBackgroundTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        patcher = patch.object(config, "LOG_DIR", Path(directory.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = app.app.test_client()
        info = {"kind": "ollama", "url": "http://extract", "model": "model",
                "text_available": True, "text_reason": ""}
        status = patch.object(app.backends, "extract_status", return_value=info)
        status.start()
        self.addCleanup(status.stop)
        self.release = threading.Event()
        self.ran_to_end = threading.Event()

        def held_run(*_):
            yield json.dumps({"event": "plan", "seed": 7}) + "\n"
            self.release.wait(5)
            yield json.dumps({"event": "progress", "done": 1, "total": 1}) + "\n"
            yield json.dumps({"event": "done", "seed": 7, "report": {"error_rate": 0}}) + "\n"
            self.ran_to_end.set()

        run = patch.object(app, "_stress_run", side_effect=held_run)
        run.start()
        self.addCleanup(run.stop)
        self.addCleanup(self.release.set)

    def start(self):
        return self.client.post("/api/stress/stream", buffered=False,
                                json={"mode": "extract", "concurrency": 1, "jobs": 1})

    def test_hanging_up_does_not_stop_the_run(self):
        response = self.start()
        run_id = response.headers["X-Stress-Run"]
        first = json.loads(next(response.response))
        self.assertEqual(first["event"], "plan")
        response.close()                            # the page went away
        self.release.set()
        self.assertTrue(self.ran_to_end.wait(5), "the run stopped with its follower")

        # A follower's stream ends only when the run has.
        replay = self.client.get(f"/api/stress/attach?id={run_id}&from=0")
        events = [json.loads(line) for line in replay.data.splitlines()]
        self.assertEqual([e["event"] for e in events], ["plan", "progress", "done"])
        active = self.client.get("/api/stress/active").get_json()["run"]
        self.assertEqual(active["id"], run_id)
        self.assertFalse(active["running"])
        saved = runhistory.get(events[-1]["history_id"])
        self.assertEqual(saved["status"], "completed")

        rest = self.client.get(f"/api/stress/attach?id={run_id}&from=1")
        self.assertEqual([json.loads(line)["event"] for line in rest.data.splitlines()],
                         ["progress", "done"])

    def test_a_second_run_is_refused_while_the_first_goes_on_unwatched(self):
        response = self.start()
        next(response.response)
        response.close()
        self.assertEqual(self.start().status_code, 409)
        self.release.set()
        self.assertTrue(self.ran_to_end.wait(5))

    def test_attaching_to_a_run_that_is_not_kept_is_refused(self):
        response = self.start()
        self.release.set()
        response.get_data()
        self.assertEqual(self.client.get("/api/stress/attach?id=nope").status_code, 404)
        self.assertEqual(self.client.get("/api/stress/attach?from=x").status_code, 400)


if __name__ == "__main__":
    unittest.main()
