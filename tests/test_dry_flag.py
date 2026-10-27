"""Whether vLLM's DRY fired on a page: replayed from its token ids, flagged in the stress test."""
import json
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import app
import dryreplay
import stress

VLLM = {"available": True, "reachable": True, "kind": "vllm", "url": "http://v",
        "model": "m"}
CHAT = "http://v/v1/chat/completions"
XARGS = {"dry_multiplier": 0.8, "dry_base": 1.75, "dry_allowed_length": 2,
         "dry_penalty_last_n": -1}


def reference(tokens, allowed, last_n):
    """Brute-force DRY (no breakers): the tokens penalised after `tokens`."""
    lo = 0 if last_n < 0 else max(0, len(tokens) - last_n)
    seq = tokens[lo:]
    hit = set()
    for i in range(len(seq) - 1):
        if seq[i] != seq[-1]:
            continue
        length = 1
        while i - length >= 0 and seq[i - length] == seq[len(seq) - 1 - length]:
            length += 1
        if length >= allowed:
            hit.add(seq[i + 1])
    return hit


class ReplayTests(unittest.TestCase):
    def test_matches_brute_force_on_random_sequences(self):
        rng = random.Random(0)
        for trial in range(300):
            allowed, last_n = rng.choice([1, 2, 3, 5]), rng.choice([-1, 5, 12, 40])
            history = []
            while len(history) < rng.randint(1, 80):
                if history and rng.random() < 0.4:
                    start = rng.randrange(len(history))
                    history.extend(history[start:start + rng.randint(1, 8)])
                else:
                    history.append(rng.randrange(6))
            fired = steered = 0
            first = None
            for n, token in enumerate(history):
                hit = reference(history[:n], allowed, last_n) if n else set()
                if hit:
                    fired += 1
                    first = n if first is None else first
                    steered += token not in hit
            got = dryreplay.replay(history, {"dry_allowed_length": allowed,
                                             "dry_penalty_last_n": last_n})
            self.assertEqual((got["fired"], got["first"], got["steered"]),
                             (fired, first, steered), (trial, history, allowed, last_n))

    def test_a_long_repeat_fires_and_a_short_one_does_not(self):
        line = list(range(10, 20))
        quiet = dryreplay.replay(line + line[:3], {"dry_allowed_length": 4})
        self.assertEqual((quiet["fired"], quiet["longest"]), (0, 3))
        loud = dryreplay.replay(line + line[:6] + [99], {"dry_allowed_length": 4})
        # Penalised at positions 14, 15 and 16; at 16 the reply left the repeat.
        self.assertEqual((loud["fired"], loud["first"], loud["steered"]), (3, 14, 1))
        self.assertIn("penalised 3 step(s) from output token 14", dryreplay.describe(loud))
        self.assertIn("not triggered", dryreplay.describe(quiet))


class _Stream:
    def __init__(self, chunks):
        self.status_code, self.text = 200, ""
        lines = [{"choices": [c]} for c in chunks]
        lines.append({"choices": [], "usage": {"completion_tokens": len(chunks)}})
        self.lines = [("data: " + json.dumps(o)).encode() for o in lines] + [b"data: [DONE]"]

    def iter_lines(self):
        yield from self.lines

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


class StreamPageTests(unittest.TestCase):
    def read(self, chunks, xargs=XARGS):
        sent = {}

        def post(url, status, json=None, **kwargs):
            sent.update(json)
            return _Stream(chunks)
        page = app.Image.new("RGB", (64, 64), "white")
        stats = {}
        with patch.object(app, "llama_status", return_value=VLLM), \
             patch.object(app, "chat_url", return_value=CHAT), \
             patch.object(app, "_batcher_for", return_value=None), \
             patch.object(app, "loop_guard", return_value=False), \
             patch.object(app.backends, "model_window", return_value=None), \
             patch.object(app.backends.settings, "vllm_xargs", return_value=xargs), \
             patch.object(app.servertime, "_fetch", return_value=None), \
             patch.object(app.backends, "http_post", post):
            "".join(app.stream_page(page, stats))
        return sent, stats

    def test_the_token_ids_vllm_streams_back_are_replayed(self):
        ids = [5, 6, 7, 5, 6, 8]
        chunks = [{"delta": {"content": "x"}, "token_ids": [t]} for t in ids]
        sent, stats = self.read(chunks)
        self.assertTrue(sent["return_token_ids"])
        self.assertEqual(stats["dry"], dryreplay.replay(ids, XARGS))
        self.assertEqual(stats["dry"]["fired"], 1)

    def test_a_server_that_sends_no_token_ids_is_unchecked_not_clean(self):
        _, stats = self.read([{"delta": {"content": "x"}}])
        self.assertEqual(stats["dry"], {"checked": False})

    def test_nothing_is_asked_or_replayed_with_dry_off(self):
        sent, stats = self.read([{"delta": {"content": "x"}}], xargs={})
        self.assertNotIn("return_token_ids", sent)
        self.assertNotIn("dry", stats)


def _page(detail):
    page = SimpleNamespace()
    page._stress_origin = {"case": "sol003", "source": "benchmark",
                           "index": 0, "detail": detail}
    return page


class StressFlagTests(unittest.TestCase):
    def _read(self, outcomes):
        """`outcomes`: (looped, dry fired) of each read, in order."""
        calls = []

        def read(page, stats, cancel=None):
            looped, fired = outcomes[len(calls)]
            calls.append(page._stress_origin["detail"])
            stats.update(looped=looped, new_tokens=10, seconds=1.0,
                         dry={"checked": True, "fired": fired, "first": 300 if fired else None,
                              "steered": int(bool(fired)), "longest": 260 if fired else 40,
                              "allowed": 256, "tokens": 900})
            return "text"

        def up(page):
            nxt = app.next_detail(page._stress_origin["detail"])
            return _page(nxt) if nxt else None
        live, rec = app._StressLive(1), {}
        with patch.object(app, "read_page", side_effect=read), \
                patch.object(app, "_stress_retry_page", side_effect=up):
            app._stress_read([_page("low")], app.MultiCancel(), rec, live,
                             job=1, case="sol003")
        return rec, live.take_alerts()

    def test_dry_on_a_clean_page_is_one_row(self):
        with patch.object(app, "STRESS_DRY_RETRY", False):   # flag only
            rec, alerts = self._read([(False, 4)])
        self.assertEqual([a["kind"] for a in alerts], ["dry"])
        self.assertEqual(alerts[0]["page"], 1)
        self.assertIn("at low: penalised 4 step(s) from output token 300", alerts[0]["message"])
        self.assertEqual((rec["dry_pages"], rec["dry_checked"]), (1, 1))

    def test_dry_across_retries_stays_one_row_and_grows(self):
        # medium fired too, so it climbs again (STRESS_DRY_RETRY)
        rec, alerts = self._read([(True, 3), (False, 2), (False, 0)])
        dry = [a for a in alerts if a["kind"] == "dry"]
        self.assertEqual(len({a["id"] for a in dry}), 1)
        self.assertIn("at low:", dry[-1]["message"])
        self.assertIn("; at medium:", dry[-1]["message"])
        self.assertEqual(rec["dry_pages"], 1)              # a page, not an attempt

    def test_no_dry_no_row(self):
        rec, alerts = self._read([(False, 0)])
        self.assertEqual(alerts, [])
        self.assertEqual((rec["dry_pages"], rec["dry_checked"]), (0, 1))

    def test_the_report_counts_dry_pages(self):
        records = [{"status": "ok", "pages": 2, "dry_pages": 1, "dry_checked": 2,
                    "total_s": 1.0, "ocr_s": 1.0}]
        out = stress.report(records, 1.0, "ocr", 1)
        self.assertEqual((out["ocr"]["dry_pages"], out["ocr"]["dry_checked"]), (1, 2))


def _png_uri(color):
    import base64
    import io
    buf = io.BytesIO()
    app.Image.new("RGB", (4, 4), color).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class EvidenceTests(unittest.TestCase):
    """Clicking a flagged row shows that page in Compare: every read, page and text."""

    def _read(self, outcomes):
        """`outcomes`: (looped, dry fired) of each read; each detail's page is a colour."""
        colours = {"low": "red", "medium": "green", "original": "blue"}

        def page_at(detail):
            page = _page(detail)
            page._data_uri = _png_uri(colours[detail])
            return page

        calls = []

        def read(page, stats, cancel=None):
            looped, fired = outcomes[len(calls)]
            detail = page._stress_origin["detail"]
            calls.append(detail)
            stats.update(looped=looped, new_tokens=10, seconds=1.0,
                         dry={"checked": True, "fired": fired, "first": 5, "steered": 0,
                              "longest": 300, "allowed": 256, "tokens": 9} if fired else None)
            return f"text read at {detail}"

        def up(page):
            nxt = app.next_detail(page._stress_origin["detail"])
            return page_at(nxt) if nxt else None
        live = app._StressLive(1, prefix="run-test-")
        with patch.object(app, "read_page", side_effect=read), \
                patch.object(app, "_stress_retry_page", side_effect=up):
            app._stress_read([page_at("low")], app.MultiCancel(), {}, live,
                             job=1, case="sol003")
        return live.take_alerts()

    def test_a_looped_page_keeps_every_read_under_its_alert(self):
        alerts = self._read([(True, 3), (False, 0)])
        loop = next(a for a in alerts if a["kind"] == "looped")
        dry = next(a for a in alerts if a["kind"] == "dry")
        self.assertTrue(loop["evidence"] and dry["evidence"])
        self.assertTrue(str(loop["id"]).startswith("run-test-"))
        client = app.app.test_client()
        got = client.get(f"/api/stress/evidence/{loop['id']}").get_json()
        self.assertEqual((got["case"], got["page"], got["truth_case"]), ("sol003", 1, "sol003"))
        self.assertEqual([(a["detail"], a["looped"], a["text"]) for a in got["attempts"]],
                         [("low", True, "text read at low"),
                          ("medium", False, "text read at medium")])
        self.assertEqual(got["attempts"][0]["dry"]["fired"], 3)
        # The DRY row on the same page shows the same reads.
        self.assertEqual(client.get(f"/api/stress/evidence/{dry['id']}").get_json()["attempts"],
                         got["attempts"])
        image = client.get(f"/api/stress/evidence/{loop['id']}/1.png")
        self.assertEqual(image.mimetype, "image/png")
        self.assertEqual(app.Image.open(__import__("io").BytesIO(image.data))
                         .getpixel((0, 0)), (0, 128, 0))          # the medium read's page

    def test_a_clean_page_keeps_nothing(self):
        self.assertEqual(self._read([(False, 0)]), [])

    def test_an_unknown_or_dropped_page_is_a_404(self):
        client = app.app.test_client()
        self.assertEqual(client.get("/api/stress/evidence/nope").status_code, 404)
        self.assertEqual(client.get("/api/stress/evidence/nope/0.png").status_code, 404)
        store = app._StressEvidence(2)
        for n in range(3):
            store.put(n, "c", 1, [], None)
        self.assertIsNone(store.get(0))                # the oldest is dropped
        self.assertEqual(store.get(2)["case"], "c")


class HistoryTests(unittest.TestCase):
    def test_an_alert_sent_again_replaces_its_row(self):
        events = [{"event": "alert", "id": "run-1", "kind": "looped", "state": "retrying"},
                  {"event": "alert", "id": "run-2", "kind": "dry"},
                  {"event": "alert", "id": "run-1", "kind": "looped", "state": "resolved"},
                  {"event": "done", "report": {}}]
        stream = (json.dumps(e) + "\n" for e in events)
        with patch.object(app.runhistory, "start", return_value=None), \
             patch.object(app, "_HistoryMonitor") as monitor:
            monitor.return_value.stop.return_value = ([], 0)
            out = [json.loads(line) for line in app._history_stream(stream, "stress", {})]
        done = next(e for e in out if e.get("event") == "done")
        alerts = done.get("alerts") or []
        self.assertEqual([(a["id"], a.get("state")) for a in alerts],
                         [("run-1", "resolved"), ("run-2", None)])


if __name__ == "__main__":
    unittest.main()
