"""A page whose read loops is read again one Detail up: low -> medium -> original."""
import io
import json
import unittest
from unittest import mock

from PIL import Image

import app
import settings


def _upload(width=3000, height=2000) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buf, format="PNG")
    return buf.getvalue()


def _stream(loop_below: int):
    """A fake `stream_page` that loops on any image under `loop_below` pixels."""
    sizes = []

    def stream(image, stats=None, profile=None, cancel=None):
        sizes.append(image.size)
        looped = image.width * image.height < loop_below
        stats.update(looped=looped, new_tokens=7, seconds=1.5,
                     resolution=f"{image.width}x{image.height}")
        yield "ซ้ำ ซ้ำ ซ้ำ" if looped else "clean text"
    return stream, sizes


class LadderTests(unittest.TestCase):
    def test_next_detail_climbs_and_stops_at_original(self):
        self.assertEqual(settings.next_detail("low"), "medium")
        self.assertEqual(settings.next_detail("medium"), "original")
        self.assertIsNone(settings.next_detail("original"))


class ReadPageRetryTests(unittest.TestCase):
    def setUp(self):
        self.patches = [mock.patch.object(app, "TRIM_MARGINS", False),
                        mock.patch.object(app, "LOOP_RETRY", True)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def read(self, detail, loop_below, data=None):
        pages, _, job, _ = app.prepare_input(data or _upload(), detail)
        stream, sizes = _stream(loop_below)
        stats = {}
        with mock.patch.object(app, "stream_page", stream):
            text = app.read_page(pages[0], stats)
        return text, stats, sizes, job

    def test_a_loop_at_low_is_read_again_at_medium(self):
        text, stats, sizes, job = self.read("low", loop_below=3_000_000)
        self.assertEqual(len(sizes), 2)
        self.assertLessEqual(sizes[0][0] * sizes[0][1], 2_000_000)
        self.assertGreater(sizes[1][0] * sizes[1][1], 3_000_000)
        self.assertIn("clean text", text)
        self.assertFalse(stats["looped"])
        self.assertEqual(stats["detail"], "medium")
        self.assertEqual([a["detail"] for a in stats["loop_retries"]], ["low"])
        self.assertEqual(stats["loop_retries"][0]["new_tokens"], 7)
        # The compare view's cached image is the one that was read.
        cached = Image.open(io.BytesIO(app.page_image(job, 0).get_data()))
        self.assertEqual(cached.size, sizes[1])

    def test_each_read_is_tagged_with_its_page_and_detail_for_the_dry_log(self):
        pages, _, _, _ = app.prepare_input(_upload(), "low")
        self.assertEqual(pages[0]._dry_tag, "upload p1/1 low")
        tags = []
        stream, _ = _stream(3_000_000)

        def tagged(image, stats=None, profile=None, cancel=None):
            tags.append(image._dry_tag)
            yield from stream(image, stats, profile, cancel)
        with mock.patch.object(app, "stream_page", tagged):
            app.read_page(pages[0], {})
        self.assertEqual(tags, ["upload p1/1 low", "upload p1/1 medium"])

    def test_a_page_that_keeps_looping_climbs_to_original_and_stops(self):
        text, stats, sizes, _ = self.read("low", loop_below=10**9)
        self.assertEqual(len(sizes), 3)
        self.assertEqual(sizes[2], (3000, 2000))
        self.assertTrue(stats["looped"])
        self.assertEqual(stats["detail"], "original")
        self.assertEqual([a["detail"] for a in stats["loop_retries"]], ["low", "medium"])

    def test_original_has_nowhere_to_go(self):
        _, stats, sizes, _ = self.read("original", loop_below=10**9)
        self.assertEqual(len(sizes), 1)
        self.assertTrue(stats["looped"])
        self.assertNotIn("loop_retries", stats)

    def test_a_clean_read_is_read_once(self):
        _, stats, sizes, _ = self.read("low", loop_below=0)
        self.assertEqual(len(sizes), 1)
        self.assertNotIn("loop_retries", stats)

    def test_switched_off(self):
        with mock.patch.object(app, "LOOP_RETRY", False):
            _, stats, sizes, _ = self.read("low", loop_below=10**9)
        self.assertEqual(len(sizes), 1)
        self.assertTrue(stats["looped"])

    def test_a_page_already_inside_the_lower_budget_is_not_sent_again(self):
        _, stats, sizes, _ = self.read("low", loop_below=10**9,
                                       data=_upload(1000, 1000))
        self.assertEqual(len(sizes), 1)
        self.assertTrue(stats["looped"])

    def test_the_summary_counts_the_retries(self):
        _, stats, _, job = self.read("low", loop_below=3_000_000)
        self.assertEqual(app.summarise([stats, {}], "low", 0, job)["loop_retries"], 1)


class StreamRouteRetryTests(unittest.TestCase):
    def test_the_stream_says_so_and_keeps_only_the_retry(self):
        with mock.patch.object(app, "TRIM_MARGINS", False), \
                mock.patch.object(app, "LOOP_RETRY", True):
            pages, detail, job, case = app.prepare_input(_upload(), "low")
        stream, sizes = _stream(3_000_000)
        with mock.patch.object(app, "stream_page", stream), \
                mock.patch.object(app, "_page_parallelism", return_value=1), \
                mock.patch.object(app, "prepare",
                                  return_value=(pages, detail, job, case, {})), \
                mock.patch.object(app, "log_run"):
            res = app.app.test_client().post("/api/ocr/stream", data={"extract": "0"})
            events = [json.loads(line) for line in res.get_data(as_text=True).splitlines()
                      if line.strip()]
        kinds = [e["event"] for e in events]
        retry = next(e for e in events if e["event"] == "loop_retry")
        self.assertEqual((retry["page"], retry["detail"]), (1, "medium"))
        self.assertLess(kinds.index("loop_retry"), kinds.index("page_done"))
        self.assertNotIn("looped", kinds)
        done = next(e for e in events if e["event"] == "done")
        self.assertEqual(done["text"].strip(), "clean text")
        self.assertEqual(done["loop_retries"], 1)
        self.assertFalse(done["looped"])


if __name__ == "__main__":
    unittest.main()
