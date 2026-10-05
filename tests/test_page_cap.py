"""`MAX_PAGES` truncates, and until 2026-09-16 it did so silently.

The failure it produced had no symptom anywhere: a hundred-page file came back as
ten pages with `status=ok` and `pages=10`, and a benchmark document was then
scored against the ground truth of all of it. It was live on `sol023`, which has
13 pages against a default cap of 10.

Four things, in the order the signal travels:

* `load_pages` reports the pages it READ and the pages the file HAS;
* `prepare_input` puts the total on the job, and every summariser reads it back
  -- one helper, so the three cannot report it differently;
* the preview says it BEFORE a run is paid for, which is the only place that can
  be said;
* `runlog._incomplete` counts a truncated read and never scores it, because its
  accuracy is against a document nobody read.
"""
import io
import pathlib
import unittest
from unittest.mock import patch

import app
import runlog

MOCKS = pathlib.Path("mockOcr")
SHORT = MOCKS / "invoice_sol002.pdf"          # 1 page
LONG = MOCKS / "multiple_sol023.pdf"          # 13 pages, over the default cap


def pdf(path):
    return path.read_bytes()


class LoadPagesTests(unittest.TestCase):
    def test_a_short_file_reads_whole_and_says_so(self):
        pages, total = app.load_pages(pdf(SHORT))
        self.assertEqual((len(pages), total), (1, 1))

    def test_a_long_file_reports_what_it_dropped(self):
        with patch.object(app, "MAX_PAGES", 10):
            pages, total = app.load_pages(pdf(LONG))
        self.assertEqual(len(pages), 10)
        self.assertEqual(total, 13)

    def test_raising_the_cap_reads_all_of_it(self):
        with patch.object(app, "MAX_PAGES", 200):
            pages, total = app.load_pages(pdf(LONG))
        self.assertEqual((len(pages), total), (13, 13))

    def test_no_cap_reads_every_page(self):
        """**0 is the default and means no cap** (2026-09-16, at the user's
        request: *no cap page*). The trap it replaced is why `_capped` exists:
        `min(total, 0)` and `zip(frames, range(0))` both read NOTHING rather
        than everything, and the first of those shipped for about a minute."""
        with patch.object(app, "MAX_PAGES", 0):
            pages, total = app.load_pages(pdf(LONG))
        self.assertEqual((len(pages), total), (13, 13))

    def test_capped_yields_all_of_it_at_zero(self):
        self.assertEqual(list(app._capped("abcde", 0)), list("abcde"))
        self.assertEqual(list(app._capped("abcde", 2)), list("ab"))
        self.assertEqual(list(app._capped("abcde", 99)), list("abcde"))
        self.assertEqual(list(app._capped("", 0)), [])

    def test_the_default_is_uncapped(self):
        import settings
        self.assertEqual(settings.MAX_PAGES, 0)

    def test_a_single_image_is_one_page_of_one(self):
        buf = io.BytesIO()
        from PIL import Image
        Image.new("RGB", (40, 40), "white").save(buf, format="PNG")
        pages, total = app.load_pages(buf.getvalue())
        self.assertEqual((len(pages), total), (1, 1))


class CoverageTests(unittest.TestCase):
    """What a summary carries, read back off the job the read registered."""

    def summary_for(self, path, cap):
        with patch.object(app, "MAX_PAGES", cap):
            prepared, detail, job, _case = app.prepare_input(
                pdf(path), "low", source={"kind": "test"})
        stats = [{"new_tokens": 1, "decode_seconds": 0.1} for _ in prepared]
        return app.summarise(stats, detail, 0.0, job)

    def test_a_truncated_read_reports_the_total_and_the_shortfall(self):
        s = self.summary_for(LONG, 10)
        self.assertEqual(s["page_count"], 10)
        self.assertEqual(s["pages_total"], 13)
        self.assertEqual(s["pages_truncated"], 3)

    def test_a_whole_read_reports_no_shortfall(self):
        s = self.summary_for(SHORT, 10)
        self.assertEqual((s["pages_total"], s["pages_truncated"]), (1, 0))

    def test_a_run_with_no_job_claims_nothing(self):
        """**Absent, not equal.** A re-extraction read no page, and a blank must
        not read as "nothing was dropped"."""
        s = app.summarise([{"new_tokens": 1}], "low", 0.0, None)
        self.assertNotIn("pages_total", s)
        self.assertNotIn("pages_truncated", s)

    def test_an_evicted_job_claims_nothing_rather_than_guessing(self):
        s = app.summarise([{"new_tokens": 1}], "low", 0.0, "nosuchjob")
        self.assertNotIn("pages_total", s)


class PreviewTests(unittest.TestCase):
    """The pre-run warning -- the only one that arrives before the OCR is paid for."""

    def setUp(self):
        self.client = app.app.test_client()

    def headers_for(self, form, cap=10):
        with patch.object(app, "MAX_PAGES", cap):
            res = self.client.post("/api/preview", data=form)
        if res.status_code != 200:
            # Only decode on failure: a success returns a PNG, and assertEqual
            # builds its message whether or not it needs it.
            self.fail("%s: %s" % (res.status_code,
                                  res.get_data(as_text=False)[:120]))
        return res.headers

    def test_a_long_file_warns_before_it_is_read(self):
        h = self.headers_for({"case": "sol023", "detail": "low"})
        self.assertEqual(h["X-Preview-Pages"], "10")
        self.assertEqual(h["X-Preview-Pages-Total"], "13")
        self.assertEqual(h["X-Preview-Max-Pages"], "10")

    def test_uncapped_never_warns_however_long_the_file(self):
        h = self.headers_for({"case": "sol023", "detail": "low"}, cap=0)
        self.assertEqual(h["X-Preview-Pages"], "13")
        self.assertEqual(h["X-Preview-Pages-Total"], "13")
        self.assertEqual(h["X-Preview-Max-Pages"], "0")

    def test_a_short_file_has_nothing_to_warn_about(self):
        h = self.headers_for({"case": "sol002", "detail": "low"})
        self.assertEqual(h["X-Preview-Pages"], h["X-Preview-Pages-Total"])

    def test_an_upload_is_measured_the_same_way(self):
        data = pdf(LONG)
        with patch.object(app, "MAX_PAGES", 10):
            res = self.client.post(
                "/api/preview",
                data={"detail": "low", "image": (io.BytesIO(data), "x.pdf")},
                content_type="multipart/form-data")
        self.assertEqual(res.headers["X-Preview-Pages-Total"], "13")


class RunLogTests(unittest.TestCase):
    """Counted, never scored -- the treatment a loop already gets."""

    def test_a_truncated_read_is_incomplete(self):
        self.assertTrue(runlog._incomplete(
            {"status": "ok", "pages": "10", "pages_total": "13"}))

    def test_a_whole_read_is_not(self):
        self.assertFalse(runlog._incomplete(
            {"status": "ok", "pages": "13", "pages_total": "13"}))

    def test_a_row_written_before_the_column_is_not_condemned(self):
        """**Blank is not a claim that nothing was dropped.** Every row in the
        log predates this column, and none of them may be re-labelled by it."""
        self.assertFalse(runlog._incomplete(
            {"status": "ok", "pages": "10", "pages_total": ""}))
        self.assertFalse(runlog._incomplete({"status": "ok", "pages": "10"}))

    def test_rubbish_in_the_columns_is_not_a_truncation(self):
        self.assertFalse(runlog._incomplete(
            {"status": "ok", "pages": "ten", "pages_total": "13"}))

    def test_the_column_is_at_the_end(self):
        """Appended, like every column before it: inserting mid-file re-labels
        every value to the right of it.

        It was the LAST column when this was written. Columns appended since
        come after it, which is the same property seen from the other side --
        so what is pinned is that nothing older than it has moved behind it."""
        columns = runlog.COLUMNS
        self.assertEqual(columns[columns.index("pages_total") - 1], "field_blame")
        self.assertEqual(columns[columns.index("pages_total") + 1:],
                         ["item_tables", "master_tables", "table_rows",
                          "table_realigned", "table_misaligned",
                          "table_cells_ok", "table_cells",
                          "master_ok", "master_scored", "extract_server",
                          "fix_moved", "fix_strays", "fix_stray_removals",
                          "fix_removals", "fix_values_removed",
                          "fix_cells_read", "fix_cells_agent", "fix_cells",
                          "server_seconds", "network_seconds", "server_timing",
                          "extract_server_seconds", "extract_network_seconds",
                          "extract_server_timing", "extract_parallel"])

    def test_a_failed_status_still_wins(self):
        self.assertTrue(runlog._incomplete(
            {"status": "looped", "pages": "13", "pages_total": "13"}))


if __name__ == "__main__":
    unittest.main()
