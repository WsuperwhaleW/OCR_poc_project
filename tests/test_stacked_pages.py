"""One image frame can hold several pages, and is read as several (2026-10-27).

A scanner that stitches a batch into a single tall TIFF frame shows one page in
a viewer, and `load_pages` read it as one: two or three pages of text in one
request, which ran into the token cap. `split_stacked_pages` cuts such a frame
at the blank band between its pages, and `load_pages` counts every piece as a
page -- in the cap and in the total alike.
"""
import io
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

import app

A4 = (1240, 1754)                     # 150 DPI


def page(size=A4, margin=120):
    """A white page with lines of 'text' from margin to margin."""
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    for y in range(margin, size[1] - margin, 40):
        draw.rectangle((margin, y, size[0] - margin, y + 18), fill="black")
    return image


def stack(pages, tall=True):
    """`pages` end to end in one image, the way a stitching scanner leaves them."""
    w = sum(p.width for p in pages) if not tall else max(p.width for p in pages)
    h = sum(p.height for p in pages) if tall else max(p.height for p in pages)
    out, at = Image.new("RGB", (w, h), "white"), 0
    for p in pages:
        out.paste(p, (0, at) if tall else (at, 0))
        at += p.height if tall else p.width
    return out


def tiff(frames):
    buf = io.BytesIO()
    frames[0].save(buf, format="TIFF", save_all=True, append_images=frames[1:],
                   compression="tiff_lzw")
    return buf.getvalue()


def blank_rows(image):
    """Is the first and last row of `image` free of ink -- a cut through margin?"""
    gray = image.convert("L")
    rows = (gray.crop((0, 0, image.width, 1)), gray.crop((0, image.height - 1,
                                                          image.width, image.height)))
    return all(r.getextrema()[0] > 200 for r in rows)


class SplitTests(unittest.TestCase):
    def test_a_real_page_is_left_alone(self):
        for size in (A4, A4[::-1], (1275, 1650)):          # portrait, landscape, Letter
            with self.subTest(size=size):
                image = page(size)
                self.assertEqual(app.split_stacked_pages(image), [image])

    def test_three_stacked_pages_come_back_as_three(self):
        pieces = app.split_stacked_pages(stack([page(), page(), page()]))
        self.assertEqual(len(pieces), 3)
        for piece in pieces:
            self.assertTrue(blank_rows(piece))
            self.assertLess(abs(piece.height - A4[1]), A4[1] * 0.15)

    def test_the_cut_finds_the_gap_when_the_pages_differ_in_length(self):
        """A boundary off the A4 grid -- one short page, one long -- is still
        cut in the margin, not through the line nearest the grid."""
        first, second = page((1240, 1600)), page((1240, 1950))
        pieces = app.split_stacked_pages(stack([first, second]))
        self.assertEqual(len(pieces), 2)
        self.assertLess(abs(pieces[0].height - first.height), 120)   # within the margin
        self.assertTrue(all(blank_rows(p) for p in pieces))

    def test_landscape_pages_side_by_side_are_cut_across(self):
        pieces = app.split_stacked_pages(stack([page(A4[::-1]), page(A4[::-1])],
                                               tall=False))
        self.assertEqual(len(pieces), 2)
        self.assertEqual({p.height for p in pieces}, {A4[0]})

    def test_a_long_receipt_is_cut_between_lines(self):
        pieces = app.split_stacked_pages(page((600, 1800), margin=30))
        self.assertEqual(len(pieces), 2)
        self.assertTrue(all(blank_rows(p) for p in pieces))
        self.assertEqual(sum(p.height for p in pieces), 1800)


class LoadPagesTests(unittest.TestCase):
    def test_a_stacked_tiff_frame_is_read_as_its_pages(self):
        pages, total = app.load_pages(tiff([stack([page(), page()])]))
        self.assertEqual((len(pages), total), (2, 2))

    def test_frames_and_pieces_keep_their_order(self):
        data = tiff([page(), stack([page(), page()]), page()])
        pages, total = app.load_pages(data)
        self.assertEqual((len(pages), total), (4, 4))
        self.assertEqual([p.height < A4[1] * 1.2 for p in pages], [True] * 4)

    def test_the_cap_counts_pieces_and_the_total_says_what_was_dropped(self):
        data = tiff([stack([page(), page(), page()]), page()])
        with patch.object(app, "MAX_PAGES", 2):
            pages, total = app.load_pages(data)
        # Two frames decoded at a cap of 2: three pieces and one page -- of
        # which two are read.
        self.assertEqual((len(pages), total), (2, 4))

    def test_it_can_be_turned_off(self):
        with patch.object(app, "SPLIT_STACKED_PAGES", False):
            pages, total = app.load_pages(tiff([stack([page(), page()])]))
        self.assertEqual((len(pages), total), (1, 1))

    def test_a_pdf_page_is_never_cut(self):
        import fitz
        doc = fitz.open()
        doc.new_page(width=595, height=842 * 2)
        pages, total = app.load_pages(doc.tobytes())
        self.assertEqual((len(pages), total), (1, 1))


if __name__ == "__main__":
    unittest.main()
