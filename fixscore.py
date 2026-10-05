"""How well the table fix agent did, against `solution/tables/<id>.md`.

**Separate from every other score, on purpose.** `table_score` measures the
table the run returned -- mostly the READ, since the cells are the transcript's
own characters -- and `field_score` measures pass 2. Neither can say whether the
fix agent did its job, because a run's table is right or wrong for reasons that
have nothing to do with it: the read dropped a row, the code re-cut a column.
This scores only the agent's own decisions, and never feeds a number into the
other two.

What the agent decides, and so what is scored:

  strays     the things stamped, stuck or written in the table's frame -- each
             line of each `### Stray:` block that has text. RECALL: of those the
             agent was shown, how many it moved out.
  removals   everything the agent moved out (`outside_text`). PRECISION: how
             many are stray, how many are ordinary page text it mislabelled, and
             how many are a value the clean table holds -- the one that costs a
             real cell.
  own lines  lines the agent said belong to the table though the read put them
             outside it (`footer`): are they the table's?
  cells      the clean table's cells, scored the way `table_score` scores them,
             on the table BEFORE the agent and AFTER it. The difference is the
             agent's effect; the level is the read's.

**A stray the read never produced is not counted against the agent** (`not
read`). The OCR model drops barcodes and stamps all the time; that is pass 1's
failure, and recall is taken over what the agent was actually shown. Nor is
one the read produced outside the lines the agent is sent (`not shown`): that is
the window's reach, not the agent's judgement. A stray squashing to under
`MIN_CONTAIN` characters counts as read only where a line prints it literally and
standing alone, and a longer one only where it is most of a line -- otherwise
`3:1` is found in any page and `SPRING` in `SPRINGROLL`.

Matching is `grounding.squash` -- loose about spacing and punctuation, strict
about content -- with containment allowed from `MIN_CONTAIN` characters, so a
barcode read `A01$T*-` still matches the printed `*A01$TX*`.

`python fixscore.py` runs the agent on the saved reads (or `--from-truth` on
the ground-truth transcripts) and prints this score alone. It writes no run-log
row.
"""

import argparse
import io
import json
import math
import re
import sys
import time
from pathlib import Path

import config
import fieldscore
import grounding
import segment
import tables

TABLES_DIR = config.SOLUTION_DIR / "tables"

# Below this many squashed characters two texts must be EQUAL to match: a short
# value is found inside almost anything.
MIN_CONTAIN = 4

_SECTION = re.compile(r"^## Table (\d+) — pages? (\d+)(?:-(\d+))?\s*$")
_STRAY = re.compile(r"^### Stray: (.+?)\s*$")
_NO_TEXT = re.compile(r"^\(.*no text\)$")


# --------------------------------------------------------------------------
# the truth file
# --------------------------------------------------------------------------

def load(case_id: str) -> list:
    """Every table section of `solution/tables/<id>.md`, in file order.

    Each is {number, page, pages, columns, rows, markdown, strays} -- `page` the
    first page, `pages` every page a table running over several covers; a stray is
    {kind, where, lines} where `lines` holds its text lines and is empty for a
    mark with no text. [] where the case has no file.
    """
    path = TABLES_DIR / f"{case_id}.md"
    if not path.is_file():
        return []
    sections, current, stray = [], None, None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        m = _SECTION.match(line)
        if m:
            first = int(m.group(2))
            last = int(m.group(3) or first)
            current = {"number": int(m.group(1)), "page": first,
                       "pages": list(range(first, last + 1)),
                       "table_lines": [], "strays": []}
            sections.append(current)
            stray = None
            continue
        if current is None:
            continue
        s = _STRAY.match(line)
        if s:
            stray = None
            if s.group(1) != "none":
                kind, _, where = s.group(1).partition(" — ")
                stray = {"kind": kind.strip(), "where": where.strip(), "lines": []}
                current["strays"].append(stray)
            continue
        if stray is None and line.startswith("|"):
            current["table_lines"].append(line)
        elif stray is not None and line.strip() and not _NO_TEXT.match(line.strip()):
            stray["lines"].append(line.strip())
    for section in sections:
        md = "\n".join(section.pop("table_lines"))
        found = tables.pipe_tables(md, min_lines=2)
        head, rows = found[0] if found else ([], [])
        section.update(columns=list(head), rows=[list(r) for r in rows], markdown=md)
    return sections


# --------------------------------------------------------------------------
# matching
# --------------------------------------------------------------------------

def _sq(text) -> str:
    return grounding.squash(str(text or "").replace("<br>", " "))


def same(a, b) -> bool:
    """Two texts are one, loosely: equal squashed, or one inside the other."""
    a, b = _sq(a), _sq(b)
    if not a or not b:
        return False
    if a == b:
        return True
    return min(len(a), len(b)) >= MIN_CONTAIN and (a in b or b in a)


# A stray that squashes this short (`3:1`, a lone `ย`) is found inside almost any
# page once squashed, so it counts as read only where the transcript prints it
# literally and not in the middle of a longer word.
_WORD = r"[^\W_]"


def _found(line: str, text: str) -> bool:
    """Whether `text` -- one line of the read -- holds this stray.

    Loose about spacing and punctuation like `same`, with two guards it lacks:
    a short stray must be printed literally and standing on its own, and a long
    one must be most of the line rather than a word inside it (SPRING is not
    read because SPRINGROLL is). A stray that runs over several read lines, or
    a read line holding several strays, still matches either way round.
    """
    a, b = _sq(line), _sq(text)
    if not a or not b:
        return False
    if len(a) < MIN_CONTAIN:
        raw = re.sub(r"\s+", " ", str(line)).strip()
        flat = re.sub(r"\s+", " ", str(text or ""))
        return bool(re.search("(?<!%s)%s(?!%s)" % (_WORD, re.escape(raw), _WORD), flat))
    if a == b:
        return True
    if a in b:
        return len(a) >= 0.6 * len(b) or bool(
            re.search(r"(?<!%s)%s(?!%s)" % (_WORD, re.escape(str(line).strip()), _WORD),
                      str(text)))
    return len(b) >= MIN_CONTAIN and b in a


def _in_read(line: str, transcript: str, transcript_lines) -> bool:
    """Whether the read produced this text at all, near enough to match."""
    return any(_found(line, t) for t in transcript_lines)


def _shown(table: dict) -> list:
    """Every piece of text the fix agent was shown: the lines before and after
    the table and the cells as they stood when it was asked. None where the
    table carries no window, so nothing is ruled out on a guess."""
    if "tail" not in table and "context_before" not in table:
        return None
    stages = {s.get("stage"): s for s in table.get("stages") or []}
    stage = stages.get("concat") or stages.get("read") or table
    cells = [c for row in stage.get("rows") or [] for c in row if not grounding.is_blank(c)]
    return (list(table.get("context_before") or []) + list(table.get("tail") or [])
            + [str(c) for c in cells])


def match_sections(sections: list, table: dict) -> list:
    """The truth sections this returned table is: its pages, its headings.

    One table joined over several pages matches one section per page. Where
    none on its pages matches, the best match anywhere in the file is taken,
    so a page-numbering slip does not leave a table unscored.
    """
    columns = list((table or {}).get("columns") or [])
    if not columns or not sections:
        return []

    def fit(section):
        return len(fieldscore._map_columns(section["columns"], columns))

    def good(section):
        need = max(1, math.ceil(0.5 * len(section["columns"])))
        return fit(section) >= need

    pages = set((table or {}).get("pages") or [])
    chosen = []
    for page in sorted(pages):
        on_page = [s for s in sections if page in s["pages"] and good(s)]
        if on_page:
            best = max(on_page, key=fit)
            if not any(best is c for c in chosen):
                chosen.append(best)
    if not chosen:
        anywhere = [s for s in sections if good(s)]
        if anywhere:
            chosen = [max(anywhere, key=fit)]
    return chosen


def _truth_item_table(chosen: list):
    """The clean table as `tables.item_table` reads it -- totals rows out --
    so it is compared on the same terms as the returned table."""
    if not chosen:
        return None
    if len(chosen) == 1:
        return tables.item_table(chosen[0]["markdown"])
    text = "\n".join("--- page %d ---\n%s" % (s["page"], s["markdown"])
                     for s in sorted(chosen, key=lambda s: s["page"]))
    return tables.item_table(text)


# --------------------------------------------------------------------------
# the score
# --------------------------------------------------------------------------

def _cells(truth, stage) -> dict:
    if not truth or not stage:
        return None
    result = fieldscore.score_table(truth, {"columns": stage.get("columns") or [],
                                            "rows": stage.get("rows") or []})
    return {"correct": result["counts"].get("correct", 0),
            "expected": result.get("expected", 0)}


def score(case_id: str, table: dict, transcript: str = "") -> dict:
    """The fix agent's score for one document's table, or None with nothing to say.

    None -- not zero -- where the case has no table truth file, or the run
    returned no table. `transcript` is the text the table was read from; without
    it every stray is assumed to have been in the read.
    """
    sections = load(case_id)
    if not sections or not isinstance(table, dict) or "rows" not in table:
        return None
    chosen = match_sections(sections, table)
    # The strays, the values and the table's own lines are judged against EVERY
    # truth table on the table's pages, not only the one it matched: the agent is
    # shown a window of the page, and a totals grid ruled beside the item table
    # (sol003, sol009) holds stamps and totals it is asked about as much as the
    # item table's own frame does.
    pages = set(table.get("pages") or [])
    around = [s for s in sections if pages & set(s["pages"])] or chosen
    record = table.get("fix_agent") or {}
    joined = table.get("concat") or {}
    result = {"case": case_id, "sections": [s["number"] for s in chosen],
              "agent": {"asked": bool(record.get("asked")),
                        "model": record.get("model") or "",
                        "skipped": record.get("skipped") or "",
                        "error": record.get("error") or ""},
              "concat": {"asked": bool(joined.get("asked")),
                         "offered": joined.get("offered", 0),
                         "joined": len(joined.get("joined") or []),
                         "by_sum": len(joined.get("by_sum") or []),
                         "rows_added": joined.get("rows_added", 0),
                         "dropped_added": joined.get("dropped_added", 0)}}
    transcript_lines = [l.strip() for l in (transcript or "").splitlines() if l.strip()]
    final_cells = [c for row in table.get("rows") or [] for c in row
                   if not grounding.is_blank(c)]
    outside = [e for e in table.get("outside_text") or [] if str(e.get("text") or "").strip()]
    footer = [e for e in table.get("footer") or [] if str(e.get("text") or "").strip()]

    # Strays: recall over what the agent was shown. A stray the read never
    # produced, or produced outside the window the agent is sent, is not its miss.
    shown = _shown(table)
    items, marks = [], 0
    for section in around:
        for stray in section["strays"]:
            if not stray["lines"]:
                marks += 1
            for line in stray["lines"]:
                if transcript and not _in_read(line, transcript, transcript_lines):
                    status = "not_read"
                elif shown is not None and not any(_found(line, t) for t in shown):
                    status = "not_shown"
                elif any(same(line, e.get("text")) for e in outside):
                    status = "moved"
                elif any(same(line, c) for c in final_cells):
                    status = "left_in_table"
                else:
                    status = "missed"
                items.append({"text": line, "kind": stray["kind"],
                              "where": stray["where"], "status": status})
    counted = [i for i in items if i["status"] not in ("not_read", "not_shown")]
    moved = sum(1 for i in counted if i["status"] == "moved")
    result["strays"] = {
        "expected": len(items),
        "not_read": sum(1 for i in items if i["status"] == "not_read"),
        "not_shown": sum(1 for i in items if i["status"] == "not_shown"),
        "scored": len(counted),
        "moved": moved,
        "left_in_table": sum(1 for i in counted if i["status"] == "left_in_table"),
        "missed": sum(1 for i in counted if i["status"] == "missed"),
        "recall": round(100.0 * moved / len(counted), 2) if counted else None,
        "marks": marks,
        "items": items,
    }

    # Removals: precision over what the agent moved out.
    stray_lines = [line for s in around for st in s["strays"] for line in st["lines"]]
    truth_cells = [c for s in around for row in s["rows"] for c in row
                   if not grounding.is_blank(c)]
    removals = []
    for entry in outside:
        text = entry.get("text")
        if any(same(text, line) for line in stray_lines):
            status = "stray"
        elif any(same(text, cell) for cell in truth_cells):
            status = "real_value"
        else:
            status = "page_text"
        removals.append({"text": text, "status": status,
                         "where": entry.get("where") or (
                             "row %d, %s" % (entry["row"] + 1, entry.get("heading") or "cell")
                             if isinstance(entry.get("row"), int) else "")})
    good = sum(1 for r in removals if r["status"] == "stray")
    result["removals"] = {
        "total": len(removals),
        "stray": good,
        "page_text": sum(1 for r in removals if r["status"] == "page_text"),
        "real_value": sum(1 for r in removals if r["status"] == "real_value"),
        "precision": round(100.0 * good / len(removals), 2) if removals else None,
        "items": removals,
    }

    # Own lines: the table's, read outside it.
    truth_rows = [" ".join(c for c in row if not grounding.is_blank(c))
                  for s in around for row in s["rows"]]
    own = []
    for entry in footer:
        text = entry.get("text")
        hit = any(same(text, row) for row in truth_rows) or \
            any(same(text, cell) for cell in truth_cells if len(_sq(cell)) >= MIN_CONTAIN)
        own.append({"text": text, "status": "table" if hit else "not_table",
                    "where": entry.get("where") or ""})
    result["own_lines"] = {
        "total": len(own),
        "table": sum(1 for o in own if o["status"] == "table"),
        "not_table": sum(1 for o in own if o["status"] == "not_table"),
        "items": own,
    }

    # Cells, before and after the agent.
    truth = _truth_item_table(chosen)

    # The table's totals: the truth's totals and notes rows (one table since
    # 2026-10-01 -- the item table and its totals), and how many the returned
    # table holds as part of itself -- in its rows, its totals/notes, or the
    # lines the fix agent kept with it. A read that cut the totals into a table
    # of their own holds none of them until the concat agent joins it.
    held = [" ".join(c for c in row if not grounding.is_blank(c))
            for row in table.get("rows") or []]
    held += [" ".join(c for c in d.get("cells") or [] if not grounding.is_blank(c))
             for d in table.get("dropped") or []]
    held += [str(e.get("text") or "") for e in footer]
    expected_totals = [" ".join(c for c in d.get("cells") or [] if not grounding.is_blank(c))
                       for d in (truth or {}).get("dropped") or []]
    expected_totals = [t for t in expected_totals if t]
    found = [t for t in expected_totals
             if any(same(t, h) for h in held)
             or all(any(same(part, h) for h in held)
                    for part in t.split() if len(_sq(part)) >= MIN_CONTAIN)]
    result["totals"] = {"expected": len(expected_totals), "in_table": len(found),
                        "missing": [t for t in expected_totals if t not in found]}
    stages = {s.get("stage"): s for s in table.get("stages") or []}
    before = _cells(truth, stages.get("read"))
    after_agent = _cells(truth, stages.get("clean")) if "clean" in stages else None
    final = _cells(truth, table)
    if before:
        result["cells"] = {
            "expected": before["expected"],
            "read": before["correct"],
            "after_agent": after_agent["correct"] if after_agent else None,
            "final": final["correct"] if final else None,
            "agent_delta": (after_agent["correct"] - before["correct"])
                           if after_agent else None,
            "final_delta": (final["correct"] - before["correct"]) if final else None,
        }
    else:
        result["cells"] = None
    return result


_COUNTS = (("strays", ("expected", "not_read", "not_shown", "scored", "moved", "left_in_table",
                       "missed", "marks")),
           ("removals", ("total", "stray", "page_text", "real_value")),
           ("own_lines", ("total", "table", "not_table")),
           ("totals", ("expected", "in_table")),
           ("concat", ("offered", "joined", "by_sum", "rows_added", "dropped_added")))


def pool(scores) -> dict:
    """Several documents' fix scores as one for the file: counts summed, the
    two rates recomputed from the sums. One score comes back untouched."""
    scores = [s for s in scores if isinstance(s, dict) and "strays" in s]
    if not scores:
        return None
    if len(scores) == 1:
        return scores[0]
    out = {"case": scores[0].get("case"), "documents": len(scores),
           "sections": [n for s in scores for n in s.get("sections") or []],
           "agent": scores[0].get("agent")}
    for block, keys in _COUNTS:
        out[block] = {k: sum((s.get(block) or {}).get(k) or 0 for s in scores)
                      for k in keys}
        if block in ("strays", "removals", "own_lines"):
            out[block]["items"] = [i for s in scores
                                   for i in (s.get(block) or {}).get("items") or []]
    out["totals"]["missing"] = [m for s in scores for m in (s.get("totals") or {}).get("missing") or []]
    st, rm = out["strays"], out["removals"]
    st["recall"] = round(100.0 * st["moved"] / st["scored"], 2) if st["scored"] else None
    rm["precision"] = round(100.0 * rm["stray"] / rm["total"], 2) if rm["total"] else None
    cells = [s["cells"] for s in scores if s.get("cells")]
    if cells:
        def total(key):
            vals = [c[key] for c in cells if c.get(key) is not None]
            return sum(vals) if len(vals) == len(cells) else None
        out["cells"] = {k: total(k) for k in ("expected", "read", "after_agent", "final")}
        out["cells"]["agent_delta"] = (out["cells"]["after_agent"] - out["cells"]["read"]
                                       if out["cells"]["after_agent"] is not None else None)
        out["cells"]["final_delta"] = (out["cells"]["final"] - out["cells"]["read"]
                                       if out["cells"]["final"] is not None else None)
    else:
        out["cells"] = None
    return out


def format_report(result) -> list:
    """The score as lines of text, for the CLI."""
    if not result:
        return ["  no table truth to score against"]
    st, rm, own, cells = (result.get("strays") or {}, result.get("removals") or {},
                          result.get("own_lines") or {}, result.get("cells"))
    agent = result.get("agent") or {}
    rate = lambda v: "-" if v is None else "%.1f%%" % v
    out = []
    if not agent.get("asked"):
        out.append("  fix agent not asked%s" % (": " + agent["skipped"] if agent.get("skipped") else ""))
    out.append("  strays     recall %-7s moved %d of %d it was shown  (left in table %d, "
               "missed %d; not in the read %d; outside the window %d; marks with no text %d)"
               % (rate(st.get("recall")), st.get("moved", 0), st.get("scored", 0),
                  st.get("left_in_table", 0), st.get("missed", 0), st.get("not_read", 0),
                  st.get("not_shown", 0),
                  st.get("marks", 0)))
    out.append("  removals   precision %-7s %d of %d stray  (page text %d, REAL VALUES %d)"
               % (rate(rm.get("precision")), rm.get("stray", 0), rm.get("total", 0),
                  rm.get("page_text", 0), rm.get("real_value", 0)))
    out.append("  own lines  %d of %d are the table's" % (own.get("table", 0), own.get("total", 0)))
    tot, cc = result.get("totals") or {}, result.get("concat") or {}
    if tot.get("expected"):
        out.append("  totals     %d of %d of the table's totals rows are held as part of it"
                   % (tot.get("in_table", 0), tot["expected"]))
    if cc.get("offered"):
        out.append("  concat     joined %d of %d other tables the read made, %d of them by a "
                   "column total (%d rows, %d totals/notes)"
                   % (cc.get("joined", 0), cc["offered"], cc.get("by_sum", 0),
                      cc.get("rows_added", 0), cc.get("dropped_added", 0)))
    if cells:
        sign = lambda v: "-" if v is None else "%+d" % v
        out.append("  cells      read %s -> after agent %s -> final %s of %s  (agent %s, all repairs %s)"
                   % (cells.get("read"), cells.get("after_agent"), cells.get("final"),
                      cells.get("expected"), sign(cells.get("agent_delta")),
                      sign(cells.get("final_delta"))))
    for item in st.get("items") or []:
        if item["status"] in ("left_in_table", "missed"):
            out.append("    %-13s %s: %s" % (item["status"], item["kind"], item["text"][:70]))
    for item in rm.get("items") or []:
        if item["status"] != "stray":
            out.append("    %-13s %s" % ("REAL VALUE" if item["status"] == "real_value"
                                         else "page text", item["text"][:70]))
    return out


# --------------------------------------------------------------------------
# CLI: run the agent on saved reads and score it, nothing else
# --------------------------------------------------------------------------

def _documents(case: dict, text: str) -> list:
    """(pages, text) per document of a case, by the manifest's own pages."""
    pages = segment.split_pages(text)
    docs = case.get("documents") or [{"pages": list(range(1, len(pages) + 1))}]
    out = []
    for doc in docs:
        nums = [p for p in doc.get("pages") or [] if 1 <= p <= len(pages)]
        if not nums:
            continue
        if len(nums) == 1:
            out.append((nums, pages[nums[0] - 1]))
        else:
            out.append((nums, "\n".join("--- page %d ---\n%s" % (p, pages[p - 1])
                                        for p in nums)))
    return out


def main(argv=None):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("cases", nargs="*", help="case ids (default: every case with a table truth file)")
    parser.add_argument("--from-truth", action="store_true",
                        help="run on solution/<id>.md instead of the saved read solution/out/<id>.txt")
    parser.add_argument("--model", help="extraction model to run the agent on (substring of a served name)")
    parser.add_argument("--json", help="also write every score to this file")
    parser.add_argument("--no-concat", action="store_true",
                        help="run without the concat agent, to compare")
    args = parser.parse_args(argv)

    import app
    import backends
    import prompts
    import scoring
    backends.autoselect()
    if args.model:
        names = [m.get("name") for m in (backends.status().get("models") or [])]
        served = [n for n in names if n and args.model in n]
        if len(served) != 1:
            parser.error("--model %r matches %s" % (args.model, served or "nothing served"))
        backends.select(backends.active_url(), served[0], unload=False)
        backends.select_extract("", unload=False)
    else:
        backends.autoselect_models()
    status = backends.extract_status()
    if not status.get("text_available", status.get("available")):
        parser.error("no extraction model available at %s" % status.get("url"))
    print("fix agent on %s at %s" % (status.get("model"), status.get("url")))
    app.TABLE_FIX_AGENT = True
    app.TABLE_CONCAT_AGENT = not args.no_concat
    app.TABLE_IDENTIFY_AGENT = False

    manifest = {c["id"]: c for c in scoring.load_manifest()}
    wanted = args.cases or sorted(p.stem for p in TABLES_DIR.glob("sol*.md"))
    every, start = [], time.time()
    for case_id in wanted:
        source = (config.SOLUTION_DIR / f"{case_id}.md") if args.from_truth \
            else (config.SOLUTION_DIR / "out" / f"{case_id}.txt")
        if not source.is_file():
            print("\n%s: no %s" % (case_id, source.name))
            continue
        text = source.read_text(encoding="utf-8")
        case = manifest.get(case_id, {})
        scores, skipped = [], 0
        docs = case.get("documents") or [{"doc_types": case.get("doc_types") or []}]
        for (pages, doc_text), doc in zip(_documents(case, text), docs):
            codes = doc.get("doc_types") or case.get("doc_types") or []
            # The form the app would ask: a type that rules its own table (a
            # withholding certificate) is not looked at for an item table.
            form = {"doc_types": codes, "items": prompts.items_for_types(codes)}
            table = app._item_table(doc_text, form, status, pages, {})
            if table is ...:
                skipped += 1
            elif isinstance(table, dict) and "rows" in table:
                scores.append(score(case_id, table, doc_text))
        result = pool(scores)
        print("\n%s" % case_id)
        if not result and skipped:
            print("  not looked at -- its type rules a table of its own")
            continue
        print("\n".join(format_report(result)))
        if result:
            every.append(result)
    total = pool(every) if len(every) > 1 else (every[0] if every else None)
    if total:
        print("\nALL CASES (%.0fs)" % (time.time() - start))
        print("\n".join(format_report(dict(total, strays=dict(total["strays"], items=[]),
                                           removals=dict(total["removals"], items=[])))))
    if args.json:
        Path(args.json).write_text(json.dumps(every, ensure_ascii=False, indent=1),
                                   encoding="utf-8")


if __name__ == "__main__":
    main()
