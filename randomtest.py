"""Randomised end-to-end runs: a random document, models, detail and shape.

**What this covers that the benchmarks do not.** `compare.py` and the field-score
sweeps feed pass 2 a ground-truth transcript on purpose, so that a field score
measures the extractor and nothing else. That leaves the whole of the real path
untested by anything automatic: decoding an upload, trimming and resizing it,
the pass-1 profile, the transcript, the case match, and only then extraction --
possibly on a different model. This walks that path with settings nobody chose,
which is how it finds the combinations a person would not think to try.

It found three on its first six rounds: a 289-entry `other_fields` loop that
collapsed to 3 distinct entries, two pass-1 loops on a fixture nothing had
flagged, and a model that reads badly while extracting well.

**The plan is separated from the running of it**, and that is the whole design:
`plan()` is a pure function of its arguments, so the same seed produces the same
list of rounds forever and a failure can be handed to someone else as a number.
`app.py` executes a plan; this module never makes a request.

**One run is one scope.** `full` reads a page and extracts from what came back;
`ocr` stops after the read; `fields` skips the read and extracts from
`solution/<id>.md`, which is what every pass-2 baseline in CLAUDE.md was
measured on. They are separate runs rather than a fourth axis to randomise
because they answer different questions -- a field score taken over a real
transcript is partly a measurement of pass 1, and only `fields` isolates the
extractor.

Two rules about what may be planned:

- **A reader is any model that reports vision.** Pass 1 sends an image, so a
  text-only model is not a candidate; nothing else is excluded. This was briefly
  narrowed to OCR fine-tunes only and was reopened on 2026-08-21 at the user's
  request, after a random round measured `qwen3.5:4b` reading sol002 at 98.5% --
  the highest character accuracy this project has recorded. A pool that cannot
  contain that result cannot find it again.
- **A case is NOT drawn at random.** Every other axis is; documents are handed
  to whichever has been read fewest times, counting the run log plus the plan so
  far. Uniform draws are only fair in the limit, and nobody watches the limit:
  over the 24 rows the log held when this was written, sol005 had seven reads
  and sol006 none. See `case_order`.
- **An extractor is the reading model itself, or a model that is NOT an OCR
  fine-tune.** That is `backends.select_extract`'s rule, mirrored here so a plan
  never contains a round the server would refuse. It is mirrored rather than
  imported-and-trusted because a plan that cannot run is worse than one that is
  slightly conservative.

**What is NOT reopened is the default.** `backends._resolve_model` still prefers
an OCR model when nobody has chosen one, so a general model reads a page because
it was picked -- here, at random, or in the picker -- and never because it
happened to be the newest pull. Allowing something and drifting into it are
different, and only the first was asked for.

Run it from the command line against a running app:

    python randomtest.py http://localhost:5000 --rounds 10
    python randomtest.py http://localhost:5000 --seed 1787218747
    python randomtest.py http://localhost:5000 --scope fields --rounds 10
"""

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import backends

# How many rounds one request may ask for. The ceiling is not arithmetic: a
# round is a real read plus a real extraction, tens of seconds each, and a page
# that asks for 500 of them has asked for a job it cannot watch and will not
# cancel cleanly.
MIN_ROUNDS, MAX_ROUNDS, DEFAULT_ROUNDS = 1, 50, 5

# How much of the pipeline one round runs. `full` is what this module was built
# for and stays the default; the other two exist because the two passes fail
# differently and a round that runs both cannot say which half a result came
# from -- a field score taken over a real transcript is partly a measurement of
# pass 1, which is why every pass-2 baseline in CLAUDE.md was taken from truth.
#
#   full    read the page, then extract from what came back.
#   ocr     read the page and stop. No extraction, so no field score.
#   fields  extract from `solution/<id>.md`. No page is read, no image is made,
#           and no model needs vision -- pass 2 sends text and gets text.
SCOPES = ("full", "ocr", "fields")
DEFAULT_SCOPE = "full"


def profile_for(model: str) -> str:
    """The pass-1 profile a model needs.

    Delegates to `backends.profile_for_model` rather than deciding again here:
    the app now sets the profile from the model on every switch, and a planner
    that disagreed with it would produce rounds whose printed profile was not the
    one that ran.

    The profile is not randomised, and that is deliberate. It decides the prompt
    AND whether a system message is sent, and the wrong combination returns an
    empty transcript at HTTP 200 rather than an error -- a random test that
    picked profiles at random would spend most of its rounds re-measuring
    something this project already documents.
    """
    return backends.profile_for_model(model)


def pools(models: list, cases: list) -> dict:
    """What a plan may choose from, given what the endpoint actually serves.

    `models` is `status()["models"]` -- dicts with `name` and `vision`. `cases`
    is the ground-truth document ids. Returns empty lists rather than raising:
    the caller has a better error to give than this does, and an empty pool is a
    perfectly ordinary state for an endpoint that is not running yet.
    """
    named = [m for m in (models or []) if m.get("name")]
    return {
        # `vision is not False` rather than `is True`: a server that did not say
        # is worth attempting, which is the same call `backends.status` makes.
        # Every vision model is a candidate -- see the module docstring on why
        # this is not narrowed to OCR fine-tunes.
        "readers": [m["name"] for m in named if m.get("vision") is not False],
        # "" is "same as the reading model", and it is in the pool rather than
        # special-cased so that the one-model setup -- the one every measurement
        # in this project was taken under -- is part of what gets tested.
        "extractors": [""] + [m["name"] for m in named
                              if not backends.is_ocr_model(m["name"])],
        # A fields-only round reads no page, so EVERY served model is a
        # candidate: vision is irrelevant to a text pass, and an OCR fine-tune
        # is refused only as a *second* model beside a reader -- which a round
        # that never reads does not have. That is the same one-model shape the
        # Fields pane measures under, and it is how the pass-2 sweep scored
        # typhoon and both dots builds on the form at all.
        "text_models": [m["name"] for m in named],
        "cases": list(cases or []),
    }


def case_order(rounds: int, cases: list, history: dict, rng) -> list:
    """Which document each round reads: the least-read one, every time.

    **Cases are the one axis that is NOT drawn uniformly**, and the reason is
    that uniform is not fair over the handful of rounds anyone actually watches.
    Five rounds over ten documents is five draws with replacement: some fixture
    gets three of them and four others get none, and the run log said so --
    sol005 at seven reads and sol006 at zero, over 24 rows.

    So each round goes to whichever case has been read fewest times, counting
    `history` (what the run log already holds, from `runlog.case_counts`) plus
    the rounds planned above it. Ties go to a document this plan has not used
    yet, and are otherwise broken by the seeded `rng` over a sorted pool -- so
    the choice is still random and still reproducible, and what is gone is only
    the possibility of drawing a document that is already ahead.

    Three consequences worth keeping:

    - **A document nobody has run wins every tie-break it is in**, so a fixture
      added to `solution/` is picked up by the next random test rather than
      waiting for a lucky draw.
    - **The plan self-balances even with no history at all.** Ten rounds over
      ten cases is one each, in a random order, not ten coin flips.
    - **Failed reads count.** `case_counts` counts them deliberately: a round
      already spent on a document is a round spent, and skipping failures here
      would funnel every future round into whichever fixture breaks most.
    """
    tally = {case: int((history or {}).get(case, 0) or 0) for case in cases}
    order, used = [], set()
    for _ in range(rounds):
        fewest = min(tally.values())
        # sorted() so the tie-break pool is in the same order on every machine;
        # the rng is what makes the choice within it random.
        pool = sorted(c for c in tally if tally[c] == fewest)
        # Among equals, a document this plan has not touched yet goes first.
        # Counts alone already balance in the long run; this is what makes ONE
        # short run watchably fair -- without it a case can win two tie-breaks
        # in six rounds while a case level with it gets none.
        fresh = [c for c in pool if c not in used]
        pick = rng.choice(fresh or pool)
        order.append({"case": pick, "seen": tally[pick]})
        tally[pick] += 1
        used.add(pick)
    return order


def apply_exclusions(pools: dict, exclude: dict = None,
                     scope: str = DEFAULT_SCOPE) -> dict:
    """The pools with the models (or documents) the caller toggled off removed.

    Added 2026-08-21 at the user's request -- *when random i dont want qwen 2b to
    be include in ocr i can click exclude that on the test*. It is the complement
    of a lock: a lock says *only this*, an exclusion says *anything but this*, and
    the two answer different questions -- the first pins an axis, the second
    narrows it while leaving it random.

    **Excluding from reading and from extracting are separate**, which is the
    point of the example: a model can be a poor page reader and a good extractor,
    and this project has two of those. `readers` narrows pass 1; `extractors`
    narrows both the second-model pool and the one-model pool a `fields` round
    draws from, because in that round the drawn model IS the extractor.

    **`""` -- "same as the reading model" -- is never excluded.** It is not a
    model; it is the one-model setup every baseline in this project was measured
    under, and dropping it would quietly turn a narrowed run into a two-model-only
    one.

    **An axis emptied by an exclusion is refused, and only where the scope needs
    it.** Excluding every extraction model is a legitimate way to say *always
    extract with the reading model*, so it is an error only for a `fields` run,
    which has nothing else to draw. The message says the exclusions did it --
    "no model reports vision" would send someone looking at their server.
    """
    exclude = {key: {name for name in ((exclude or {}).get(key) or []) if name}
               for key in ("readers", "extractors", "cases")}
    out = dict(pools)
    if exclude["readers"]:
        out["readers"] = [m for m in pools.get("readers") or []
                          if m not in exclude["readers"]]
        if not out["readers"] and scope != "fields":
            raise ValueError("Every model that can read a page is excluded. "
                             "Put one back, or run a fields-only test.")
    if exclude["extractors"]:
        # "" stays: see the docstring.
        out["extractors"] = [m for m in pools.get("extractors") or []
                             if not m or m not in exclude["extractors"]]
        out["text_models"] = [m for m in pools.get("text_models") or []
                              if m not in exclude["extractors"]]
        if not out["text_models"] and scope == "fields":
            raise ValueError("Every model is excluded from extracting, so a "
                             "fields-only test has nothing to run.")
    if exclude["cases"]:
        out["cases"] = [c for c in pools.get("cases") or []
                        if c not in exclude["cases"]]
        if not out["cases"]:
            raise ValueError("Every document is excluded.")
    return out


def plan(rounds: int, cases: list, readers: list, extractors: list,
         details: list, modes: list, seed=None, history: dict = None,
         scope: str = DEFAULT_SCOPE, text_models: list = None,
         lock: dict = None) -> dict:
    """Build the list of rounds. Pure: same seed and same history, same plan.

    Returns {"seed": int, "rounds": [ ... ]}. The seed is returned whether or
    not one was given, because a run nobody can repeat is a bug report nobody
    can act on -- and the interesting rounds are exactly the ones that surprised
    somebody.

    **`history` is now part of what a seed reproduces**, and that is the price
    of spreading the rounds across the documents: the same seed re-run after
    some rounds have been logged plans the same models, details and shapes in
    the same order, over whichever cases are furthest behind *now*. Pass the
    same `history` back to reproduce a plan exactly; pass none and every case
    starts level. Each round carries `seen`, the number of reads that document
    had before it, so a plan says on its face why it chose what it chose.

    **`scope` says how much of the pipeline a round runs** -- see `SCOPES`. It is
    one choice for the whole plan rather than a fourth thing to randomise: the
    three scopes answer different questions, and a run that mixed them would
    report a mean over two of them. Every round carries it anyway, so a result
    always states what it ran.

    **`lock` pins an axis instead of drawing it** -- `{"case": ..., "reader":
    ..., "extractor": ..., "mode": ...}`, any subset, added 2026-08-21 at the
    user's request and extended to the extraction shape on 2026-08-24.
    A locked axis is not randomness with a preference; it is that axis removed
    from the experiment, which is what makes a run answer a question about the
    others. Locking the document is how a model comparison stops being a
    document comparison as well.

    A lock this endpoint cannot honour is **refused, not ignored**: silently
    drawing at random after being told to pin would produce a plan that answers a
    different question than the one asked, and nothing in the result would say so.

    **A round only draws the settings its scope actually uses**, and the ones it
    does not are `""` rather than absent. A `fields` round has no Detail because
    no image is made, an `ocr` round has no extraction shape because pass 2 does
    not run, and printing a value for either would name a setting that had no
    part in the result. The shape of the dict is the same on all three so the
    page and the CLI can read one round without asking which kind it is.
    """
    scope = (scope or DEFAULT_SCOPE).strip().lower()
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {', '.join(SCOPES)}.")
    # What "nothing to test" means depends on the scope, and saying so precisely
    # is the difference between a message someone can act on and one that sends
    # them looking for a vision model they do not need.
    if not cases:
        raise ValueError("Nothing to test: no document here has both a "
                         "transcript truth and a field truth.")
    if scope != "fields" and not (readers and details):
        raise ValueError("Nothing to read a page with: no model at this "
                         "endpoint reports vision.")
    if scope != "ocr" and not modes:
        raise ValueError("No extraction shape to run.")
    if scope == "fields" and not text_models:
        raise ValueError("Nothing to extract with: this endpoint serves no "
                         "models.")

    lock = {key: (value or "").strip()
            for key, value in (lock or {}).items() if (value or "").strip()}
    pinned_case = lock.get("case")
    pinned_reader = lock.get("reader")
    pinned_extractor = lock.get("extractor")
    pinned_mode = lock.get("mode")
    if pinned_case and pinned_case not in cases:
        raise ValueError(f"{pinned_case} is not a document that can be scored "
                         "on both passes here, so it cannot be locked.")
    if pinned_reader and scope != "fields" and pinned_reader not in readers:
        raise ValueError(f"{pinned_reader} is not served here, or does not "
                         "report vision, so it cannot be locked as the reader.")
    # A fields round has no reader, so its locked model is the extraction one --
    # which is drawn from every served model there, and from the non-OCR ones
    # elsewhere. Checking against the pool the round will actually draw from is
    # what makes the refusal mean something.
    extract_pool = text_models if scope == "fields" else extractors
    if pinned_extractor and pinned_extractor not in (extract_pool or []):
        raise ValueError(
            f"{pinned_extractor} cannot be locked as the extraction model here: "
            + ("it is not served at this endpoint." if scope == "fields" else
               "pass 2 runs on the reading model or on a model that is not an "
               "OCR fine-tune."))
    # The shape is a lock like any other, and the measured reason to want it is
    # in CLAUDE.md: single and agentic are not two samples of one setting -- the
    # gap between them on typhoon was 2.5x on the 29-key schema, and qwen3.5:2b
    # prefers the shape the others do not. A sweep of anything else has to hold
    # it still or it is measuring both.
    if pinned_mode and scope != "ocr" and pinned_mode not in (modes or []):
        raise ValueError(f"{pinned_mode} is not an extraction shape: "
                         f"{', '.join(modes or [])}.")

    rounds = max(MIN_ROUNDS, min(int(rounds or DEFAULT_ROUNDS), MAX_ROUNDS))
    seed = int(time.time()) if seed in (None, "") else int(seed)
    rng = random.Random(seed)

    if pinned_case:
        # The fairness rule is about spreading rounds over documents, and a
        # locked document is the user saying not to. `seen` still counts up from
        # the log, so the round still says how much history it is adding to.
        base = int((history or {}).get(pinned_case, 0) or 0)
        order = [{"case": pinned_case, "seen": base + i} for i in range(rounds)]
    else:
        order = case_order(rounds, cases, history, rng)

    planned = []
    for chosen in order:
        round_ = {**chosen, "scope": scope, "reader": "", "profile": "",
                  "extractor": "", "detail": "", "mode": ""}
        if scope == "fields":
            # No reader at all. The one model in force does the extracting, which
            # is the one-model setup `backends.select_extract` never refuses --
            # and the only way an OCR fine-tune can be measured on the form.
            round_["extractor"] = pinned_extractor or rng.choice(text_models)
        else:
            reader = pinned_reader or rng.choice(readers)
            round_["reader"] = reader
            round_["profile"] = profile_for(reader)
            round_["detail"] = rng.choice(details)
            if scope == "full":
                extractor = (pinned_extractor if "extractor" in lock
                             else rng.choice(extractors or [""]))
                # Never plan the one combination the server refuses. It cannot
                # arise from `pools` above, but a caller may pass its own lists.
                if (extractor and extractor != reader
                        and backends.is_ocr_model(extractor)):
                    extractor = ""
                round_["extractor"] = extractor
        if scope != "ocr":
            round_["mode"] = pinned_mode or rng.choice(modes)
        planned.append(round_)
    # Returned so the page and the CLI can say what was pinned. A plan that looks
    # unusually repetitive should say why on its own face.
    return {"seed": seed, "scope": scope, "rounds": planned, "lock": lock}


# What a contest can be about, and where each one gets its ranking. Added
# 2026-08-21 at the user's request -- *allow to choose top what (ocr model,
# extract model, solution file or full pipeline on each top 3)*.
#
# The subject decides the scope too, and that is not a shortcut: a contest
# between readers that also extracted would be scored partly on pass 2, and a
# contest between extractors has no page to read. Pinning the scope to the
# subject is the same rule the whole mode is built on -- one thing varies.
SUBJECTS = {
    "ocr_model": {"scope": "ocr", "ranking": ("ocr", "models"),
                  "label": "OCR model"},
    "extract_model": {"scope": "fields", "ranking": ("extract", "models"),
                      "label": "extraction model"},
    "case": {"scope": "ocr", "ranking": ("ocr", "cases"), "label": "document"},
    "pipeline": {"scope": "full", "ranking": ("ocr", "models"),
                 "label": "reader + extractor"},
}
DEFAULT_SUBJECT = "ocr_model"

# How many from each end of the ranking a contest runs, and the Detail it runs
# them at. The Detail is FIXED and that is the whole point of the mode: the
# random test gives every round its own Detail on purpose, which is what makes
# its rounds incomparable with each other. A contest is the opposite question --
# same page, same pixel budget, same shape, one thing changed -- so it pins
# everything the ranking is not about.
#
# `medium` (4 MP) rather than `original`, because it is the measured best and not
# merely the biggest: past ~4 MP the extra visual tokens degrade this model, and
# native resolution took twice as long and looped on the one page it was measured
# on. See settings.DETAIL_PRESETS.
CONTEST_TOP, CONTEST_BOTTOM = 5, 5
CONTEST_DETAIL = "medium"


def contenders(ranked: list, available: list, top: int = CONTEST_TOP,
               bottom: int = CONTEST_BOTTOM) -> dict:
    """Which models a contest runs, out of a `runlog.standouts` ranking.

    The top `top` and the bottom `bottom` of the ranking, deduplicated, in rank
    order. Where the two ends overlap -- fewer than ten models ranked -- every
    ranked model runs once and nothing is run twice.

    Three rules, each of which is a way the contest would otherwise lie:

    - **Only ranked models enter.** A model with no score has nothing to defend
      and nothing to answer for; a contest is a rematch, not a first outing.
      Anything unranked is reported as `unranked` so the page can say so rather
      than silently leaving it out.
    - **A model this endpoint cannot run is dropped, and named.** The ranking
      comes from the log and the log outlives a `ollama rm`; a plan containing a
      model nobody serves would fail every round it appears in.
    - **The order is the ranking's**, so the first rounds are the leaders and a
      run stopped half way still answers the top half of the question.

    Returns {"models": [...], "top": [...], "bottom": [...], "unranked": [...],
    "missing": [...]} -- the parts named apart because the page reports them
    differently: `missing` is a caveat about this endpoint, `unranked` is a
    caveat about the log.
    """
    served = {name for name in (available or []) if name}
    scored = [e["key"] for e in (ranked or []) if e.get("score") is not None]
    unranked = [e["key"] for e in (ranked or []) if e.get("score") is None]
    missing = [name for name in scored if name not in served]
    scored = [name for name in scored if name in served]

    leaders = scored[:max(0, top)]
    trailers = scored[-max(0, bottom):] if bottom else []
    picked, seen = [], set()
    for name in leaders + trailers:
        if name not in seen:
            seen.add(name)
            picked.append(name)
    # **A model at both ends is at neither.** With five models ranked and five
    # asked for from each end, every one of them is a "leader" and every one is
    # a "trailer" -- and tagging the last-placed model `top` because the slice
    # reached it is exactly the kind of confidently wrong label this project
    # refuses elsewhere. Where the ends overlap the contest is simply *everyone*,
    # and `all_ranked` says so instead.
    both = set(leaders) & set(trailers)
    return {"models": picked,
            "top": [n for n in leaders if n not in both],
            "bottom": [n for n in trailers if n not in both],
            "all_ranked": len(picked) == len(scored) and bool(scored),
            "unranked": unranked, "missing": missing}


def _ranked_for(standouts: dict, subject: str) -> list:
    """The ranking a subject is contested on, out of `runlog.standouts`."""
    pass_, kind = SUBJECTS[subject]["ranking"]
    return ((standouts or {}).get(pass_, {}).get(kind, {}) or {}).get("ranked") or []


def _leader(standouts: dict, pass_: str, available: list) -> str:
    """The best-scoring model of one pass that this endpoint still serves.

    What a document contest holds fixed. A contest between documents has to read
    them with something, and reading them with the leader is the only choice that
    does not need a second opinion: any other model makes the result a statement
    about that model as much as about the page.
    """
    served = {name for name in (available or []) if name}
    for entry in ((standouts or {}).get(pass_, {}).get("models", {}) or {}).get("ranked") or []:
        if entry.get("score") is not None and entry["key"] in served:
            return entry["key"]
    return ""


def contest_plan(standouts: dict, cases: list, readers: list, text_models: list,
                 extractors: list = None, mode: str = "single",
                 subject: str = DEFAULT_SUBJECT, documents: int = 1,
                 top: int = CONTEST_TOP, bottom: int = CONTEST_BOTTOM,
                 seed=None, history: dict = None,
                 detail: str = CONTEST_DETAIL) -> dict:
    """A rematch: the ends of one ranking, re-run with everything else pinned.

    **This is the random test's opposite and shares its runner.** A round here is
    the same dict a random round is, so `app._run_round`, the page's table and
    the run log need to know nothing about contests -- what differs is only how
    the list is built.

    - **The subject is the only thing that varies.** Detail, extraction shape and
      the documents are the same for every contender, because a ranking is worth
      re-testing only on a level field. `SUBJECTS` says which scope each subject
      implies and which ranking it comes from.
    - **`bottom` may be 0**, which runs the leaders alone -- the toggle on the
      page. Worth having because the two halves answer different questions: the
      top is *which should I use*, the bottom is *is this really as bad as the
      log says*, and the second costs as much as the first.
    - **Documents come from the fairness rule**, `case_order`, so a contest also
      spends its rounds on the fixtures that have had the fewest -- and every
      contender gets the same ones. The `case` subject is the exception: there
      the documents ARE the contenders.
    - **The contender is the outer loop.** Ollama loads a model per switch, so
      contender-then-document pays one load each; the other order pays one per
      round.

    Everything is clamped to `MAX_ROUNDS`: a contest nobody can watch to the end
    answers nothing, and what was clamped is reported back.
    """
    subject = (subject or DEFAULT_SUBJECT).strip().lower()
    if subject not in SUBJECTS:
        raise ValueError(f"subject must be one of {', '.join(SUBJECTS)}.")
    if not cases:
        raise ValueError("Nothing to test: no document here has both a "
                         "transcript truth and a field truth.")
    scope = SUBJECTS[subject]["scope"]
    top, bottom = max(0, int(top or 0)), max(0, int(bottom or 0))
    if not (top or bottom):
        raise ValueError("A contest with no contenders: ask for a top, a "
                         "bottom, or both.")

    ranked = _ranked_for(standouts, subject)
    # What the ranking is allowed to nominate, per subject. Checking against the
    # pool the rounds will actually draw from is what stops a plan containing a
    # model this endpoint stopped serving a week ago.
    pool = {"ocr_model": readers, "pipeline": readers,
            "extract_model": text_models, "case": cases}[subject]
    picked = contenders(ranked, pool, top, bottom)
    if not picked["models"]:
        raise ValueError(
            f"No contender: nothing in the run log has a score for a "
            f"{SUBJECTS[subject]['label']} yet, or the ones that do are not "
            "available here. Run a random test first.")

    seed = int(time.time()) if seed in (None, "") else int(seed)
    rng = random.Random(seed)
    extra = {}

    if subject == "case":
        # The documents are the contest, so there is no document count to draw:
        # one round each, read by the model the log currently ranks first.
        reader = _leader(standouts, "ocr", readers)
        if not reader:
            raise ValueError("No model with a score to read them with. Run a "
                             "random test first.")
        chosen = [{"case": name,
                   "seen": int((history or {}).get(name, 0) or 0)}
                  for name in picked["models"]][:MAX_ROUNDS]
        combos = [(reader, "")]
        extra = {"reader": reader}
    else:
        documents = max(1, min(int(documents or 1), len(cases)))
        if subject == "pipeline":
            # Two rankings, one round each way round. The extraction pool
            # excludes "" -- "same as reading" is a one-model run, which is what
            # the ocr_model subject already measures.
            second = contenders(_ranked_for(standouts, "extract_model"),
                                [m for m in (extractors or []) if m], top, bottom)
            if not second["models"]:
                raise ValueError("No extraction model with a score to pair "
                                 "with. Run a fields test first.")
            first_side, second_side = list(picked["models"]), list(second["models"])
            # Trim from the tail until the whole thing fits. The tail is the
            # bottom of each ranking, so what a clamp costs is the least
            # interesting half of the question rather than the leaders.
            while (len(first_side) * len(second_side) * documents > MAX_ROUNDS
                   and (len(first_side) > 1 or len(second_side) > 1)):
                if len(first_side) >= len(second_side) and len(first_side) > 1:
                    first_side.pop()
                elif len(second_side) > 1:
                    second_side.pop()
                else:
                    break
            documents = max(1, min(documents,
                                   MAX_ROUNDS // max(1, len(first_side) * len(second_side))))
            combos = [(r, e) for r in first_side for e in second_side]
            extra = {"readers": first_side, "extractors": second_side,
                     "extract_top": second["top"], "extract_bottom": second["bottom"],
                     "dropped": (len(picked["models"]) * len(second["models"])
                                 - len(combos))}
        else:
            documents = max(1, min(documents,
                                   MAX_ROUNDS // max(1, len(picked["models"]))))
            combos = [((name, "") if subject == "ocr_model" else ("", name))
                      for name in picked["models"]]
        chosen = case_order(documents, cases, history, rng)

    planned = []
    for reader, extractor in combos:
        for case in chosen:
            name = case["case"] if subject == "case" else (reader or extractor)
            round_ = {**case, "scope": scope, "reader": "", "profile": "",
                      "extractor": "", "detail": "", "mode": "",
                      # Which end of the ranking this contender came from, for
                      # the table. Not used by the runner.
                      "contender": ("top" if name in picked["top"]
                                    else "bottom" if name in picked["bottom"]
                                    else "")}
            if scope != "fields":
                round_["reader"] = reader
                round_["profile"] = profile_for(reader)
                round_["detail"] = detail
            if scope != "ocr":
                round_["mode"] = mode
            if extractor:
                round_["extractor"] = extractor
            elif scope == "fields":
                round_["extractor"] = reader
            planned.append(round_)
    planned = planned[:MAX_ROUNDS]

    # The summary names only what the rounds actually pinned. A fields contest
    # that reported a Detail would be naming a setting no round of it used --
    # the same rule the rounds themselves follow.
    return {"seed": seed, "scope": scope, "rounds": planned,
            "contest": {**picked, **extra, "subject": subject,
                        "label": SUBJECTS[subject]["label"],
                        "documents": len(chosen),
                        "detail": detail if scope != "fields" else "",
                        "mode": mode if scope != "ocr" else ""}}


def summarise_round(result: dict) -> dict:
    """The few numbers worth showing per round, out of a whole OCR payload.

    Kept here beside `plan` so the CLI and the page report the same things: two
    readouts of one run that disagree about what it did are worse than one.
    """
    result = result or {}
    truth = result.get("truth") or {}
    extracted = result.get("extracted") or {}
    score = extracted.get("field_score") or {}
    p1 = (score.get("scalars") or {}).get("p1") or {}
    counts = p1.get("counts") or {}
    expected = p1.get("expected") or 0
    correct, partial = counts.get("correct", 0), counts.get("partial", 0)
    entries = (extracted.get("fields") or {}).get("other_fields") or []
    return {
        "read_by": result.get("model") or "",
        "extract_by": extracted.get("model") or "",
        "char_accuracy": truth.get("char_accuracy"),
        "status": result.get("status") or "ok",
        # Half credit for a partial, the same rule the setting tables rank on.
        "field": (round(100.0 * (correct + 0.5 * partial) / expected, 1)
                  if expected else None),
        "correct": correct,
        "partial": partial,
        "expected": expected,
        "other": len(entries),
        "partial_reply": bool(extracted.get("partial")),
        "error": extracted.get("error") or "",
        # Set where the read was too poor for its fields to be scored -- the
        # extraction ran, and what it scored would have been the read's fault.
        # Carried as the reason rather than a flag: "not scored" without a
        # because reads as a defect in the harness.
        "unscored": result.get("fields_unscored") or "",
    }


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------

def _call(app_url, path, body=None, form=None, timeout=1800):
    if form is not None:
        data = urllib.parse.urlencode(form).encode()
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
    else:
        data = json.dumps(body).encode()
        headers = {"Content-Type": "application/json"}
    request_ = urllib.request.Request(app_url + path, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request_, timeout=timeout) as response:
            return json.loads(response.read().decode()), None
    except urllib.error.HTTPError as err:
        try:
            return json.loads(err.read().decode()), err.code
        except ValueError:
            return {}, err.code


def _describe(round_: dict) -> str:
    """One round's settings on one line, naming only what its scope used.

    Shared shape with the page's table, for the reason `summarise_round` gives:
    two readouts of one run that disagree about what it did are worse than one.
    """
    scope = round_.get("scope", DEFAULT_SCOPE)
    reader = (round_.get("reader") or "").split("/")[-1]
    extractor = (round_.get("extractor") or "").split("/")[-1]
    rank = f"[{round_['contender']}] " if round_.get("contender") else ""
    if scope == "fields":
        return f"{rank}fields only - {extractor} - {round_.get('mode')}"
    if scope == "ocr":
        return (f"{rank}read only - {reader} ({round_.get('profile')}) - "
                f"{round_.get('detail')}")
    return (f"{rank}{reader} ({round_.get('profile')}) -> {extractor or 'same'} - "
            f"{round_.get('detail')} - {round_.get('mode')}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("app", nargs="?", default="http://localhost:5000")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--scope", choices=SCOPES, default=DEFAULT_SCOPE,
                        help="full: read and extract. ocr: read only. "
                             "fields: extract from solution/<id>.md only.")
    parser.add_argument("--contest", action="store_true",
                        help=f"re-run the top {CONTEST_TOP} and bottom "
                             f"{CONTEST_BOTTOM} models from the run log's "
                             f"ranking on the same documents at "
                             f"{CONTEST_DETAIL} detail, instead of a random plan")
    parser.add_argument("--documents", type=int, default=1,
                        help="how many documents each contender runs (contest only)")
    parser.add_argument("--subject", choices=sorted(SUBJECTS), default=DEFAULT_SUBJECT,
                        help="what a contest is between (contest only). The "
                             "subject decides the scope as well")
    parser.add_argument("--top", type=int, default=CONTEST_TOP,
                        help="how many from each end of the ranking (contest only)")
    parser.add_argument("--no-bottom", action="store_true",
                        help="run the leaders alone, without the trailing end")
    parser.add_argument("--exclude-reader", action="append", default=[],
                        metavar="MODEL",
                        help="never draw this model to read a page; repeatable")
    parser.add_argument("--exclude-extractor", action="append", default=[],
                        metavar="MODEL",
                        help="never draw this model to extract; repeatable")
    parser.add_argument("--lock-case", default=None,
                        help="run every round on this document instead of "
                             "spreading them")
    parser.add_argument("--lock-reader", default=None,
                        help="read every page with this model")
    parser.add_argument("--lock-extractor", default=None,
                        help="extract with this model in every round")
    parser.add_argument("--lock-mode", choices=("single", "agentic"), default=None,
                        help="extract in this shape in every round")
    args = parser.parse_args(argv)

    lock = {key: value for key, value in
            (("case", args.lock_case), ("reader", args.lock_reader),
             ("extractor", args.lock_extractor), ("mode", args.lock_mode))
            if value}
    # Sent with both shapes, like the page does: an exclusion is a statement
    # about the run, not about which button started it.
    exclude = {"readers": args.exclude_reader,
               "extractors": args.exclude_extractor}

    request_body = ({"contest": True, "documents": args.documents,
                     "seed": args.seed, "subject": args.subject,
                     "top": args.top, "exclude": exclude,
                     "bottom": 0 if args.no_bottom else args.top} if args.contest
                    else {"rounds": args.rounds, "seed": args.seed,
                          "scope": args.scope, "lock": lock,
                          "exclude": exclude})
    body, code = _call(args.app, "/api/randomtest", request_body, timeout=120)
    if code:
        print(f"could not plan: {body.get('error', code)}")
        return 1
    repeat = (f"--contest --subject {args.subject}" if args.contest
              else f"--scope {args.scope}")
    print(f"seed {body['seed']} - repeat with --seed {body['seed']} "
          f"{repeat}" + chr(10))
    if args.contest and body.get("contest"):
        # What the contest assembled, before it starts: which models it found at
        # each end of the ranking and which it could not run here. A contest that
        # silently dropped half its contenders would look like a short run.
        c = body["contest"]
        print(f"contest ({c['label']}): {len(c['models'])} contender(s) x "
              f"{c['documents']} document(s) at {c['detail'] or 'no image'}"
              + (f", {len(c['missing'])} ranked model(s) not served here"
                 if c["missing"] else ""))
        for name in c["models"]:
            end = ("top" if name in c["top"]
                   else "bottom" if name in c["bottom"] else "")
            print(f"  {name}" + (f"  [{end}]" if end else ""))
        print()

    request_ = urllib.request.Request(
        args.app + "/api/randomtest/stream",
        data=json.dumps({**request_body, "seed": body["seed"]}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request_, timeout=None) as stream:
        for line in stream:
            line = line.decode().strip()
            if not line:
                continue
            event = json.loads(line)
            if event.get("event") == "round":
                plan_, out = event["plan"], event.get("result") or {}
                seen = plan_.get("seen")
                print(f"[{event['index']}/{event['total']}] {plan_['case']}"
                      + ("" if seen is None else f" (read {seen}x before)")
                      + " - " + _describe(plan_))
                if event.get("error"):
                    print(f"     FAILED: {event['error']}")
                else:
                    # A line per pass, and only for the passes the scope ran: a
                    # field score printed under a read-only round reads as an
                    # extraction that found nothing.
                    scope = plan_.get("scope", DEFAULT_SCOPE)
                    char = out.get("char_accuracy")
                    read = "unscored" if char is None else f"{char * 100:.1f}%"
                    field = "n/a" if out.get("field") is None else f"{out['field']:.1f}%"
                    if scope != "fields":
                        print(f"     read  {read:>8}  status={out.get('status')}"
                              f"  {event.get('seconds', 0):.0f}s")
                    if scope != "ocr" and out.get("unscored"):
                        # The counts are dropped with the score, but what came
                        # back is still worth seeing: the extraction ran.
                        print(f"     field {'unscored':>8}  ({out['unscored']})"
                              f"  other={out.get('other')}")
                    elif scope != "ocr":
                        print(f"     field {field:>8}"
                              f"  ({out.get('correct')}+{out.get('partial')}p/{out.get('expected')})"
                              f"  other={out.get('other')}"
                              + ("  PARTIAL" if out.get("partial_reply") else "")
                              + (f"  {event.get('seconds', 0):.0f}s"
                                 if scope == "fields" else ""))
                print(flush=True)
            elif event.get("event") == "done":
                print(f"done: {event['completed']}/{event['total']} rounds, "
                      f"{event['failed']} failed, {event['seconds']:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
