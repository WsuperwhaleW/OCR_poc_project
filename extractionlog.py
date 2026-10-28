"""Download-only extraction diagnostics. No filesystem writes or model requests."""
import hashlib
import json
from collections import Counter
from pathlib import Path


DIAGNOSTICS_REVISION = 1


def _file_identity(path):
    try:
        return {"sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
    except (OSError, UnicodeError, TypeError) as err:
        return {"error": str(err)}


def provenance(prompt_file, app_file, settings, extra_files=None, benchmarks=None):
    """Capture the code/prompt identity once, before the timed run."""
    paths = {"prompts": prompt_file, "app": app_file, **(extra_files or {})}
    out = {"files": {label: _file_identity(path) for label, path in paths.items()},
           "settings": settings, "snapshot": "before_benchmark"}
    if benchmarks:
        snapshots = {}
        for case, info in benchmarks.items():
            entry = {name: _file_identity(info.get(name)) for name in ("transcript", "fields")}
            try:
                raw = json.loads(Path(info["fields"]).read_text("utf-8"))
                if isinstance(raw, dict):
                    docs = raw.get("documents") or []
                    if docs:
                        entry["expected_documents"] = [{"pages": d.get("pages") or [],
                            "doc_types": d.get("doc_types") or []} for d in docs if isinstance(d, dict)]
                    elif isinstance(info.get("pages"), int) and info["pages"] > 0:
                        entry["expected_documents"] = [{"pages": list(range(1, info["pages"] + 1)),
                                                       "doc_types": info.get("doc_types") or []}]
            except (OSError, ValueError, TypeError):
                pass  # Missing or old truth files still have an explicit hash error.
            snapshots[case] = entry
        out["benchmarks"] = snapshots
    return out


def _score_summary(score):
    """Keep denominators and averaging scope, without repeating field rows."""
    return {key: score[key] for key in (
        "overall", "optional", "coverage", "unknown_type", "scored_scope",
        "normalization_version", "documents", "scored_documents", "warnings", "error")
        if key in score}


def _step_stats(step, catalog):
    # Every step gets these small records, including fully correct steps.
    # Full replies/values stay in the existing problem-step evidence only.
    out = {key: step[key] for key in ("id", "keys", "request_aliases", "seconds", "attempts") if key in step}
    out["question_id"] = _reference(catalog, "questions", step.get("prompt"))
    if step.get("chats"):
        out["chats"] = [{key: chat[key] for key in (
            "request", "in_tokens", "out_tokens", "used_tokens", "reply_cap",
            "context_limit", "context_source", "used_pct", "seconds", "http_status",
            "schema", "error") if key in chat} for chat in step["chats"]]
    return out


def _alignment(documents, benchmark):
    """Flag spans crossing truth documents; do not change segmentation or scoring."""
    expected = (benchmark or {}).get("expected_documents") or []
    if not expected:
        return []
    issues = []
    for number, document in enumerate(documents, 1):
        pages = document.get("pages") or []
        if not pages:
            continue
        touched = [d.get("pages") or [] for d in expected
                   if set(pages).intersection(d.get("pages") or [])]
        selected = next((d.get("pages") or [] for d in expected
                         if pages[0] in (d.get("pages") or [])), [])
        if not selected or set(pages) != set(selected):
            issues.append({"document": document.get("document", number), "pages": pages,
                           "truth_pages_by_first_page": selected,
                           "overlapping_truth_documents": touched,
                           "crosses_truth_documents": len(touched) > 1})
    return issues


def _reference(catalog, section, text):
    """Identical document contexts and questions are stored once per download."""
    if not text:
        return None
    identity = hashlib.sha256(text.encode("utf-8")).hexdigest()
    catalog[section].setdefault(identity, text)
    return identity


def _step_detail(step, catalog):
    out = {key: step[key] for key in (
        "id", "keys", "request_aliases", "values", "attempts", "retried", "schema_retry", "salvaged",
        "error", "retry_error", "extras_error", "ungrounded", "seconds", "chats") if key in step}
    out["question_id"] = _reference(catalog, "questions", step.get("prompt"))
    out["raw"] = [{**{key: reply[key] for key in ("label", "text") if key in reply},
                   "question_id": _reference(catalog, "questions", reply.get("prompt"))}
                  for reply in step.get("raw") or []]
    return out


def job_detail(record, spec, source, warmup=0, catalog=None, benchmark=None):
    """Field outcomes for all steps, with response evidence only where useful for tuning."""
    catalog = catalog if catalog is not None else {"contexts": {}, "questions": {}}
    result = record.get("_extraction_result")
    rows, documents = [], []
    if isinstance(result, dict):
        segments = result.get("segments") or []
        for number, document in enumerate(result.get("documents") or [result], 1):
            fields = document.get("fields") or {}
            steps = document.get("steps") or []
            step_keys = {key: step["id"] for step in steps for key in step.get("keys") or []}
            pages = document.get("pages") or (segments[number - 1].get("pages")
                                             if number <= len(segments) else []) or []
            score = document.get("field_score") or {}
            judged = []
            for section in ("scalars", "line_items", "income_items", "other_fields"):
                judged.extend((score.get(section) or {}).get("rows") or [])
            checked = {row["path"] for row in judged}
            # Missing ground truth is unscored, never an empty expected value.
            judged += [{"path": key, "actual": value, "expected": None, "status": "unscored"}
                       for key, value in fields.items() if key not in checked
                       and not isinstance(value, (list, dict))]
            for row in judged:
                path = row["path"]
                key = path.split("[", 1)[0]
                row = dict(row)
                normalized = document.get("normalized") or {}
                if row["status"] == "unscored" and key in normalized and normalized[key] != row.get("actual"):
                    row["normalized_actual"] = normalized[key]
                rows.append({**row, "document": document.get("document", number),
                             "pages": pages, "doc_types": document.get("doc_types") or [],
                             "step": step_keys.get(key), "actual_returned": key in fields})
            problem_keys = {row["path"].split("[", 1)[0] for row in judged
                            if row["status"] not in ("correct", "absent")}
            evidence = [step for step in steps
                        if problem_keys.intersection(step.get("keys") or [])
                        or any(step.get(key) for key in ("error", "retry_error", "extras_error",
                                                       "retried", "schema_retry", "salvaged"))]
            entry = {"document": document.get("document", number), "pages": pages,
                     **{key: document[key] for key in (
                         "doc_types", "doc_type_expected", "doc_type_from", "doc_type_heading",
                         "doc_type_confidence", "doc_type_escalated", "error", "partial",
                         "field_truth_scope", "normalization_version") if document.get(key) is not None},
                     "context_id": _reference(catalog, "contexts",
                                              document.get("prompt_prefix") or document.get("prompt")),
                     "steps": [_step_detail(step, catalog) for step in evidence]}
            if score:
                entry["score"] = _score_summary(score)
            if document.get("validation_image_rechecks"):
                entry["validation_image_rechecks"] = document["validation_image_rechecks"]
            form = document.get("classified") or {}
            entry["form"] = {key: value for key, value in {
                "keys": document.get("fields_asked", form.get("fields")),
                "items": document.get("items_asked", form.get("items")),
                "mandatory": form.get("mandatory"),
                "mandatory_items": form.get("mandatory_items"),
                "unknown_type": form.get("unknown_type"),
            }.items() if value is not None}
            if steps:
                entry["step_stats"] = [_step_stats(step, catalog) for step in steps]
            # Tables not judged against ground truth still need their actual values
            # for manual inspection; scored table cells are already in field_rows.
            scored_roots = {key.split("[", 1)[0] for key in checked}
            unscored = {key: value for key, value in fields.items()
                        if isinstance(value, (list, dict)) and value and key not in scored_roots}
            if unscored:
                entry["unscored_fields"] = unscored
            if not steps and (problem_keys or document.get("error")):
                entry["raw"] = document.get("raw", "")
            documents.append(entry)
    detail = {"job": spec["index"] + 1, "queue_position": spec["index"] - warmup + 1,
            "case": spec.get("case"), "source": source,
            "status": record.get("status"), "error": record.get("error", ""),
            "timing": {key: record[key] for key in ("wait_s", "total_s", "ocr_s", "extract_s")
                       if key in record},
            "field_rows": rows, "documents": documents}
    if isinstance(result, dict) and result.get("field_score"):
        detail["score"] = _score_summary(result["field_score"])
    alignment = _alignment(documents, benchmark)
    if alignment:
        detail["document_alignment"] = alignment
    return detail


def field_summary(counts):
    """Pooled verdict counts, with invented values reported separately."""
    rows = []
    for field, verdicts in sorted(counts.items()):
        scored = sum(verdicts.get(key, 0) for key in ("correct", "partial", "wrong", "missed"))
        rows.append({"field": field, "counts": dict(verdicts), "scored": scored,
                     "accuracy": ((verdicts.get("correct", 0) + .5 * verdicts.get("partial", 0))
                                  / scored if scored else None)})
    return rows


def events(records, specs, sources, warmup, identity, report=None):
    """Serialize only after the runner stopped its benchmark clock; never save on disk."""
    yield {"event": "detail_log_start", "format_version": 2,
           "diagnostics_revision": DIAGNOSTICS_REVISION, "provenance": identity}
    counts, catalog = {}, {"contexts": {}, "questions": {}}
    for record, spec in zip(records, specs):
        detail = job_detail(record, spec, sources.get(spec.get("case"), {}), warmup, catalog,
                            (identity.get("benchmarks") or {}).get(spec.get("case")))
        for row in detail["field_rows"]:
            counts.setdefault(row["path"], Counter())[row["status"]] += 1
        yield {"event": "detail_log_job", "job": detail}
    yield {"event": "detail_log_end", "jobs": len(records),
           "field_summary": field_summary(counts), **catalog,
           "report": compact_report(report or {})}


def compact_report(report):
    """Keep run outcomes and latency, leaving server monitoring in History."""
    return {key: report[key] for key in (
        "mode", "jobs", "attempted", "completed", "counts", "error_rate", "errors",
        "wall_seconds", "jobs_per_minute", "extract") if key in report}


def compact_download(download):
    """Convert a previously downloaded v1 log locally, without rerunning inference."""
    if download.get("format_version") != 1:
        raise ValueError("Expected a version 1 extraction detail log.")
    counts, catalog, jobs = {}, {"contexts": {}, "questions": {}}, []
    for job in download.get("jobs") or []:
        record = {**(job.get("timing") or {}), "status": job.get("status"),
                  "error": job.get("error", ""), "_extraction_result": job.get("extraction")}
        warmup = job["job"] - job["queue_position"]
        detail = job_detail(record, {"index": job["job"] - 1, "case": job.get("case")},
                            job.get("source") or {}, warmup, catalog)
        # Preserve the original verdicts exactly, including old scoring behavior.
        detail["field_rows"] = job.get("field_rows") or detail["field_rows"]
        for row in detail["field_rows"]:
            counts.setdefault(row["path"], Counter())[row["status"]] += 1
        jobs.append(detail)
    identity = download.get("provenance") or {}
    identity = {**identity, "files": {name: {key: value for key, value in entry.items()
                if key != "source"} for name, entry in (identity.get("files") or {}).items()}}
    return {**{key: download[key] for key in ("format", "configuration", "plan", "completion")
               if key in download}, "format_version": 2, "provenance": identity,
            "jobs": jobs, "field_summary": field_summary(counts), **catalog,
            "report": compact_report(download.get("report") or {})}
