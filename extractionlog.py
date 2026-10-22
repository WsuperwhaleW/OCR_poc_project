"""Download-only extraction diagnostics. No filesystem writes or model requests."""
import hashlib
from collections import Counter
from pathlib import Path


def provenance(prompt_file, app_file, settings):
    """Capture the code/prompt identity once, before the timed run."""
    files = {}
    for label, path in (("prompts", prompt_file), ("app", app_file)):
        try:
            data = Path(path).read_bytes()
            files[label] = {"sha256": hashlib.sha256(data).hexdigest()}
        except (OSError, UnicodeError) as err:
            files[label] = {"error": str(err)}
    return {"files": files, "settings": settings}


def _reference(catalog, section, text):
    """Identical document contexts and questions are stored once per download."""
    if not text:
        return None
    identity = hashlib.sha256(text.encode("utf-8")).hexdigest()
    catalog[section].setdefault(identity, text)
    return identity


def _step_detail(step, catalog):
    out = {key: step[key] for key in (
        "id", "keys", "values", "attempts", "retried", "schema_retry", "salvaged",
        "error", "retry_error", "extras_error", "ungrounded", "seconds", "chats") if key in step}
    out["question_id"] = _reference(catalog, "questions", step.get("prompt"))
    out["raw"] = [{**{key: reply[key] for key in ("label", "text") if key in reply},
                   "question_id": _reference(catalog, "questions", reply.get("prompt"))}
                  for reply in step.get("raw") or []]
    return out


def job_detail(record, spec, source, warmup=0, catalog=None):
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
    return {"job": spec["index"] + 1, "queue_position": spec["index"] - warmup + 1,
            "case": spec.get("case"), "source": source,
            "status": record.get("status"), "error": record.get("error", ""),
            "timing": {key: record[key] for key in ("wait_s", "total_s", "ocr_s", "extract_s")
                       if key in record},
            "field_rows": rows, "documents": documents}


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
    yield {"event": "detail_log_start", "format_version": 2, "provenance": identity}
    counts, catalog = {}, {"contexts": {}, "questions": {}}
    for record, spec in zip(records, specs):
        detail = job_detail(record, spec, sources.get(spec.get("case"), {}), warmup, catalog)
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
