"""vLLM's batch route for the stress test: requests in flight together, sent as one.

**Built 2026-10-20 at the user's request** -- *allow stress test to activate
batch request to vllm (vllm have /batch)*, as a toggle. vLLM serves
`POST /v1/chat/completions/batch`: `messages` is a LIST of conversations, every
other field (max_tokens, sampling, response_format ...) is shared by all of
them, and the reply is one ordinary chat completion with one choice per
conversation, `index` 0..N-1, and `usage` summed. It does not stream.

The stress runner's workers each send one request at a time, the same calls a
real run makes. With batching on, a request bound for a vLLM
`/v1/chat/completions` is handed to a `Batcher` instead: requests whose
non-message fields are identical are collected into one group, and the group is
sent as one batch request when it holds `size` of them (the run's concurrency)
or `wait` seconds after its first one arrived, whichever is sooner. Each worker
gets back its own conversation's choice, shaped as the single reply it would
have had.

**What a batched request cannot give**, and the report says so rather than
filling it in:

* no stream, so no time to first token and no per-request decode rate, and the
  loop guard cannot cut a reply off mid-way -- a looping page runs to its cap,
  and is still flagged looped from the finished text;
* `usage` is the batch's total. A conversation's own completion tokens are read
  from `token_ids` (asked for with `return_token_ids`); where the server does
  not send them, and for prompt tokens, which it never splits, the total is
  shared out and the reply carries `usage_estimated`;
* one `cache_salt` per batch. With "Bypass prompt cache" on, the conversations
  of one batch may share a cached prefix with each other, never with anything
  sent before.

This module holds no knowledge of the app: `send(url, body)` is handed in, so
the grouping and the split are tested without a server.
"""

import json
import secrets
import threading

BATCH_SUFFIX = "/batch"
CHAT_PATH = "/v1/chat/completions"
# How long a group waits for company before it is sent short. Small against any
# model request; long enough that N workers released together land in one batch.
WAIT_SECONDS = 0.2
MAX_SIZE = 256

# Fields that differ per request and do not stop two requests sharing a batch:
# the conversation itself, the streaming switches (a batch never streams) and
# the per-request cache salt, which the batch replaces with one of its own.
_PER_REQUEST = ("messages", "stream", "stream_options", "cache_salt")


class BatchError(ValueError):
    """A batch that failed, for every request in it. `status_code` is the HTTP
    status where the server answered one, else 0."""

    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code


def batchable(url: str) -> bool:
    return bool(url) and url.rstrip("/").endswith(CHAT_PATH)


def batch_url(url: str) -> str:
    return url.rstrip("/") + BATCH_SUFFIX


def group_key(url: str, payload: dict) -> str:
    shared = {k: v for k, v in payload.items() if k not in _PER_REQUEST}
    return url + "\n" + json.dumps(shared, sort_keys=True, default=str)


def batch_body(payloads: list) -> dict:
    """One batch request from requests that share every non-message field."""
    first = payloads[0]
    body = {k: v for k, v in first.items() if k not in _PER_REQUEST}
    body["messages"] = [p["messages"] for p in payloads]
    body["return_token_ids"] = True
    if any("cache_salt" in p for p in payloads):
        body["cache_salt"] = secrets.token_urlsafe(32)
    return body


def _share(total, weights: list) -> list:
    """`total` split in proportion to `weights`, whole numbers summing to it."""
    if total is None:
        return [None] * len(weights)
    total = int(total)
    weight = sum(weights)
    if weight <= 0:
        weights, weight = [1] * len(weights), len(weights)
    out = [total * w // weight for w in weights]
    for i in range(total - sum(out)):           # the remainder, one each
        out[i % len(out)] += 1
    return out


def split_reply(body: dict, n: int) -> list:
    """The batch reply as `n` single replies, in submission order.

    Raises `BatchError` when the reply does not hold one choice per conversation.
    """
    choices = sorted((body or {}).get("choices") or [],
                     key=lambda c: c.get("index", 0))
    if len(choices) != n:
        raise BatchError(f"the batch reply held {len(choices)} choice(s) for "
                         f"{n} conversation(s)")
    usage = body.get("usage") or {}
    texts = [((c.get("message") or {}).get("content") or "") for c in choices]
    counted = [c.get("token_ids") for c in choices]
    estimated = False
    if all(isinstance(t, list) for t in counted):
        completion = [len(t) for t in counted]
    else:
        completion = _share(usage.get("completion_tokens"), [len(t) for t in texts])
        estimated = usage.get("completion_tokens") is not None
    prompt = _share(usage.get("prompt_tokens"), [1] * n)
    estimated = estimated or (n > 1 and usage.get("prompt_tokens") is not None)
    out = []
    for i, choice in enumerate(choices):
        # `token_ids` stays: a page read replays DRY from it (dryreplay.py).
        choice = dict(choice, index=0)
        mine = {"prompt_tokens": prompt[i], "completion_tokens": completion[i]}
        if mine["prompt_tokens"] is not None and mine["completion_tokens"] is not None:
            mine["total_tokens"] = mine["prompt_tokens"] + mine["completion_tokens"]
        one = {k: v for k, v in body.items() if k not in ("choices", "usage")}
        one.update(choices=[choice],
                   usage={k: v for k, v in mine.items() if v is not None},
                   batch={"size": n, "position": i, "usage": usage})
        if estimated:
            one["usage_estimated"] = True
        out.append(one)
    return out


class Ticket:
    """One request's place in a batch: `result()` blocks for its reply.

    `close()` gives up on it -- the stress test's Stop hangs up every request in
    flight through `close`, and a batch in flight is the server's, not this
    request's, to abandon. The batch carries on; this waiter is released.
    """

    def __init__(self, payload: dict):
        self.payload = payload
        self._done = threading.Event()
        self._body = None
        self._error = None

    def _resolve(self, body=None, error=None):
        if self._done.is_set():
            return
        self._body, self._error = body, error
        self._done.set()

    def close(self):
        self._resolve(error=BatchError("stopped"))

    def result(self, timeout: float = None) -> dict:
        if not self._done.wait(timeout):
            raise BatchError("the batch did not answer in time")
        if self._error is not None:
            raise self._error
        return self._body


class Batcher:
    """Collects requests into batches. Thread-safe; one per stress run.

    `send(url, body)` posts one batch and returns the parsed reply, raising
    `BatchError` (or anything else) when it fails. It is called on a thread of
    its own, so a slow batch never holds up the next one filling.
    """

    def __init__(self, size: int, send, wait: float = WAIT_SECONDS):
        self.size = max(1, min(int(size), MAX_SIZE))
        self.wait = wait
        self._send = send
        self._lock = threading.Lock()
        self._groups = {}          # key -> {"url", "tickets", "timer"}
        self._closed = False
        self._counts = {"batches": 0, "requests": 0, "max_size": 0, "failed": 0}

    def submit(self, url: str, payload: dict) -> Ticket:
        ticket = Ticket(payload)
        key = group_key(url, payload)
        ready = None
        with self._lock:
            if self._closed:
                ticket._resolve(error=BatchError("batching has ended"))
                return ticket
            group = self._groups.get(key)
            if group is None:
                group = self._groups[key] = {"url": url, "tickets": []}
                timer = threading.Timer(self.wait, self._flush, (key, group))
                timer.daemon = True
                group["timer"] = timer
                timer.start()
            group["tickets"].append(ticket)
            if len(group["tickets"]) >= self.size:
                ready = self._take(key, group)
        if ready:
            self._launch(*ready)
        return ticket

    def _take(self, key, group):
        """Detach `group` if it is still the open one. Caller holds the lock."""
        if self._groups.get(key) is not group:
            return None
        del self._groups[key]
        group["timer"].cancel()
        return group["url"], group["tickets"]

    def _flush(self, key, group):
        with self._lock:
            ready = self._take(key, group)
        if ready:
            self._launch(*ready)

    def _launch(self, url, tickets):
        threading.Thread(target=self._run, args=(url, tickets), daemon=True,
                         name="vllm-batch").start()

    def _run(self, url, tickets):
        with self._lock:
            self._counts["batches"] += 1
            self._counts["requests"] += len(tickets)
            self._counts["max_size"] = max(self._counts["max_size"], len(tickets))
        try:
            reply = self._send(batch_url(url), batch_body([t.payload for t in tickets]))
            bodies = split_reply(reply, len(tickets))
        except Exception as err:                    # noqa: BLE001
            with self._lock:
                self._counts["failed"] += 1
            error = err if isinstance(err, BatchError) else BatchError(
                f"{type(err).__name__}: {err}")
            for ticket in tickets:
                ticket._resolve(error=error)
            return
        for ticket, body in zip(tickets, bodies):
            ticket._resolve(body=body)

    def reset_counts(self):
        with self._lock:
            for key in self._counts:
                self._counts[key] = 0

    def summary(self) -> dict:
        """Batches sent, requests in them, mean and largest size, failures."""
        with self._lock:
            c = dict(self._counts)
        c["size"] = self.size
        c["mean_size"] = round(c["requests"] / c["batches"], 2) if c["batches"] else None
        return c

    def close(self):
        """Send whatever is still waiting, and take no more."""
        with self._lock:
            self._closed = True
            ready = [self._take(key, group) for key, group in list(self._groups.items())]
        for item in ready:
            if item:
                self._launch(*item)


def _selftest():
    p = {"messages": [{"role": "user", "content": "a"}], "max_tokens": 8,
         "stream": True, "cache_salt": "x"}
    q = dict(p, messages=[{"role": "user", "content": "b"}], cache_salt="y")
    assert group_key("u", p) == group_key("u", q)
    assert group_key("u", p) != group_key("u", dict(q, max_tokens=9))
    body = batch_body([p, q])
    assert body["messages"] == [p["messages"], q["messages"]] and "stream" not in body
    assert body["cache_salt"] not in ("x", "y")
    reply = {"model": "m", "choices": [
        {"index": 1, "message": {"content": "bb"}, "token_ids": [1, 2, 3]},
        {"index": 0, "message": {"content": "a"}, "token_ids": [4]}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 4}}
    one, two = split_reply(reply, 2)
    assert one["choices"][0]["message"]["content"] == "a"
    assert one["usage"]["completion_tokens"] == 1 and two["usage"]["completion_tokens"] == 3
    assert one["usage"]["prompt_tokens"] + two["usage"]["prompt_tokens"] == 11
    assert _share(10, [1, 3]) == [3, 7] and sum(_share(7, [1, 1, 1])) == 7
    assert batch_url("http://h/v1/chat/completions") == "http://h/v1/chat/completions/batch"
    t = Ticket({})
    t.close()
    try:
        t.result(0)
        raise AssertionError("a closed ticket must raise")
    except BatchError:
        pass


_selftest()
