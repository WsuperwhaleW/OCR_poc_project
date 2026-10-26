"""Did vLLM's DRY processor fire on a reply? Worked out here, from its token ids.

**Built 2026-10-26 at the user's request** -- *the stress test flag doc that
got dry too*. DRY runs inside the vLLM server
(../vllm/podman-ocr-server/dry_logits_processor.py) and a logits processor has
no way to put anything in the response; the server logs a line per request, on
a machine this app may not be on. But DRY is a pure function of the OUTPUT
token ids and the settings this app sent in `vllm_xargs`, so replaying it on
the ids the server returns (`return_token_ids`) gives the server's own answer,
step for step -- without its logits, so it says where DRY penalised the next
token, not what that token would otherwise have been.

**Keep this in step with the server's `DryState`.** It assumes the server's
default of no sequence breakers (`VLLM_DRY_SEQUENCE_BREAKERS` unset); a server
started with breakers fires less often than this says.
"""


def replay(token_ids, xargs: dict) -> dict:
    """What DRY did to one reply, given the `vllm_xargs` it was sent with.

    * `fired` -- steps where DRY penalised a token (a repeat of at least
      `allowed` tokens was running), and `first`, the output position of the first;
    * `steered` -- of those, steps where the token that came out was NOT the
      penalised one: the reply left the repeat while DRY was pushing it to.
      Under greedy decoding this is the most DRY can have changed;
    * `longest` -- the longest verbatim repeat in the reply, fired or not.
    """
    allowed = int(xargs.get("dry_allowed_length", 2))
    last_n = int(xargs.get("dry_penalty_last_n", -1))
    tokens, positions, match = [], {}, {}
    fired = steered = longest = 0
    first = None
    for token in token_ids:
        if token is None or token < 0:
            continue
        n = len(tokens)
        # What the server penalised for position n: the token after each
        # earlier position whose repeat, ending at the last token, is long enough.
        hit = {tokens[i + 1] for i, length in match.items() if length >= allowed}
        if hit:
            fired += 1
            if first is None:
                first = n
            if token not in hit:
                steered += 1
        lo = 0 if last_n < 0 else max(0, n + 1 - last_n)
        new = {}
        for i in reversed(positions.get(token, ())):
            if i < lo:
                break
            new[i] = min(match.get(i - 1, 0) + 1, i - lo + 1)
            if new[i] > longest:
                longest = new[i]
        match = new
        positions.setdefault(token, []).append(n)
        tokens.append(token)
    return {"checked": True, "tokens": len(tokens), "allowed": allowed,
            "fired": fired, "first": first, "steered": steered, "longest": longest}


def describe(result: dict) -> str:
    """One line for a person, e.g. for the stress test's alert."""
    if not result.get("fired"):
        return (f"not triggered, longest repeat {result.get('longest', 0)} "
                f"of {result.get('allowed')} allowed")
    return (f"penalised {result['fired']} step(s) from output token {result['first']}, "
            f"longest repeat {result['longest']}, left the repeat "
            f"{result['steered']} time(s)")
