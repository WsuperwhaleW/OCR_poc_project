"""Workers seek a confirmed start before owning any part of a long PDF.

No nominal slice edge becomes a document boundary. Previous owners must agree
with each handoff, and reconciliation refuses holes, duplicates and conflicts.
"""
import threading
from concurrent.futures import Future

import segment


class Conflict(RuntimeError):
    pass


class Questions:
    """One file's bounded, exact-conversation question cache, also used on fallback."""
    def __init__(self, limit):
        self.limit = limit
        self.lock = threading.Lock()
        self.cache = {}
        self.sent = self.answered = self.refused = self.hits = self.denied = 0
        self.waits = 0

    def has_capacity(self):
        with self.lock:
            return self.sent < self.limit

    def call(self, key, request, reserve=0):
        with self.lock:
            future = self.cache.get(key)
            owner = future is None
            if owner:
                if self.sent >= max(0, self.limit - reserve):
                    if self.sent >= self.limit:
                        self.denied += 1
                    else:
                        self.waits += 1
                    return False, None
                future = self.cache[key] = Future()
                self.sent += 1
            else:
                self.hits += 1
        if owner:
            try:
                answer = request()
            except BaseException as err:
                future.set_exception(err)
                raise
            with self.lock:
                if answer is None:
                    self.refused += 1
                else:
                    self.answered += 1
            future.set_result(answer)
        return True, future.result()

    def stats(self):
        with self.lock:
            return {'questions_sent':self.sent, 'questions_answered':self.answered,
                    'questions_refused':self.refused, 'question_cache_hits':self.hits,
                    'question_budget_denied':self.denied, 'question_budget_waits':self.waits}


class Coordinator:
    def __init__(self, pages, workers, overlap, classify, make_chat, min_confidence=0,
                 progress=None):
        self.pages, self.classify, self.make_chat = pages, classify, make_chat
        self.min_confidence, self.progress = min_confidence, progress
        self.starts = sorted({max(0, len(pages) * i // workers - (overlap if i else 0))
                              for i in range(workers)})
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.anchors = {1: 0}
        self.checked = set()
        self.leader_finished_file = False
        self.leader_page = 0
        self.last_pages = {}
        self.verified = {1}
        self.handoffs = {}
        self.deferred = {}
        self.conflict_reason = ''
        self.used_owners = set()

    def _verify(self, first, following, decision):
        if decision != (True, True):
            self.conflict_reason = 'workers disagree at page %d' % following
            self.cancel.set()
            return
        if decision == (True, True):
            self.handoffs[following] = first
            changed = True
            while changed:
                before = len(self.verified)
                self.verified.update(n for n, prior in self.handoffs.items() if prior in self.verified)
                changed = len(self.verified) != before

    def reserve(self, index):
        """Seekers may use spare questions, never the forward owner's reserve."""
        with self.lock:
            trusted = [owner for first, owner in self.anchors.items() if first in self.verified]
            if index in trusted:
                return 0
            frontier = max((min(self.last_pages.get(owner, 0),
                               self.deferred.get(owner, len(self.pages))) for owner in trusted), default=0)
            return max(0, len(self.pages) - frontier)

    def budget_deferred(self, index):
        with self.lock:
            if index in self.anchors.values():
                self.deferred.setdefault(index, self.last_pages.get(index, 0))

    def wait_budget(self, index):
        if self.cancel.wait(.02):
            raise Conflict(self.conflict_reason or 'grouping closed')
        with self.lock:
            return not (self.leader_finished_file and index not in self.anchors.values())

    def worker(self, index):
        seed = self.starts[index]
        owned = 1 if index == 0 else None
        if seed and (segment.page_position(self.pages[seed]) or (0, 0))[0] == 1:
            with self.lock:
                if seed + 1 not in self.anchors:
                    owned = seed + 1
                    self.anchors[owned] = index
        decisions, events = {}, []

        def boundary(number, new, certain, why):
            if self.cancel.is_set():
                raise Conflict(self.conflict_reason or 'grouping closed')
            with self.lock:
                if new is not None:
                    decisions[number] = (new, certain)
                if owned is None:
                    return self.leader_finished_file
                following = min((n for n in self.anchors if n > owned), default=None)
                if following and (number > following or (number == following and new is not None)):
                    self._verify(owned, following, decisions.get(following))
                return bool(following and (number > following or
                            (number == following and new is not None)))

        stream = segment.walk(self.pages, self.classify, ask=self.make_chat(index),
                              min_confidence=self.min_confidence, start=seed,
                              stop_boundary=boundary)
        try:
            while True:
                try:
                    step = next(stream)
                except StopIteration as done:
                    groups = done.value
                    break
                events.append(step)
                with self.lock:
                    self.checked.add(step['page'])
                    self.last_pages[index] = step['page']
                    if index == 0:
                        self.leader_page = step['page']
                    if owned is None and step['page'] > seed + 1 and step['opened'] and step['certain']:
                        # The initial fragment is never assigned to this worker.
                        if step['page'] in self.anchors:
                            break
                        owned = step['page']
                        self.anchors[owned] = index
                    if self.progress:
                        self.progress({'pages_checked':len(self.checked), 'starts_found':len(self.anchors)})
        finally:
            stream.close()
        # A duplicate seeker has no owned pages and no result to reconcile.
        if owned is None:
            groups = []
        if index == 0 and groups and groups[-1]['pages'][-1] == len(self.pages):
            with self.lock:
                self.leader_finished_file = True
        return {'owned':owned, 'groups':groups, 'events':events, 'decisions':decisions}

    def assemble(self, results, whole=None):
        leader = results[0]
        if [n for group in leader['groups'] for n in group['pages']] == list(range(1, len(self.pages) + 1)):
            self.used_owners = {0}
            if len(leader['groups']) == 1 and whole is not None:
                leader['groups'][0]['text'] = whole
            return leader['groups'], leader['events']
        anchors = sorted(self.anchors)
        groups, steps = [], []
        for position, first in enumerate(anchors):
            self.used_owners.add(self.anchors[first])
            if self.anchors[first] in self.deferred:
                raise Conflict('a speculative worker deferred questions needed for a verified handoff')
            result = results[self.anchors[first]]
            following = anchors[position + 1] if position + 1 < len(anchors) else len(self.pages) + 1
            if following <= len(self.pages) and result['decisions'].get(following) != (True, True):
                raise Conflict('workers disagree at page %d' % following)
            chosen = [g for g in result['groups'] if first <= g['pages'][0] < following]
            if not chosen or chosen[0]['pages'][0] != first:
                raise Conflict('worker did not establish its starting page')
            for group in chosen:
                if group['pages'][-1] >= following:
                    raise Conflict('a handoff cuts through a document')
                if first != 1 and group['pages'][0] == first and first == self.starts[self.anchors[first]] + 1:
                    group['start_reason'] = 'the page numbers itself 1 of %d' % segment.page_position(self.pages[first - 1])[1]
                groups.append(group)
            steps.extend(s for s in result['events'] if first <= s['page'] < following)
        if [n for group in groups for n in group['pages']] != list(range(1, len(self.pages) + 1)):
            raise Conflict('grouping would omit or duplicate pages')
        by_page = {s['page']:s for s in steps}
        events = []
        for doc, group in enumerate(groups, 1):
            for n in group['pages']:
                step = by_page[n]
                events.append({**step, 'document':doc, 'doc_pages':[p for p in group['pages'] if p <= n],
                               'doc_codes':list(group['codes']),
                               'why':group['start_reason'] if n == group['pages'][0] else step['why']})
        if len(groups) == 1 and whole is not None:
            groups[0]['text'] = whole
        return groups, events
