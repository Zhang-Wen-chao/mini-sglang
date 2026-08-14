"""Continuous batching scheduler: request queue, KV block management, preemption.

Corresponds to SGLang's `sglang/srt/managers/scheduler.py`, simplified:

- FCFS waiting queue; lower request id = higher priority.
- A request prefills its whole prompt in one step (no chunked prefill); then
  decodes one token per step. Block size = radix cache page size.
- Only COMPLETE blocks enter the radix cache. A request's block table is
  `[radix-covered blocks] + [at most one private partial block]`.
- Memory pressure: evict LRU radix nodes first; if still short, preempt the
  lowest-priority running request (recompute mode: drop its KV, re-queue it,
  re-prefill later).
- `step()` produces a batch; the engine runs the model; `finish_step()` does
  all bookkeeping (KV completion, radix insert, sampling-agnostic advance).
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import List, Optional

from .kv_pool import KVBlockPool
from .radix_cache import MatchResult, RadixCache


@dataclass
class Request:
    rid: int
    prompt: List[int]
    priority: int
    token_ids: List[int] = field(default_factory=list)  # prompt + generated
    block_table: List[int] = field(default_factory=list)  # physical blocks, logical order
    past_len: int = 0  # tokens whose KV has been written (attention coverage)
    cached_len: int = 0  # tokens covered by the request's radix-cache path
    last_node: Optional[object] = None  # deepest radix node the request refs
    status: str = "waiting"  # waiting | running | finished
    prefill_done: bool = False
    num_generated: int = 0
    num_preemptions: int = 0
    finish_reason: Optional[str] = None

    def __post_init__(self):
        if not self.token_ids:
            self.token_ids = list(self.prompt)

    @property
    def is_running(self) -> bool:
        return self.status == "running"


@dataclass
class ScheduledReq:
    """One request's forward work in one step."""

    req: Request
    input_ids: List[int]  # tokens to process now
    is_prefill: bool  # False: decode-style single-token forward
    block_table: List[int]  # snapshot of the request's block table
    write_offset: int  # logical position of input_ids[0]


class Scheduler:
    def __init__(
        self,
        kv_pool: KVBlockPool,
        radix_cache: RadixCache,
        max_running: int,
        max_new_tokens: int = 16,
        eos_id: Optional[int] = None,
    ):
        self.kv_pool = kv_pool
        self.radix_cache = radix_cache
        self.block_size = kv_pool.block_size
        assert radix_cache.page_size == self.block_size, (
            "radix cache page size must equal the KV block size"
        )
        self.max_running = max_running
        self.max_new_tokens = max_new_tokens
        self.eos_id = eos_id
        self.waiting: collections.deque[Request] = collections.deque()
        self.running: List[Request] = []  # kept sorted by priority (best first)
        self.finished: List[Request] = []
        self._rid = 0
        self.stats = {
            "prefill_tokens": 0,
            "decode_tokens": 0,
            "cached_tokens": 0,
            "preemptions": 0,
            "evictions": 0,
        }

    # ---- public API -----------------------------------------------------

    def submit(self, prompt_ids: List[int]) -> Request:
        assert prompt_ids, "empty prompt"
        req = Request(rid=self._rid, prompt=list(prompt_ids), priority=self._rid)
        self._rid += 1
        self.waiting.append(req)
        return req

    def is_done(self) -> bool:
        return not self.waiting and not self.running

    def step(self) -> List[ScheduledReq]:
        """Admit waiting requests, then build the forward batch for this step."""
        # ---- admission (FCFS, within max_running and block budget) ----
        while self.waiting and len(self.running) < self.max_running:
            req = self.waiting.popleft()
            if not self._admit(req):
                self.waiting.appendleft(req)
                break  # can't fit the highest-priority waiting request
        # ---- batch build ----
        batch: List[ScheduledReq] = []
        for req in list(self.running):
            if req.status != "running":
                continue  # preempted during admission
            if not req.prefill_done:
                input_ids = req.token_ids[req.past_len :]
                batch.append(
                    ScheduledReq(req, input_ids, True, list(req.block_table), req.past_len)
                )
            else:
                if req.past_len % self.block_size == 0:
                    keep = {id(s.req) for s in batch}  # never preempt the batch
                    if not self._free_blocks_needed(1, keep=keep):
                        continue  # this request waits a step
                    if req.status != "running":  # preempted above
                        continue
                    req.block_table.append(self.kv_pool.alloc(1)[0])
                tok = req.token_ids[req.past_len]
                batch.append(ScheduledReq(req, [tok], False, list(req.block_table), req.past_len))
        return batch

    def finish_step(self, batch: List[ScheduledReq], next_token_ids: List[int]) -> None:
        """Advance the batch: KV bookkeeping, radix insert, token append, finish."""
        assert len(batch) == len(next_token_ids)
        for sched, tok in zip(batch, next_token_ids):
            req = sched.req
            req.past_len += len(sched.input_ids)
            if sched.is_prefill:
                req.prefill_done = True
            self._cache_completed_blocks(req)
            req.token_ids.append(tok)
            req.num_generated += 1
            if self.eos_id is not None and tok == self.eos_id:
                self._finish(req, "eos")
            elif req.num_generated >= self.max_new_tokens:
                self._finish(req, "length")

    # ---- memory management ----------------------------------------------

    def _free_blocks_needed(
        self,
        n_blocks: int,
        worse_than: Optional[int] = None,
        keep: Optional[set] = None,
    ) -> bool:
        """Make `n_blocks` free blocks available; evict radix, then preempt.

        `worse_than` (a priority value): only preempt requests with strictly
        lower priority (used at admission to keep FCFS fairness).
        `keep`: requests that must not be preempted (already scheduled in the
        current batch).
        """
        while self.kv_pool.num_free() < n_blocks:
            need_tokens = (n_blocks - self.kv_pool.num_free()) * self.block_size
            evicted = self.radix_cache.evict(need_tokens, evict_callback=self.kv_pool.free)
            self.stats["evictions"] += evicted
            if evicted > 0:
                continue
            victim = self._lowest_priority_running(keep)
            if victim is None:
                return False
            if worse_than is not None and victim.priority <= worse_than:
                return False  # would break FCFS: keep waiting instead
            self._preempt(victim)
        return True

    def _lowest_priority_running(self, keep: Optional[set] = None) -> Optional[Request]:
        """Worst-priority running request, excluding any in `keep` (id set)."""
        for req in reversed(self.running):
            if keep is not None and id(req) in keep:
                continue
            return req
        return None

    def _preempt(self, req: Request) -> None:
        """Recompute preemption: drop the request's KV, re-queue it in front."""
        self.stats["preemptions"] += 1
        req.num_preemptions += 1
        self.running.remove(req)
        self.radix_cache.dec_ref_count(req.last_node)
        req.last_node = None
        req.cached_len = 0
        self.kv_pool.free(req.block_table[req.past_len // self.block_size :])
        req.block_table = []
        req.past_len = 0
        req.prefill_done = False
        req.status = "waiting"
        self.waiting.appendleft(req)

    # ---- admission ------------------------------------------------------

    def _admit(self, req: Request) -> bool:
        """Match the request's cached prefix, reserve blocks, move to running."""
        while True:
            match = self.radix_cache.match_prefix(req.token_ids)
            extend = len(req.token_ids) - match.matched_len
            needed = (extend + self.block_size - 1) // self.block_size
            if extend == 0:
                needed = 1  # fully cached prompt: one private block for the last token
            if self.kv_pool.num_free() >= needed:
                break
            if not self._free_blocks_needed(needed, worse_than=req.priority):
                return False  # no room without breaking FCFS: stay waiting
            # eviction may have changed the tree: re-match next iteration
        self.radix_cache.inc_ref_count(match.node)
        req.last_node = match.node
        req.cached_len = match.matched_len
        req.block_table = list(match.blocks)
        req.past_len = match.matched_len
        if extend == 0:
            # whole prompt cached: first forward is a decode of the last prompt
            # token into a private block (its shared block stays read-only)
            req.block_table = req.block_table[:-1] + self.kv_pool.alloc(1)
            req.past_len = len(req.token_ids) - 1
            req.prefill_done = True
        else:
            req.block_table.extend(self.kv_pool.alloc(needed))
            req.prefill_done = False
        req.status = "running"
        self.running.append(req)
        self.stats["cached_tokens"] += match.matched_len
        return True

    # ---- radix cache integration ----------------------------------------

    def _cache_completed_blocks(self, req: Request) -> None:
        """Insert newly completed blocks into the radix cache.

        Pages whose tokens the tree already had are duplicates: they are freed
        and the request's block table is rebound to the shared blocks.
        """
        bs = self.block_size
        completed = (req.past_len // bs) * bs
        if completed <= req.cached_len:
            return
        key = req.token_ids[:completed]
        value = req.block_table[: completed // bs]
        node, new_start = self.radix_cache.insert(key, value)
        dup_from = req.cached_len // bs
        dup_to = new_start // bs
        if dup_to > dup_from:
            self.kv_pool.free(req.block_table[dup_from:dup_to])
        match = self.radix_cache.match_prefix(key)
        req.block_table[: completed // bs] = match.blocks
        req.cached_len = completed
        self.radix_cache.dec_ref_count(req.last_node)
        self.radix_cache.inc_ref_count(match.node)
        req.last_node = match.node

    def _finish(self, req: Request, reason: str) -> None:
        req.status = "finished"
        req.finish_reason = reason
        self.running.remove(req)
        self.radix_cache.dec_ref_count(req.last_node)
        req.last_node = None
        req.cached_len = 0
        self.kv_pool.free(req.block_table[req.past_len // self.block_size :])
        req.block_table = []
        self.finished.append(req)
