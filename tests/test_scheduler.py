"""Unit tests for the continuous batching scheduler (Phase 2).

Uses a deterministic token function instead of a model, so all KV-block,
radix-cache, preemption and reuse semantics are tested on their own.
"""

import pytest

from mini_sglang.kv_pool import KVBlockPool
from mini_sglang.radix_cache import RadixCache
from mini_sglang.scheduler import Scheduler

BS = 4  # small block size keeps pool numbers readable


def make_scheduler(num_blocks=8, max_running=2, max_new_tokens=5, eos_id=None):
    pool = KVBlockPool(num_blocks, BS, n_layers=1, n_kv_heads=1, head_dim=8)
    cache = RadixCache(page_size=BS)
    sched = Scheduler(pool, cache, max_running=max_running,
                      max_new_tokens=max_new_tokens, eos_id=eos_id)
    return sched


def token_fn(req):
    """Deterministic fake model output, different per request and per step."""
    return (req.rid * 5 + req.num_generated * 2) % 6 + 10


def run_to_completion(sched, fn=token_fn, max_steps=500):
    steps = 0
    while not sched.is_done():
        batch = sched.step()
        if batch:
            sched.finish_step(batch, [fn(s.req) for s in batch])
        steps += 1
        assert steps < max_steps, "scheduler did not converge"
    return sched


# ---- basic flow -----------------------------------------------------------


def test_prefill_then_decode_then_finish():
    sched = make_scheduler()
    req = sched.submit([1, 2, 3, 4, 5, 6])
    batch = sched.step()
    assert len(batch) == 1
    assert batch[0].is_prefill
    assert batch[0].input_ids == [1, 2, 3, 4, 5, 6]
    assert batch[0].write_offset == 0
    assert len(batch[0].block_table) == 2  # 6 tokens -> 2 blocks
    sched.finish_step(batch, [7])
    assert req.prefill_done and req.past_len == 6 and req.num_generated == 1

    batch = sched.step()
    assert len(batch) == 1 and not batch[0].is_prefill
    assert batch[0].input_ids == [7] and batch[0].write_offset == 6
    sched.finish_step(batch, [8])

    run_to_completion(sched)
    assert req.status == "finished"
    assert req.finish_reason == "length"
    assert req.num_generated == 5
    assert req.token_ids[:6] == [1, 2, 3, 4, 5, 6]


def test_block_allocation_counting():
    sched = make_scheduler()
    req = sched.submit([1, 2, 3, 4, 5, 6])
    batch = sched.step()
    blocks = batch[0].block_table
    assert len(blocks) == 2
    # prefill writes 6 tokens: block 0 full, block 1 partial
    sched.finish_step(batch, [7])
    # next decode token at position 6 -> writes into partial block, no alloc
    batch = sched.step()
    assert batch[0].block_table == blocks
    sched.finish_step(batch, [8])
    # past_len 7 -> next token at position 7 still fits block 1
    batch = sched.step()
    assert batch[0].block_table == blocks
    sched.finish_step(batch, [9])
    # past_len 8 -> next token at position 8 needs a fresh block
    batch = sched.step()
    assert batch[0].block_table == blocks + [max(blocks) + 1]
    sched.finish_step(batch, [10])


def test_eos_terminates_request():
    sched = make_scheduler(eos_id=11)
    req = sched.submit([1, 2, 3, 4])
    run_to_completion(sched, fn=lambda r: 11 if r.num_generated >= 2 else token_fn(r))
    assert req.status == "finished"
    assert req.finish_reason == "eos"
    assert req.token_ids[-1] == 11


def test_blocks_are_returned_on_finish():
    sched = make_scheduler()
    req = sched.submit([1, 2, 3, 4, 5, 6])
    run_to_completion(sched)
    # completed blocks are cached (evictable), the partial tail was freed
    assert sched.kv_pool.num_free() == 8 - 2
    assert sched.radix_cache.total_len() == 8  # [0,8) cached


# ---- radix cache integration ----------------------------------------------


def test_completed_blocks_inserted_into_radix_after_prefill():
    sched = make_scheduler()
    req = sched.submit([1, 2, 3, 4, 5, 6])
    batch = sched.step()
    sched.finish_step(batch, [7])
    # 4 complete tokens cached after prefill
    m = sched.radix_cache.match_prefix([1, 2, 3, 4])
    assert m.matched_len == 4
    assert len(m.blocks) == 1
    # block fills during decode -> inserted too
    run_to_completion(sched)
    m = sched.radix_cache.match_prefix(req.token_ids[:8])
    assert m.matched_len == 8
    assert len(m.blocks) == 2


def test_sequential_match_reuses_blocks():
    sched = make_scheduler()
    a = sched.submit([1, 2, 3, 4, 5, 6])
    batch = sched.step()
    sched.finish_step(batch, [token_fn(a)])
    a_block0 = a.block_table[0]  # captured before the request finishes
    run_to_completion(sched)
    cached_before = sched.radix_cache.total_len()
    assert cached_before >= 8
    b = sched.submit([1, 2, 3, 4, 7, 8])
    batch = sched.step()
    sched.finish_step(batch, [token_fn(b)])
    assert sched.stats["cached_tokens"] == 4
    # b's first block is physically the same block a used
    assert b.block_table[0] == a_block0
    # b's 2 extend tokens don't complete a block yet: tree unchanged
    assert sched.radix_cache.total_len() == cached_before


def test_concurrent_prefix_sharing_via_rebind():
    sched = make_scheduler()
    a = sched.submit([1, 2, 3, 4, 5, 6])
    b = sched.submit([1, 2, 3, 4, 7, 8])
    batch = sched.step()
    assert len(batch) == 2
    sched.finish_step(batch, [token_fn(a), token_fn(b)])
    # b's prefill duplicated A's completed prefix -> freed and rebound to the
    # shared physical block (duplicate handling in _cache_completed_blocks)
    assert b.block_table[0] == a.block_table[0]
    assert b.block_table[1] != a.block_table[1]
    # the shared node is referenced by both requests
    node = sched.radix_cache.match_prefix([1, 2, 3, 4]).node
    assert node.ref_count == 2
    run_to_completion(sched)
    assert node.ref_count == 0


def test_fully_cached_prompt_starts_as_decode():
    sched = make_scheduler()
    a = sched.submit([1, 2, 3, 4, 5, 6, 7, 8])
    run_to_completion(sched)
    b = sched.submit([1, 2, 3, 4, 5, 6, 7, 8])
    batch = sched.step()
    assert len(batch) == 1
    # no prefill: the first forward is a decode of the last prompt token
    assert not batch[0].is_prefill
    assert batch[0].input_ids == [8]
    assert batch[0].write_offset == 7
    assert sched.stats["cached_tokens"] == 8
    sched.finish_step(batch, [token_fn(b)])
    run_to_completion(sched)
    assert b.num_generated == 5


def test_unmatched_prompt_prefills_everything():
    sched = make_scheduler()
    a = sched.submit([1, 2, 3, 4])
    run_to_completion(sched)
    b = sched.submit([9, 9, 9, 9])
    batch = sched.step()
    assert batch[0].is_prefill and batch[0].input_ids == [9, 9, 9, 9]
    sched.finish_step(batch, [token_fn(b)])


# ---- memory pressure: eviction and preemption -----------------------------


def test_evict_before_admitting_when_cached_blocks_available():
    # single request at a time: finished requests leave cached blocks behind,
    # and admitting the next one evicts them instead of preempting
    sched = make_scheduler(num_blocks=4, max_running=1)
    a = sched.submit([1, 2, 3, 4])  # 1 block
    run_to_completion(sched)
    b = sched.submit([5, 6, 7, 8, 9, 10, 11, 12])  # 2 blocks + decode
    run_to_completion(sched)
    assert sched.stats["evictions"] > 0 or sched.stats["preemptions"] > 0
    c = sched.submit([13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])  # 3 blocks
    run_to_completion(sched)
    # A and B finished long ago: eviction alone must make room for C
    assert sched.stats["evictions"] > 0
    assert sched.stats["preemptions"] == 0
    assert all(r.status == "finished" for r in sched.finished)


def test_fcfs_priority_never_preempts_better_request():
    sched = make_scheduler(num_blocks=6, max_new_tokens=8)
    a = sched.submit([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # 5 blocks at peak
    b = sched.submit([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # shares the prefix
    run_to_completion(sched)
    # A arrived first: it must never be preempted to serve B
    assert a.num_preemptions == 0
    assert b.num_preemptions >= 1
    assert all(r.status == "finished" for r in sched.finished)


def test_preemption_recompute_is_output_equivalent():
    """Same requests under memory pressure must produce identical outputs."""

    def run(num_blocks):
        sched = make_scheduler(num_blocks=num_blocks, max_new_tokens=8)
        prompts = [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 4],
        ]
        reqs = [sched.submit(p) for p in prompts]
        run_to_completion(sched)
        return [(r.token_ids, r.finish_reason, r.num_preemptions) for r in reqs]

    roomy = run(64)
    tight = run(7)
    assert sum(r[2] for r in tight) > 0  # pressure really preempted something
    assert [r[:2] for r in roomy] == [r[:2] for r in tight]


def test_preempted_request_requeued_in_front():
    sched = make_scheduler(num_blocks=4, max_new_tokens=30)
    a = sched.submit([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    b = sched.submit([13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
    sched.step()  # A prefills
    sched.step()  # B admitted, prefills
    while not sched.is_done():
        batch = sched.step()
        if not batch:
            break
        sched.finish_step(batch, [token_fn(s.req) for s in batch])
    assert sched.stats["preemptions"] >= 1
    assert all(r.status == "finished" for r in sched.finished)


# ---- request lifecycle ----------------------------------------------------


def test_waiting_queue_fcfs_order():
    sched = make_scheduler(num_blocks=64)
    a = sched.submit([1, 2])
    b = sched.submit([3, 4])
    c = sched.submit([5, 6])
    sched.step()
    assert sched.running == [a, b]  # c waits (max_running=2)
    assert list(sched.waiting) == [c]


def test_submit_rejects_empty_prompt():
    sched = make_scheduler()
    with pytest.raises(AssertionError):
        sched.submit([])


def test_radix_page_size_must_match_block_size():
    pool = KVBlockPool(8, BS, 1, 1, 8)
    cache = RadixCache(page_size=2)
    with pytest.raises(AssertionError):
        Scheduler(pool, cache, max_running=2)
