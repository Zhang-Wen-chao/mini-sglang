"""Unit tests for the KV block pool (Phase 2)."""

import pytest
import torch

from mini_sglang.kv_pool import KVBlockPool


def make_pool(num_blocks=8, block_size=4):
    return KVBlockPool(num_blocks, block_size, n_layers=2, n_kv_heads=1, head_dim=8)


def test_alloc_and_free():
    pool = make_pool(4)
    assert pool.num_free() == 4
    ids = pool.alloc(3)
    assert len(ids) == 3 and len(set(ids)) == 3
    assert pool.num_free() == 1
    pool.free(ids)
    assert pool.num_free() == 4


def test_alloc_reuses_freed_blocks():
    pool = make_pool(4)
    a = pool.alloc(2)
    assert a == [0, 1]
    pool.free(a)
    # FIFO free list: new allocs take from the front, freed blocks recycle later
    b = pool.alloc(2)
    assert b == [2, 3]
    pool.free(b)
    assert pool.alloc(2) == [0, 1]  # recycled


def test_alloc_exhausted_raises():
    pool = make_pool(2)
    pool.alloc(2)
    with pytest.raises(RuntimeError):
        pool.alloc(1)


def test_double_free_raises():
    pool = make_pool(2)
    ids = pool.alloc(1)
    pool.free(ids)
    with pytest.raises(AssertionError):
        pool.free(ids)


def test_write_read_roundtrip():
    pool = make_pool(4, block_size=4)
    bid = pool.alloc(1)[0]
    k = torch.arange(2 * 1 * 8, dtype=torch.float32).reshape(2, 1, 8)
    v = k + 100
    pool.write(0, bid, offset=1, k=k, v=v)
    rk, rv = pool.read(0, bid, num_tokens=3)
    assert torch.equal(rk[1:3], k)
    assert torch.equal(rv[1:3], v)
    assert torch.equal(rk[0], torch.zeros(1, 8))  # untouched slot
    pool.free(bid)
