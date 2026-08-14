"""Unit tests for the radix cache (Phase 1).

Pure data-structure tests: no torch, no model. `value` is a per-token list of
KV units (physical block ids in later phases); here tests use opaque ints.
"""

import pytest

from mini_sglang.radix_cache import RadixCache


def val(n):
    """Per-token value helper: token i -> KV unit i."""
    return list(range(n))


# ---- match --------------------------------------------------------------


def test_match_full_prefix():
    tree = RadixCache()
    tree.insert([1, 2, 3, 4], val(4))
    r = tree.match_prefix([1, 2, 3, 4])
    assert r.matched_len == 4
    assert r.blocks == [0, 1, 2, 3]
    assert r.node.length == 4


def test_match_partial_prefix():
    tree = RadixCache()
    tree.insert([1, 2, 3, 4], val(4))
    r = tree.match_prefix([1, 2, 3, 9, 9])
    assert r.matched_len == 3
    assert r.blocks == [0, 1, 2]


def test_match_no_prefix():
    tree = RadixCache()
    tree.insert([1, 2, 3], val(3))
    r = tree.match_prefix([9, 9, 9])
    assert r.matched_len == 0
    assert r.blocks == []
    assert r.node is tree.root


def test_match_ending_inside_node_splits_tree():
    tree = RadixCache()
    tree.insert([1, 2, 3, 4, 5], val(5))
    r = tree.match_prefix([1, 2, 3])
    assert r.matched_len == 3
    assert r.blocks == [0, 1, 2]
    # the tree is now split: query boundary is a real node
    assert r.node.length == 3
    assert r.node.children  # suffix [4, 5] still reachable
    r2 = tree.match_prefix([1, 2, 3, 4, 5])
    assert r2.matched_len == 5


def test_match_shares_values_after_split():
    tree = RadixCache()
    tree.insert([1, 2, 3, 4], val(4))
    # match ends inside the node; split must keep the prefix's values
    r = tree.match_prefix([1, 2])
    assert r.blocks == [0, 1]


# ---- insert / split -----------------------------------------------------


def test_insert_shares_prefix_between_two_sequences():
    tree = RadixCache()
    tree.insert([1, 2, 3], val(3))
    tree.insert([1, 2, 4, 5], [10, 11, 12, 13])
    # one shared node [1,2] with two leaf children
    root_children = list(tree.root.children.values())
    assert len(root_children) == 1
    shared = root_children[0]
    assert shared.length == 2
    assert shared.value == [0, 1]
    assert len(shared.children) == 2
    assert tree.total_len() == 5  # deduplicated: 2 shared + 1 + 2


def test_insert_suffix_after_shared_prefix():
    tree = RadixCache()
    tree.insert([1, 2, 3], val(3))
    tree.insert([1, 2, 3, 4, 5], val(5))
    r = tree.match_prefix([1, 2, 3, 4])
    assert r.matched_len == 4
    assert r.blocks == [0, 1, 2, 3]


def test_insert_duplicate_key_is_idempotent():
    tree = RadixCache()
    tree.insert([1, 2, 3], val(3))
    tree.insert([1, 2, 3], val(3))
    assert tree.total_len() == 3
    assert tree.sum_ref_count() == 0


def test_insert_shorter_key_splits_longer_node():
    tree = RadixCache()
    tree.insert([1, 2, 3, 4, 5], val(5))
    tree.insert([1, 2, 3], val(3))
    r = tree.match_prefix([1, 2, 3])
    assert r.matched_len == 3
    assert r.node.length == 3
    # the longer key is still fully matchable
    assert tree.match_prefix([1, 2, 3, 4, 5]).matched_len == 5


def test_insert_without_value():
    tree = RadixCache()
    tree.insert([1, 2, 3])
    r = tree.match_prefix([1, 2, 3])
    assert r.matched_len == 3
    assert r.blocks == []  # no KV units attached


# ---- ref counting -------------------------------------------------------


def test_inc_ref_blocks_eviction():
    tree = RadixCache()
    tree.insert([1, 2, 3], val(3))
    node = tree.match_prefix([1, 2, 3]).node
    tree.inc_ref_count(node)
    assert tree.sum_ref_count() == 1  # one compressed node on the path
    assert tree.evict(100) == 0  # referenced: nothing evictable
    tree.dec_ref_count(node)
    assert tree.sum_ref_count() == 0
    assert tree.evict(100) == 3  # now evictable


def test_inc_ref_path_shared_by_two_requests():
    tree = RadixCache()
    tree.insert([1, 2, 3], val(3))
    tree.insert([1, 2, 4], [10, 11, 12])
    shared = tree.match_prefix([1, 2]).node
    a = tree.match_prefix([1, 2, 3]).node
    b = tree.match_prefix([1, 2, 4]).node
    assert shared.length == 2
    tree.inc_ref_count(a)
    tree.inc_ref_count(b)
    assert shared.ref_count == 2  # both requests hold the shared prefix
    assert tree.evict(100) == 0
    tree.dec_ref_count(a)
    # leaf [3] now evictable but the shared node [1,2] still referenced by b
    assert tree.evict(100) == 1
    tree.dec_ref_count(b)
    assert tree.evict(100) == 3  # [4] and then the now-unreferenced [1,2]


# ---- eviction -----------------------------------------------------------


def test_evict_lru_order():
    tree = RadixCache()
    tree.insert([1, 2, 3], val(3))  # t=1
    tree.insert([4, 5, 6], [7, 8, 9])  # t=2
    tree.match_prefix([1, 2, 3])  # refresh [1,2,3] to newest (t=3)
    # eviction takes whole leaf nodes; [4,5,6] is now the LRU leaf
    tree.evict(1)
    assert tree.total_len() == 3
    assert tree.match_prefix([1, 2, 3]).matched_len == 3
    assert tree.match_prefix([4, 5, 6]).matched_len == 0


def test_evict_children_then_parent():
    tree = RadixCache()
    tree.insert([1, 2, 3], val(3))
    tree.insert([1, 2, 4], [10, 11, 12])
    tree.evict(4)
    assert tree.total_len() == 0  # [3], then shared [1,2], then [4]: all evictable


def test_evict_callback_receives_kv_units():
    tree = RadixCache()
    tree.insert([1, 2, 3], val(3))  # older
    tree.insert([4, 5], [7, 8])
    freed = []

    tree.evict(3, evict_callback=lambda units: freed.append(units))
    assert freed == [[0, 1, 2]]  # LRU leaf evicted first


def test_evict_returns_evicted_token_count():
    tree = RadixCache()
    tree.insert([1, 2, 3], val(3))
    tree.insert([4, 5], [7, 8])
    # eviction is whole-node: first call takes the entire [1,2,3] leaf
    assert tree.evict(1) == 3
    assert tree.evict(1) == 2
    assert tree.evict(1) == 0
    assert tree.total_len() == 0


def test_evict_zero_or_negative_is_noop():
    tree = RadixCache()
    tree.insert([1, 2, 3], val(3))
    assert tree.evict(0) == 0
    assert tree.evict(-1) == 0
    assert tree.total_len() == 3


# ---- misc ---------------------------------------------------------------


def test_pretty_print_roundtrip():
    tree = RadixCache()
    tree.insert([1, 2, 3], val(3))
    out = tree.pretty_print()
    assert "#tokens: 3" in out
    assert "r=0" in out


def test_interleaved_lifecycle():
    """A realistic cycle: insert -> match -> ref -> insert diverge -> release."""
    tree = RadixCache()
    tree.insert([1, 2, 3, 4], val(4))
    r = tree.match_prefix([1, 2, 3, 4])
    tree.inc_ref_count(r.node)
    tree.insert([1, 2, 3, 5], [20, 21, 22, 23])
    assert tree.match_prefix([1, 2, 3, 5]).matched_len == 4
    tree.dec_ref_count(r.node)
    # whole-node eviction overshoots the 4-token request: all 5 tokens go
    assert tree.evict(4) == 5
    assert tree.evict(4) == 0
    assert tree.total_len() == 0
