"""Radix cache: a radix tree (compressed trie) that reuses KV cache by longest-common-prefix matching.

Faithful to SGLang's `sglang/srt/mem_cache/radix_cache.py`, simplified:

- A node's key is an interval `(start, length)` into a shared, growing token
  array, so a shared prefix is stored once and referenced by many nodes.
- `value` maps every key token to the KV cache unit (a physical block id in
  Phase 2+) that holds that token's KV. Splitting a node slices the value.
- `ref_count` is the number of running requests currently referencing the
  node's prefix; only `ref_count == 0` leaves may be evicted.
- Eviction is LRU over evictable leaves by `last_access_time`.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Callable, List, Optional


class _Node:
    __slots__ = (
        "id",
        "parent",
        "children",
        "start",
        "length",
        "value",
        "last_access_time",
        "ref_count",
    )

    _counter = 0

    def __init__(
        self,
        parent: Optional["_Node"],
        start: int,
        length: int,
        value: Optional[List[int]],
    ):
        self.id = _Node._counter
        _Node._counter += 1
        self.parent = parent
        self.children: dict[int, "_Node"] = {}
        self.start = start
        self.length = length
        self.value = value  # per-key-token KV unit id, or None for the root
        self.last_access_time = 0
        self.ref_count = 0

    def __lt__(self, other: "_Node") -> bool:
        return (self.last_access_time, self.id) < (other.last_access_time, other.id)

    def __repr__(self) -> str:
        return f"Node({self.start},{self.length}) r={self.ref_count} v={self.value}"


@dataclass
class MatchResult:
    matched_len: int  # number of query tokens covered by the cached prefix
    blocks: List[int] = field(default_factory=list)  # concatenated node values
    node: Optional[_Node] = None  # deepest node fully matched (or split boundary)


class RadixCache:
    """Token-level radix tree over a shared token store."""

    def __init__(self):
        self.tokens: List[int] = []  # shared token array (grows on insert)
        self.root = _Node(parent=None, start=0, length=0, value=None)
        self.root.ref_count = 1  # root is never evictable
        self._access_time = 0
        self._evictable_leaves: set[_Node] = set()

    # ---- node helpers ---------------------------------------------------

    def _node_key(self, node: _Node) -> List[int]:
        return self.tokens[node.start : node.start + node.length]

    def _touch(self, node: _Node) -> None:
        self._access_time += 1
        node.last_access_time = self._access_time

    def _update_leaf_status(self, node: _Node) -> None:
        if node.ref_count > 0 or node.children:
            self._evictable_leaves.discard(node)
        else:
            self._evictable_leaves.add(node)

    def _split_node(self, node: _Node, split_len: int) -> _Node:
        """Split `node` into a prefix node of `split_len` tokens + a suffix node.

        The new prefix node takes `node`'s position in the tree (same parent,
        ref_count and value prefix); `node` keeps the suffix, its children and
        slides its interval forward.
        """
        new_node = _Node(
            parent=node.parent,
            start=node.start,
            length=split_len,
            value=node.value[:split_len] if node.value is not None else None,
        )
        new_node.ref_count = node.ref_count
        new_node.last_access_time = node.last_access_time
        new_node.children = {self.tokens[node.start + split_len]: node}
        node.parent = new_node
        node.start += split_len
        node.length -= split_len
        if node.value is not None:
            node.value = node.value[split_len:]
        node.ref_count = 0
        if new_node.parent is not None:
            new_node.parent.children[self.tokens[new_node.start]] = new_node
        return new_node

    def _delete_leaf(self, node: _Node) -> None:
        parent = node.parent
        del parent.children[self.tokens[node.start]]
        node.parent = None  # marks the node deleted (stale heap entries skip it)
        self._evictable_leaves.discard(node)
        self._update_leaf_status(parent)

    # ---- public API -----------------------------------------------------

    def match_prefix(self, ids: List[int]) -> MatchResult:
        """Return the longest cached prefix of `ids` and its KV units."""
        result = MatchResult(matched_len=0)
        node = self.root
        self._touch(node)
        i = 0
        while i < len(ids) and ids[i] in node.children:
            child = node.children[ids[i]]
            self._touch(child)
            j = 0
            k = child.start
            while (
                j < child.length
                and i + j < len(ids)
                and self.tokens[k + j] == ids[i + j]
            ):
                j += 1
            if j < child.length:
                # query ends (or diverges) inside the child: expose boundary
                child = self._split_node(child, j)
                node = child
                result.matched_len += j
                if child.value is not None:
                    result.blocks.extend(child.value)
                break
            node = child
            i += child.length
            result.matched_len += child.length
            if child.value is not None:
                result.blocks.extend(child.value)
        result.node = node
        return result

    def insert(self, ids: List[int], value: Optional[List[int]] = None) -> None:
        """Insert `ids` into the tree; `value` maps each key token to its KV unit.

        `value` may be `None` (tokens cached without KV, e.g. in tests).
        """
        if not ids:
            return
        start = len(self.tokens)
        self.tokens.extend(ids)
        self._insert_helper(self.root, start, len(ids), value)

    def _insert_helper(
        self,
        node: _Node,
        key_start: int,
        key_len: int,
        value: Optional[List[int]],
    ) -> None:
        self._touch(node)
        if key_len == 0:
            return
        ch = self.tokens[key_start]
        if ch in node.children:
            child = node.children[ch]
            i = 0
            while (
                i < child.length
                and i < key_len
                and self.tokens[child.start + i] == self.tokens[key_start + i]
            ):
                i += 1
            if i == child.length:
                # whole child is a prefix of the new key
                self._insert_helper(
                    child,
                    key_start + child.length,
                    key_len - child.length,
                    value[child.length:] if value is not None else None,
                )
            else:
                new_node = self._split_node(child, i)
                self._insert_helper(
                    new_node,
                    key_start + i,
                    key_len - i,
                    value[i:] if value is not None else None,
                )
        else:
            new_node = _Node(parent=node, start=key_start, length=key_len, value=value)
            node.children[ch] = new_node
            self._touch(new_node)
            self._update_leaf_status(node)
            self._update_leaf_status(new_node)

    def inc_ref_count(self, node: Optional[_Node]) -> None:
        """Mark one more request as referencing the path root..node."""
        cur = node
        while cur is not None and cur.parent is not None:
            cur.ref_count += 1
            self._update_leaf_status(cur)
            cur = cur.parent

    def dec_ref_count(self, node: Optional[_Node]) -> None:
        """Release one request's reference to the path root..node."""
        cur = node
        while cur is not None and cur.parent is not None:
            cur.ref_count -= 1
            self._update_leaf_status(cur)
            cur = cur.parent

    def evict(
        self,
        num_tokens: int,
        evict_callback: Optional[Callable[[List[int]], None]] = None,
    ) -> int:
        """Evict up to `num_tokens` of LRU, unreferenced leaf tokens.

        `evict_callback(units)` is called with the evicted node's KV units so
        the owner can free the underlying KV blocks. Returns tokens evicted.
        """
        if num_tokens <= 0:
            return 0
        heap = [(n.last_access_time, n) for n in self._evictable_leaves]
        heapq.heapify(heap)
        evicted = 0
        while heap and evicted < num_tokens:
            _, leaf = heapq.heappop(heap)
            if leaf.ref_count > 0 or leaf.parent is None:
                continue
            parent = leaf.parent
            if evict_callback is not None and leaf.value is not None:
                evict_callback(leaf.value)
            self._delete_leaf(leaf)
            evicted += leaf.length
            if parent is not self.root and parent.ref_count == 0 and not parent.children:
                heapq.heappush(heap, (parent.last_access_time, parent))
        return evicted

    def total_len(self) -> int:
        """Total number of tokens currently cached in the tree."""
        total = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            for child in node.children.values():
                total += child.length
                stack.append(child)
        return total

    def sum_ref_count(self) -> int:
        """Sum of ref_count over all nodes (excluding root)."""
        total = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            for child in node.children.values():
                total += child.ref_count
                stack.append(child)
        return total

    def pretty_print(self) -> str:
        lines: List[str] = []

        def dfs(node: _Node, indent: int) -> None:
            prefix = self._node_key(node)
            lines.append(
                f"{'  ' * indent}{node.length} {prefix[:10]} r={node.ref_count} v={node.value}"
            )
            for child in node.children.values():
                dfs(child, indent + 1)

        dfs(self.root, 0)
        lines.append(f"#tokens: {self.total_len()}")
        return "\n".join(lines)


if __name__ == "__main__":
    tree = RadixCache()
    tree.insert([1, 2, 3])
    tree.insert([1, 2, 4, 5])
    tree.insert([1, 2, 4, 5, 6, 7])
    tree.insert([8, 9, 10])
    print(tree.pretty_print())
    r = tree.match_prefix([1, 2, 3, 13, 14])
    print("match:", r.matched_len, "node:", r.node)
