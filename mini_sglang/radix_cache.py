"""Radix cache: a radix tree (compressed trie) that reuses KV cache by longest-common-prefix matching.

Faithful to SGLang's `sglang/srt/mem_cache/radix_cache.py`, simplified:

- A node's key is an interval `(start, length)` into a shared, growing token
  array, so a shared prefix is stored once and referenced by many nodes.
- `value` maps key pages to KV cache units: one unit per `page_size` tokens
  (a physical block id in later phases). Splitting a node slices the value.
- Matching and inserting round split positions DOWN to page boundaries, so a
  node never shares a partial page/block with another node (this is SGLang's
  `page_size` behavior and prevents double-freeing KV units).
- `ref_count` is the number of running requests currently referencing the
  node's prefix; only `ref_count == 0` leaves may be evicted.
- Eviction is LRU over evictable leaves by `last_access_time`.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple


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
        self.children: dict[tuple, "_Node"] = {}
        self.start = start
        self.length = length
        self.value = value  # per-page KV unit id, or None for the root
        self.last_access_time = 0
        self.ref_count = 0

    def __lt__(self, other: "_Node") -> bool:
        return (self.last_access_time, self.id) < (other.last_access_time, other.id)

    def __repr__(self) -> str:
        return f"Node({self.start},{self.length}) r={self.ref_count} v={self.value}"


@dataclass
class MatchResult:
    matched_len: int  # page-aligned number of query tokens covered
    blocks: List[int] = field(default_factory=list)  # concatenated per-page values
    node: Optional[_Node] = None  # deepest node fully matched (or split boundary)


class RadixCache:
    """Page-aligned radix tree over a shared token store."""

    def __init__(self, page_size: int = 1):
        assert page_size >= 1
        self.page_size = page_size
        self.tokens: List[int] = []  # shared token array (grows on insert)
        self.root = _Node(parent=None, start=0, length=0, value=None)
        self.root.ref_count = 1  # root is never evictable
        self._access_time = 0
        self._evictable_leaves: set[_Node] = set()

    # ---- node helpers ---------------------------------------------------

    def _node_key(self, node: _Node) -> List[int]:
        return self.tokens[node.start : node.start + node.length]

    def _page_floor(self, n: int) -> int:
        return n // self.page_size * self.page_size

    def _page_key(self, start_or_node, offset: int = 0) -> tuple:
        """Dict key: the first page of a node, of `self.tokens[start:...]`, or of a list."""
        if isinstance(start_or_node, _Node):
            seq, start = self.tokens, start_or_node.start + offset
        elif isinstance(start_or_node, int):
            seq, start = self.tokens, start_or_node + offset
        else:
            seq, start = start_or_node, offset
        return tuple(seq[start : start + self.page_size])

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

        `split_len` must be page-aligned. The new prefix node takes `node`'s
        position in the tree (same parent, ref_count and value prefix); `node`
        keeps the suffix, its children and slides its interval forward.
        """
        new_node = _Node(
            parent=node.parent,
            start=node.start,
            length=split_len,
            value=node.value[: split_len // self.page_size]
            if node.value is not None
            else None,
        )
        new_node.ref_count = node.ref_count
        new_node.last_access_time = node.last_access_time
        node.parent = new_node
        node.start += split_len
        node.length -= split_len
        if node.value is not None:
            node.value = node.value[split_len // self.page_size :]
        node.ref_count = 0
        new_node.children = {self._page_key(node): node}
        if new_node.parent is not None:
            new_node.parent.children[self._page_key(new_node)] = new_node
        return new_node

    def _delete_leaf(self, node: _Node) -> None:
        parent = node.parent
        del parent.children[self._page_key(node)]
        node.parent = None  # marks the node deleted (stale heap entries skip it)
        self._evictable_leaves.discard(node)
        self._update_leaf_status(parent)

    # ---- public API -----------------------------------------------------

    def match_prefix(self, ids: List[int]) -> MatchResult:
        """Return the longest page-aligned cached prefix of `ids` and its KV units."""
        result = MatchResult(matched_len=0)
        node = self.root
        self._touch(node)
        i = 0
        while i + self.page_size <= len(ids) and self._page_key(ids, i) in node.children:
            child = node.children[self._page_key(ids, i)]
            self._touch(child)
            j = 0
            k = child.start
            while (
                j < child.length
                and i + j < len(ids)
                and self.tokens[k + j] == ids[i + j]
            ):
                j += 1
            j = self._page_floor(j)
            if j < child.length:
                # query diverges (or ends) inside this node: expose aligned boundary
                if j > 0:
                    child = self._split_node(child, j)
                node = child
                result.matched_len += j
                if child.value is not None:
                    result.blocks.extend(child.value[: j // self.page_size])
                break
            node = child
            i += child.length
            result.matched_len += child.length
            if child.value is not None:
                result.blocks.extend(child.value)
        result.node = node
        return result

    def insert(
        self, ids: List[int], value: Optional[List[int]] = None
    ) -> Tuple[_Node, int]:
        """Insert `ids` into the tree.

        `value` maps key pages to KV units (`len(value) == len(ids) // page_size`),
        or `None` (tokens cached without KV, e.g. in tests).

        Returns `(deepest_node, new_start)`: the deepest node on the inserted
        path, and the token offset from which the tree did NOT already have
        coverage (pages before `new_start // page_size` were duplicates).
        """
        if not ids:
            return self.root, 0
        start = len(self.tokens)
        self.tokens.extend(ids)
        return self._insert_helper(self.root, start, len(ids), value)

    def _insert_helper(
        self,
        node: _Node,
        key_start: int,
        key_len: int,
        value: Optional[List[int]],
    ) -> Tuple[_Node, int]:
        self._touch(node)
        if key_len == 0:
            return node, 0
        key = self._page_key(key_start)
        if key in node.children:
            child = node.children[key]
            i = 0
            while (
                i < child.length
                and i < key_len
                and self.tokens[child.start + i] == self.tokens[key_start + i]
            ):
                i += 1
            i = self._page_floor(i)
            if i == 0:
                # no full page shared: create a sibling under `node`
                new_node = _Node(parent=node, start=key_start, length=key_len, value=value)
                node.children[key] = new_node
                self._touch(new_node)
                self._update_leaf_status(node)
                self._update_leaf_status(new_node)
                return new_node, 0
            if i == child.length:
                # whole child is a prefix of the new key
                node, new_start = self._insert_helper(
                    child,
                    key_start + child.length,
                    key_len - child.length,
                    value[child.length // self.page_size :] if value is not None else None,
                )
                return node, child.length + new_start
            # diverge inside the child: split at the shared (page-aligned) prefix
            new_node = self._split_node(child, i)
            node, new_start = self._insert_helper(
                new_node,
                key_start + i,
                key_len - i,
                value[i // self.page_size :] if value is not None else None,
            )
            return node, i + new_start
        new_node = _Node(parent=node, start=key_start, length=key_len, value=value)
        node.children[key] = new_node
        self._touch(new_node)
        self._update_leaf_status(node)
        self._update_leaf_status(new_node)
        return new_node, 0

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
    tree = RadixCache(page_size=2)
    tree.insert([1, 2, 3, 4, 5, 6], [10, 11, 12])
    tree.insert([1, 2, 7, 8], [20, 21])
    tree.insert([1, 2, 3, 4, 9, 10], [30, 31, 32])
    print(tree.pretty_print())
    r = tree.match_prefix([1, 2, 3, 13, 14])
    print("match:", r.matched_len, "blocks:", r.blocks, "node:", r.node)
