"""Physical KV block pool: allocation, free lists, and block-granular K/V storage.

Corresponds to SGLang's block allocator + KV pool (`sglang/srt/mem_cache/`),
simplified: no contiguous-allocation requirement, no PagedAttention kernels.
A block is `block_size` token slots of K/V per layer; requests reference
blocks via an explicit block table.
"""

from __future__ import annotations

from collections import deque
from typing import List, Sequence, Union

import torch


class KVBlockPool:
    """Fixed pool of physical blocks; each block holds `block_size` token slots."""

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ):
        assert num_blocks >= 1 and block_size >= 1
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.n_layers = n_layers
        self.dtype = dtype
        self.device = torch.device(device)
        shape = (num_blocks, block_size, n_kv_heads, head_dim)
        self.k_pools = [torch.zeros(shape, dtype=dtype, device=self.device) for _ in range(n_layers)]
        self.v_pools = [torch.zeros(shape, dtype=dtype, device=self.device) for _ in range(n_layers)]
        self._free: deque[int] = deque(range(num_blocks))
        self._live: set[int] = set()

    # ---- allocation -----------------------------------------------------

    def num_free(self) -> int:
        return len(self._free)

    def num_live(self) -> int:
        return len(self._live)

    def alloc(self, n: int = 1) -> List[int]:
        """Allocate `n` physical blocks (non-contiguous is fine)."""
        if n > self.num_free():
            raise RuntimeError(
                f"KV pool exhausted: need {n} blocks, only {self.num_free()} free"
            )
        ids = [self._free.popleft() for _ in range(n)]
        self._live.update(ids)
        return ids

    def free(self, block_ids: Union[int, Sequence[int]]) -> None:
        """Return blocks to the pool. Double-free raises."""
        if isinstance(block_ids, int):
            block_ids = [block_ids]
        for bid in block_ids:
            if bid not in self._live:
                raise AssertionError(f"double free of block {bid}")
            self._live.discard(bid)
            self._free.append(bid)

    # ---- K/V access (used by the model in Phase 3) -----------------------

    def write(self, layer: int, block_id: int, offset: int, k: torch.Tensor,
              v: torch.Tensor) -> None:
        """Write `k, v` of shape (T, n_kv_heads, head_dim) at `offset` in a block."""
        t = k.shape[0]
        self.k_pools[layer][block_id, offset : offset + t] = k
        self.v_pools[layer][block_id, offset : offset + t] = v

    def read(self, layer: int, block_id: int, num_tokens: int):
        """Read the first `num_tokens` slots of a block as (k, v)."""
        return (
            self.k_pools[layer][block_id, :num_tokens],
            self.v_pools[layer][block_id, :num_tokens],
        )
