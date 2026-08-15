"""Minimal inference engine: tokenize -> prefill -> decode -> response.

Continuous batching loop over the scheduler: each step schedules a mixed
prefill/decode batch, the model runs it (writing KV into the block pool),
tokens are sampled, and `finish_step` advances the scheduler. No async, no
chunked prefill, no speculative decoding.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import torch

from .kv_pool import KVBlockPool
from .model import ForwardEntry, LlamaLike
from .scheduler import Scheduler, ScheduledReq


class Engine:
    def __init__(
        self,
        model: LlamaLike,
        tokenizer,
        kv_pool: KVBlockPool,
        scheduler: Scheduler,
        temperature: float = 0.0,
        seed: int = 0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.kv_pool = kv_pool
        self.scheduler = scheduler
        self.temperature = temperature
        self._rng = random.Random(seed)
        self.device = kv_pool.device

    # ---- public API -----------------------------------------------------

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """Generate a response for one prompt (text in, text out)."""
        return self.generate_batch([prompt], max_new_tokens)[0]

    def generate_batch(
        self, prompts: List[str], max_new_tokens: Optional[int] = None
    ) -> List[str]:
        """Continuously batch a list of prompts and return all responses."""
        if max_new_tokens is not None:
            self.scheduler.max_new_tokens = max_new_tokens
        reqs = [self.scheduler.submit(self.tokenizer.encode(p)) for p in prompts]
        self._run_until_done()
        return [
            self.tokenizer.decode(r.token_ids[len(r.prompt) :]) for r in reqs
        ]

    @property
    def stats(self) -> dict:
        return self.scheduler.stats

    # ---- internals ------------------------------------------------------

    def _run_until_done(self) -> None:
        while not self.scheduler.is_done():
            batch = self.scheduler.step()
            if not batch:
                raise RuntimeError("engine stalled: scheduler made no progress")
            logits = self._forward(batch)
            tokens = [self._sample(lg[-1]) for lg in logits]
            self.scheduler.finish_step(batch, tokens)

    def _forward(self, batch: List[ScheduledReq]) -> List[torch.Tensor]:
        entries = [
            ForwardEntry(
                input_ids=torch.tensor(s.input_ids, device=self.device, dtype=torch.long),
                block_table=s.block_table,
                write_offset=s.write_offset,
            )
            for s in batch
        ]
        with torch.no_grad():
            return self.model.forward(self.kv_pool, entries)

    def _sample(self, logits: torch.Tensor) -> int:
        """Greedy (temperature 0) or temperature-scaled multinomial sampling."""
        if self.temperature == 0.0:
            return int(logits.argmax().item())
        probs = torch.softmax(logits / self.temperature, dim=-1)
        return int(torch.multinomial(probs, 1).item())
