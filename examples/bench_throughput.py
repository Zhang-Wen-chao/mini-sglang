"""Rough throughput comparison: mini-sglang vs official SGLang 0.5.17.

Environment-bound record only (L20, llama-68m, bf16, triton attention for
sglang). Not a benchmark claim; see README for the correctness comparison.
"""

import os
import sys
import time

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mini_sglang.engine import Engine
from mini_sglang.kv_pool import KVBlockPool
from mini_sglang.radix_cache import RadixCache
from mini_sglang.scheduler import Scheduler
from examples.chat import build_hf

PROMPTS = [
    "The capital of France is",
    "The quick brown fox jumps over",
    "Explain why the sky is blue in one sentence.",
    "Write a haiku about a mountain lake.",
]
MAX_NEW = 32
BATCHES = [4, 16]


def bench_mini(prompts, max_new):
    dtype, device = torch.bfloat16, torch.device("cuda")
    tok, model = build_hf("JackFram/llama-68m", dtype, device)
    model = model.to(device).eval()
    hd = model.config.hidden_size // model.config.num_attention_heads
    pool = KVBlockPool(8192, 64, model.config.num_hidden_layers,
                       model.config.num_key_value_heads, hd,
                       dtype=dtype, device=device)
    sched = Scheduler(pool, RadixCache(page_size=64), max_running=32,
                      max_new_tokens=max_new, eos_id=tok.eos_token_id)
    eng = Engine(model, tok, pool, sched)
    eng.generate("warmup prompt")
    results = {}
    for b in BATCHES:
        t0 = time.time()
        eng.generate_batch(prompts * b)
        t1 = time.time()
        n = len(prompts) * b * max_new
        results[b] = n / (t1 - t0)
    return results


def bench_sglang(prompts, max_new):
    import sglang

    engine = sglang.Engine(
        model_path="JackFram/llama-68m", dtype="bfloat16",
        max_total_tokens=16384, mem_fraction_static=0.6,
        attention_backend="triton",
    )
    params = {"temperature": 0, "max_new_tokens": max_new}
    engine.generate(["warmup"], sampling_params=params)
    results = {}
    for b in BATCHES:
        t0 = time.time()
        engine.generate(prompts * b, sampling_params=params)
        t1 = time.time()
        n = len(prompts) * b * max_new
        results[b] = n / (t1 - t0)
    engine.shutdown()
    return results


def main():
    print("mini-sglang...", flush=True)
    mini = bench_mini(PROMPTS, MAX_NEW)
    print("sglang 0.5.17...", flush=True)
    ref = bench_sglang(PROMPTS, MAX_NEW)
    print(f"batch | mini-sglang | sglang 0.5.17 | ratio")
    for b in BATCHES:
        print(f"{b:5d} | {mini[b]:9.1f} tok/s | {ref[b]:9.1f} tok/s | {ref[b]/mini[b]:.2f}x")


if __name__ == "__main__":
    main()
