# mini-sglang

> A teaching implementation of SGLang's core inference ideas in pure PyTorch.

Radix Cache (prefix KV reuse) + continuous batching scheduler + a minimal
inference engine, in well under 1,000 lines of readable code.

**This is not a production framework.** It is a learning artifact, written
following the same style as [mini-megatron](https://github.com/Zhang-Wen-chao/mini-megatron)
and [mini-deepspeed](https://github.com/Zhang-Wen-chao/mini-deepspeed).

## What it reproduces

| SGLang component | This repo |
| --- | --- |
| `srt/mem_cache/radix_cache.py` | `mini_sglang/radix_cache.py` — radix tree with interval keys, prefix split/share, ref counting, LRU eviction |
| `srt/mem_cache/block_allocator.py` | `mini_sglang/kv_pool.py` — physical block pool + per-request block tables |
| `srt/managers/scheduler.py` | `mini_sglang/scheduler.py` — waiting queue, memory budget, prefill/decode scheduling, recompute preemption |
| `srt/managers/engine.py` | `mini_sglang/engine.py` — tokenize → prefill → decode → response loop |

## Quick start

```bash
pip install -e '.[dev]'
pytest -q
python examples/chat.py --prompt "Once upon a time"
```

## Project status

- Phase 1 (radix cache) complete — CPU unit tests pass.
- Phase 2 (scheduler) in progress.
- Phase 3 (engine) planned.

See [plan.md](plan.md) for the phase breakdown.
