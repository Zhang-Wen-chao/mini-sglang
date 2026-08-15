# mini-sglang

> A teaching reproduction of SGLang's core inference ideas in **pure PyTorch**.

Radix Cache (prefix KV reuse) + continuous batching scheduler + a minimal
inference engine, in ~1,000 lines of readable code. No custom CUDA kernels,
no PagedAttention, no async engine.

This is **not a production framework**. It is a learning artifact, written in
the same style as [mini-megatron](https://github.com/Zhang-Wen-chao/mini-megatron)
and [mini-deepspeed](https://github.com/Zhang-Wen-chao/mini-deepspeed).

## What it reproduces

| SGLang component | This repo | Notes |
| --- | --- | --- |
| `srt/mem_cache/radix_cache.py` | `mini_sglang/radix_cache.py` | radix tree with interval keys, page-aligned prefix match/split, ref counting, LRU eviction |
| `srt/mem_cache/block_allocator.py`, `req_to_token_pool` | `mini_sglang/kv_pool.py` | physical block pool; per-request block tables replace the token pool |
| `srt/managers/scheduler.py` | `mini_sglang/scheduler.py` | FCFS queue, memory budget, evict-then-preempt, recompute preemption, block-completion rules |
| `srt/managers/engine.py` | `mini_sglang/engine.py` | tokenize -> prefill -> decode -> response, continuous batching loop |
| Llama model | `mini_sglang/model.py` | structural clone of HF `LlamaForCausalLM` (same state_dict names), KV via the block pool |

## The core ideas, in one page

**Radix cache.** KV blocks of complete token prefixes are stored in a radix
tree (compressed trie). A new request `match_prefix`es its prompt, and the
longest cached prefix's physical blocks are shared instead of recomputed.
Nodes hold an interval into a shared token array; matching/insertion split
nodes at block-aligned boundaries; `ref_count` protects blocks in use by
running requests; eviction frees LRU unreferenced leaves.

**Continuous batching.** No batch barriers: each scheduler step mixes prefill
(new requests) and decode (running requests) into one batch. Memory pressure
evicts radix cache first, then preempts the lowest-priority running request
(recompute mode: drop its KV, re-queue it in front, re-prefill later). FCFS
priority ensures early requests are never preempted to serve later ones.

**Block-granular KV.** Only COMPLETE blocks enter the radix cache; a request
holds `[shared blocks] + [one private partial block]`. The model attends by
reading blocks from the pool and writes new KV into request-private blocks.
If another request already cached the same tokens, the duplicate blocks are
freed and the request rebinds to the shared ones.

## Quick start

```bash
pip install -e '.[dev]'
pytest -q                      # 52 CPU unit tests, no GPU needed
python examples/chat.py --prompt "hello"    # toy model, zero external deps
```

Real weights on a GPU (needs `transformers`):

```bash
python examples/chat.py --hf-model JackFram/llama-68m --device cuda \
    --dtype float32 --max-new-tokens 24 --prompt "The capital of France is"
```

Verified on 4xL20 (2026-08-15, llama-68m, fp32): identical outputs with and
without radix-prefix reuse; 64-token shared prefixes hit the cache across
consecutive requests (`cached_tokens` grows, prefill shrinks).

## Repository layout

```
mini_sglang/
├── radix_cache.py   # radix tree: match / insert / split / refcount / evict
├── kv_pool.py       # physical block pool + K/V storage
├── scheduler.py     # continuous batching: queues, preemption, block rules
├── model.py         # Llama-like model (HF-compatible state_dict), pool KV
├── tokenizer.py     # zero-dependency byte tokenizer (HF tokenizers plug in)
└── engine.py        # inference loop: schedule -> forward -> sample -> advance
examples/chat.py     # toy demo + real-weight demo
tests/               # 52 CPU unit tests (radix / pool / scheduler / engine)
plan.md              # phase breakdown and status
```

## Known simplifications vs SGLang

- Prefill is not chunked; a request prefills its whole prompt in one step.
- The unaligned tail of a finished request's last block is freed, not padded
  and cached (so sub-block prompts never hit the cache).
- Decode forwards are per-request; SGLang batches them with ragged tensors.
- No async engine, no speculative decoding, no swap-to-CPU preemption.
- Eviction is a plain LRU over leaves (SGLang has policy plugins).
