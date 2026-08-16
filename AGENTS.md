# mini-sglang

Teaching reproduction of SGLang's core inference ideas in pure PyTorch:
radix cache (prefix KV reuse) + continuous batching scheduler + minimal
inference engine, in ~1,000 lines.

**Independent repo**: <https://github.com/Zhang-Wen-chao/mini-sglang>

## Key conventions

- Pure PyTorch core; runtime deps are `torch` only. HF `transformers` only in
  `examples/chat.py` (optional demo).
- Components map 1:1 to SGLang sources: `radix_cache.py` ->
  `sglang/srt/mem_cache/radix_cache.py`, `scheduler.py` ->
  `sglang/srt/managers/scheduler.py`, etc. Align abstractions, not APIs.
- Only COMPLETE blocks enter the radix cache; page size == block size;
  splits/matches are page-aligned (prevents double-freed blocks).
- Preemption is recompute-only; evict radix before preempting; FCFS priority;
  requests already in the current batch are never preempted.
- Model is a structural clone of HF `LlamaForCausalLM` (same state_dict
  names); KV goes through the block pool.
- Run `pytest -q` locally; real-model verification on a GPU host via
  `examples/chat.py --hf-model JackFram/llama-68m --device cuda`
  (if huggingface.co is blocked, set `HF_ENDPOINT` to a reachable mirror).
- Do not promise features that are not implemented: no chunked prefill, no
  ragged decode batching, no async, no swap, no kernel optimizations.
