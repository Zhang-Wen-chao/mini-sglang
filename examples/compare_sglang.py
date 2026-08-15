"""Compare mini-sglang outputs against the official SGLang engine (or HF as fallback).

Same model, same prompts, greedy sampling. Verifies that block-granular KV
reuse does not change model behavior.

Official SGLang (needs `pip install sglang`; set HF_ENDPOINT to a reachable
mirror if huggingface.co is blocked):

    python examples/compare_sglang.py --model JackFram/llama-68m \
        --device cuda --backend sglang

HF fallback (uses transformers' native greedy generate as the reference):

    python examples/compare_sglang.py --model JackFram/llama-68m \
        --device cuda --backend hf

Reports exact-match rate per token for every prompt.
"""

import argparse
import os
import sys

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


def run_mini(model, tok, device, prompts, max_new_tokens, dtype):
    model = model.to(device).eval()
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    pool = KVBlockPool(
        4096, 64, model.config.num_hidden_layers,
        model.config.num_key_value_heads, head_dim,
        dtype=dtype, device=device,
    )
    cache = RadixCache(page_size=64)
    sched = Scheduler(pool, cache, max_running=8,
                      max_new_tokens=max_new_tokens, eos_id=tok.eos_token_id)
    eng = Engine(model, tok, pool, sched)
    return [eng.generate(p) for p in prompts]


def run_sglang(model_id, prompts, max_new_tokens, dtype):
    import sglang

    engine = sglang.Engine(
        model_path=model_id,
        dtype=dtype,
        max_total_tokens=16384,
        mem_fraction_static=0.6,
        attention_backend="triton",  # flashinfer has no fp32 prefill
    )
    params = {"temperature": 0, "max_new_tokens": max_new_tokens}
    outs = engine.generate(prompts, sampling_params=params)
    engine.shutdown()
    return [o["text"] if isinstance(o, dict) else o.text for o in outs]


def run_hf(model_id, prompts, max_new_tokens, device, dtype):
    from transformers import AutoTokenizer, LlamaForCausalLM

    hf = LlamaForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
    tok = AutoTokenizer.from_pretrained(model_id)
    outs = []
    with torch.no_grad():
        for p in prompts:
            ids = tok(p, return_tensors="pt").input_ids.to(device)
            out = hf.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                              pad_token_id=tok.eos_token_id)
            outs.append(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="JackFram/llama-68m")
    ap.add_argument("--backend", choices=["sglang", "hf"], default="sglang")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)

    tok, mini_model = build_hf(args.model, dtype, device)
    mini_outs = run_mini(mini_model, tok, device, PROMPTS, args.max_new_tokens, dtype)

    if args.backend == "sglang":
        ref_outs = run_sglang(args.model, PROMPTS, args.max_new_tokens, args.dtype)
    else:
        ref_outs = run_hf(args.model, PROMPTS, args.max_new_tokens, device, dtype)

    all_exact = True
    all_norm = True
    for p, mine, ref in zip(PROMPTS, mini_outs, ref_outs):
        exact = mine == ref
        norm = mine.strip() == ref.strip()  # tolerate bf16 first-token space flip
        all_exact &= exact
        all_norm &= norm
        print(f"prompt: {p[:40]}")
        print(f"  mini-sglang : {mine[:70]!r}")
        print(f"  {args.backend:<12}: {ref[:70]!r}")
        print(f"  exact match : {exact} | normalized match: {norm}")
    print("ALL PROMPTS EXACT MATCH:", all_exact)
    print("ALL PROMPTS NORMALIZED MATCH:", all_norm)


if __name__ == "__main__":
    main()
