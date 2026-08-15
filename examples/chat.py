"""End-to-end demo: toy model by default, or real Llama weights on GPU.

Toy mode (no external deps, runs anywhere):

    python examples/chat.py --prompt "hello"

Real weights (L20, needs transformers + a Llama-arch model):

    python examples/chat.py --hf-model JackFram/llama-68m --device cuda \
        --dtype bfloat16 --prompt "Once upon a time"

Any tokenizer with `encode(text)->list[int]` / `decode(ids)->str` works.
"""

import argparse
import json
import random

import torch

from mini_sglang.engine import Engine
from mini_sglang.kv_pool import KVBlockPool
from mini_sglang.model import LlamaConfig, LlamaLike
from mini_sglang.radix_cache import RadixCache
from mini_sglang.scheduler import Scheduler
from mini_sglang.tokenizer import TinyTokenizer


def build_toy():
    tok = TinyTokenizer()
    cfg = LlamaConfig(
        vocab_size=tok.vocab_size,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        intermediate_size=256,
        max_position_embeddings=512,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    torch.manual_seed(42)
    return tok, LlamaLike(cfg)


def build_hf(model_id, dtype, device):
    from transformers import LlamaForCausalLM, LlamaTokenizerFast

    hf = LlamaForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    tok = LlamaTokenizerFast.from_pretrained(model_id)
    c = hf.config
    cfg = LlamaConfig(
        vocab_size=c.vocab_size,
        hidden_size=c.hidden_size,
        num_hidden_layers=c.num_hidden_layers,
        num_attention_heads=c.num_attention_heads,
        num_key_value_heads=c.num_key_value_heads,
        intermediate_size=c.intermediate_size,
        max_position_embeddings=c.max_position_embeddings,
        rms_norm_eps=c.rms_norm_eps,
        rope_theta=c.rope_theta,
        tie_word_embeddings=c.tie_word_embeddings,
        bos_token_id=c.bos_token_id or 1,
        eos_token_id=c.eos_token_id or 2,
    )
    model = LlamaLike(cfg)
    model.load_state_dict(hf.state_dict())
    del hf
    return tok, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--hf-model", default=None, help="HF Llama model id to load")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--num-blocks", type=int, default=1024)
    ap.add_argument("--max-running", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    tok, model = build_hf(args.hf_model, dtype, device) if args.hf_model else build_toy()
    model.to(device=device, dtype=dtype)
    model.eval()
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    pool = KVBlockPool(
        args.num_blocks, args.block_size,
        model.config.num_hidden_layers, model.config.num_key_value_heads,
        head_dim, dtype=dtype, device=device,
    )
    cache = RadixCache(page_size=args.block_size)
    sched = Scheduler(
        pool, cache, max_running=args.max_running,
        max_new_tokens=args.max_new_tokens, eos_id=tok.eos_token_id,
    )
    engine = Engine(model, tok, pool, sched,
                    temperature=args.temperature, seed=args.seed)

    print(f"model: {args.hf_model or 'toy'} | device: {device} | "
          f"KV pool: {args.num_blocks} blocks x {args.block_size} tokens")
    while True:
        prompt = args.prompt
        text = engine.generate(prompt)
        print(f"\nprompt : {prompt}")
        print(f"output : {text}")
        print(f"stats  : {engine.stats}")
        try:
            args.prompt = input("prompt> ").strip()
        except EOFError:
            break
        if not args.prompt:
            break


if __name__ == "__main__":
    main()
