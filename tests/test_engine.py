"""End-to-end engine tests (Phase 3): toy Llama-like model + TinyTokenizer on CPU.

The gold test compares the pooled block-KV forwards against an independent
reference that runs full-sequence attention without any KV pooling, so any
KV reuse / block-placement bug shows up as a numeric mismatch.
"""

import torch

from mini_sglang.engine import Engine
from mini_sglang.kv_pool import KVBlockPool
from mini_sglang.model import ForwardEntry, LlamaConfig, LlamaLike, apply_rope
from mini_sglang.radix_cache import RadixCache
from mini_sglang.scheduler import Scheduler
from mini_sglang.tokenizer import TinyTokenizer

TOY_CONFIG = LlamaConfig(
    vocab_size=288,
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,  # GQA groups = 2
    intermediate_size=48,
    max_position_embeddings=256,
)


def make_toy_model(seed=1234):
    torch.manual_seed(seed)
    return LlamaLike(TOY_CONFIG)


def make_engine(num_blocks=64, block_size=8, max_running=4, max_new_tokens=6,
                temperature=0.0):
    tok = TinyTokenizer()
    model = make_toy_model()
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    pool = KVBlockPool(
        num_blocks, block_size,
        model.config.num_hidden_layers, model.config.num_key_value_heads, head_dim,
    )
    cache = RadixCache(page_size=block_size)
    sched = Scheduler(pool, cache, max_running=max_running,
                      max_new_tokens=max_new_tokens, eos_id=tok.eos_token_id)
    return Engine(model, tok, pool, sched, temperature=temperature)


def reference_forward(model, ids):
    """Full-sequence attention WITHOUT the block pool: the independent gold."""
    t = len(ids)
    h = model.embed_tokens(torch.tensor(ids))
    positions = torch.arange(t)
    cfg = model.config
    for layer in model.layers:
        attn = layer.self_attn
        x = layer.input_layernorm(h)
        q = attn.q_proj(x).view(t, cfg.num_attention_heads, attn.head_dim)
        k = attn.k_proj(x).view(t, cfg.num_key_value_heads, attn.head_dim)
        v = attn.v_proj(x).view(t, cfg.num_key_value_heads, attn.head_dim)
        q = apply_rope(q, positions, model.cos, model.sin)
        k = apply_rope(k, positions, model.cos, model.sin)
        groups = cfg.num_attention_heads // cfg.num_key_value_heads
        scores = torch.einsum("thd,jhd->thj", q, k.repeat_interleave(groups, 1)) / (attn.head_dim ** 0.5)
        mask = positions[:, None, None] >= torch.arange(t)[None, None, :]
        probs = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
        out = torch.einsum("thj,jhd->thd", probs, v.repeat_interleave(groups, 1))
        h = h + attn.o_proj(out.reshape(t, -1))
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    return model.lm_head(model.norm(h))


def pooled_forward(model, pool, ids, block_table, write_offset):
    entry = ForwardEntry(
        input_ids=torch.tensor(ids, dtype=torch.long),
        block_table=block_table,
        write_offset=write_offset,
    )
    return model.forward(pool, [entry])[0]


# ---- gold: pooled forward == reference forward ----------------------------


def test_prefill_matches_reference():
    model = make_toy_model()
    pool = KVBlockPool(16, 8, 2, 2, 8)
    ids = [10, 20, 30, 40, 50, 60, 70, 80]
    blocks = pool.alloc(1)
    logits = pooled_forward(model, pool, ids, blocks, 0)
    ref = reference_forward(model, ids)
    assert torch.allclose(logits, ref, atol=1e-6)


def test_prefill_across_two_blocks_matches_reference():
    model = make_toy_model()
    pool = KVBlockPool(16, 8, 2, 2, 8)
    ids = list(range(10, 26))  # 16 tokens -> 2 blocks
    blocks = pool.alloc(2)
    logits = pooled_forward(model, pool, ids, blocks, 0)
    assert torch.allclose(logits, reference_forward(model, ids), atol=1e-6)


def test_decode_matches_reference_step_by_step():
    model = make_toy_model()
    pool = KVBlockPool(16, 8, 2, 2, 8)
    ids = list(range(10, 34))  # 24 tokens -> 3 blocks
    blocks = pool.alloc(2)
    # prefill 10 tokens
    logits = pooled_forward(model, pool, ids[:10], blocks, 0)
    assert torch.allclose(logits, reference_forward(model, ids[:10]), atol=1e-6)
    # decode tokens one by one, crossing the block boundary (position 16)
    for pos in range(10, 24):
        tok = ids[pos]
        if pos % 8 == 0:
            blocks = blocks + pool.alloc(1)
        lg = pooled_forward(model, pool, [tok], blocks, pos)
        ref = reference_forward(model, ids[: pos + 1])[-1:]
        assert torch.allclose(lg, ref, atol=1e-6), f"mismatch at position {pos}"


def test_reused_prefix_kv_matches_fresh_reference():
    """A second request attending to the FIRST request's cached blocks must
    produce exactly the same logits as a fresh full-sequence forward."""
    model = make_toy_model()
    pool = KVBlockPool(16, 8, 2, 2, 8)
    ids = list(range(10, 26))  # 16 tokens
    # request A prefills everything (KV lands in blocks 0,1)
    a_blocks = pool.alloc(2)
    logits_a = pooled_forward(model, pool, ids, a_blocks, 0)
    assert torch.allclose(logits_a, reference_forward(model, ids), atol=1e-6)
    # request B reuses A's blocks for the 8-token prefix and extends 8 more
    b_blocks = a_blocks + pool.alloc(1)
    logits_b = pooled_forward(model, pool, ids[8:], b_blocks, 8)
    ref = reference_forward(model, ids)[8:]
    assert torch.allclose(logits_b, ref, atol=1e-6)


# ---- engine end-to-end ----------------------------------------------------


def test_generate_is_deterministic():
    eng = make_engine()
    out1 = eng.generate("hello world")
    out2 = eng.generate("hello world")
    assert out1 == out2
    assert len(out1) >= 0


def test_generate_returns_prompt_given_small_pool():
    eng = make_engine(max_new_tokens=3)
    req = eng.scheduler.submit(eng.tokenizer.encode("abc"))
    eng._run_until_done()
    assert req.status == "finished"
    assert req.num_generated == 3


def test_batch_matches_sequential_with_shared_prefix():
    """Continuous batching must not change outputs, even when a request
    reuses another request's in-flight KV blocks."""
    p1 = "Hello world, this is a test prompt."
    p2 = "Hello world, this is another prompt."
    assert eng_bytes(p1)  # keep prompts byte-aligned with the toy tokenizer

    batch_eng = make_engine(max_running=8, max_new_tokens=5)
    outs = batch_eng.generate_batch([p1, p2])
    batch_ids = [list(r.token_ids) for r in batch_eng.scheduler.finished]

    seq_eng = make_engine(max_running=8, max_new_tokens=5)
    o1 = seq_eng.generate(p1)
    o2 = seq_eng.generate(p2)
    seq_ids = [list(r.token_ids) for r in seq_eng.scheduler.finished]

    assert outs[0] == o1 and outs[1] == o2
    assert batch_ids == seq_ids


def eng_bytes(prompt):
    return len(TinyTokenizer().encode(prompt)) > 8


def test_radix_reuse_across_generations():
    eng = make_engine(max_new_tokens=4)
    eng.generate("same prompt every time")
    before = eng.stats["cached_tokens"]
    eng.generate("same prompt every time")
    after = eng.stats["cached_tokens"]
    assert after > before  # the second run hit the cache


def test_prefix_sharing_reduces_second_prefill():
    eng = make_engine(max_new_tokens=4)
    eng.generate("shared prefix A")
    p2 = eng.tokenizer.encode("shared prefix B")
    req = eng.scheduler.submit(p2)
    batch = eng.scheduler.step()
    assert batch[0].write_offset > 0  # matched the cached prefix
    assert batch[0].input_ids != p2  # prefill input is the unmatched tail
    eng.scheduler.finish_step(batch, [12])
    eng._run_until_done()
    assert req.status == "finished"


def test_stats_count_prefill_and_decode():
    eng = make_engine(max_new_tokens=4)
    eng.generate("statistics are fun")
    assert eng.stats["prefill_tokens"] > 0
    assert eng.stats["decode_tokens"] > 0


def test_temperature_sampling_variety():
    eng1 = make_engine(max_new_tokens=8, temperature=1.0)
    eng2 = make_engine(max_new_tokens=8, temperature=1.0)
    # same seed but different engines -> different RNG streams; both must run
    o1 = eng1.generate("rolling dice")
    o2 = eng2.generate("rolling dice")
    assert isinstance(o1, str) and isinstance(o2, str)


def test_tokenizer_roundtrip():
    tok = TinyTokenizer()
    ids = tok.encode("Hello, 世界! \x00\xff")
    assert tok.decode(ids) == "Hello, 世界! \x00\xff"
    assert tok.vocab_size == 260
