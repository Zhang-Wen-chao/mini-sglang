"""Llama-style decoder-only model with block-granular KV cache I/O.

Module and parameter names mirror HF `LlamaForCausalLM` exactly, so real
pretrained weights load with `load_state_dict`. The difference: attention
reads and writes KV through the block pool (`block_table` + logical write
offset) instead of HF's `past_key_values`.

RoPE, RMSNorm, GQA, SiLU MLP: standard Llama. No custom kernels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kv_pool import KVBlockPool


@dataclass
class LlamaConfig:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    max_position_embeddings: int
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False
    bos_token_id: int = 1
    eos_token_id: int = 2


@dataclass
class ForwardEntry:
    """One request's forward work: `input_ids` is the token slice to process."""
    input_ids: torch.Tensor  # (T,) int64
    block_table: List[int]
    write_offset: int  # logical position of input_ids[0]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def apply_rope(x: torch.Tensor, positions: torch.Tensor, cos: torch.Tensor,
               sin: torch.Tensor) -> torch.Tensor:
    """Rotary position embedding; x: (..., D), tables: (max_pos, D // 2)."""
    c = cos[positions].unsqueeze(1)  # (T, 1, D // 2)
    s = sin[positions].unsqueeze(1)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


class LlamaLikeAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, x, positions, cos, sin, kv_pool, layer_idx, block_table, write_offset):
        t = x.shape[0]
        q = self.q_proj(x).view(t, self.config.num_attention_heads, self.head_dim)
        k = self.k_proj(x).view(t, self.config.num_key_value_heads, self.head_dim)
        v = self.v_proj(x).view(t, self.config.num_key_value_heads, self.head_dim)
        q = apply_rope(q, positions, cos, sin)
        k = apply_rope(k, positions, cos, sin)
        past_k, past_v = self._gather_past(kv_pool, layer_idx, block_table, write_offset)
        # write this forward's KV into the pool (after reading past, so a
        # request-private partial block is read-then-written safely)
        self._write_new(kv_pool, layer_idx, k, v, block_table, write_offset)
        k_total = torch.cat([past_k, k], dim=0)
        v_total = torch.cat([past_v, v], dim=0)
        groups = self.config.num_attention_heads // self.config.num_key_value_heads
        k_rep = k_total.repeat_interleave(groups, dim=1)
        v_rep = v_total.repeat_interleave(groups, dim=1)
        scores = torch.einsum("thd,jhd->thj", q, k_rep) / (self.head_dim ** 0.5)
        mask = positions[:, None, None] >= torch.arange(write_offset + t, device=x.device)[None, None, :]
        scores = scores.masked_fill(~mask, float("-inf"))
        probs = F.softmax(scores, dim=-1)
        out = torch.einsum("thj,jhd->thd", probs, v_rep)  # (T, heads, head_dim)
        return self.o_proj(out.reshape(t, -1))

    def _gather_past(self, kv_pool, layer_idx, block_table, write_offset):
        bs = kv_pool.block_size
        n_blocks = (write_offset + bs - 1) // bs
        ks, vs = [], []
        for i in range(n_blocks):
            used = min(bs, write_offset - i * bs)
            k, v = kv_pool.read(layer_idx, block_table[i], used)
            ks.append(k)
            vs.append(v)
        if ks:
            return torch.cat(ks, dim=0), torch.cat(vs, dim=0)
        empty = torch.empty(0, self.config.num_key_value_heads, self.head_dim, device=kv_pool.device)
        return empty, empty

    def _write_new(self, kv_pool, layer_idx, k, v, block_table, write_offset):
        bs = kv_pool.block_size
        for i in range(k.shape[0]):
            pos = write_offset + i
            kv_pool.write(layer_idx, block_table[pos // bs], pos % bs, k[i : i + 1], v[i : i + 1])


class LlamaLikeMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaLikeDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.self_attn = LlamaLikeAttention(config)
        self.mlp = LlamaLikeMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, x, positions, cos, sin, kv_pool, layer_idx, block_table, write_offset):
        h = self.self_attn(
            self.input_layernorm(x), positions, cos, sin,
            kv_pool, layer_idx, block_table, write_offset,
        )
        x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class LlamaLike(nn.Module):
    """Structural clone of LlamaForCausalLM with pool-based block KV cache."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [LlamaLikeDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        head_dim = config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (
            config.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        t = torch.arange(config.max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # (max_pos, head_dim // 2)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, kv_pool: KVBlockPool, entries: List[ForwardEntry]) -> List[torch.Tensor]:
        """Run all entries; returns per-entry logits (T, vocab) in input order."""
        return [self._forward_one(kv_pool, e) for e in entries]

    def _forward_one(self, kv_pool: KVBlockPool, e: ForwardEntry) -> torch.Tensor:
        t = e.input_ids.shape[0]
        h = self.embed_tokens(e.input_ids)
        positions = torch.arange(
            e.write_offset, e.write_offset + t, device=e.input_ids.device, dtype=torch.long
        )
        for i, layer in enumerate(self.layers):
            h = layer(h, positions, self.cos, self.sin, kv_pool, i, e.block_table, e.write_offset)
        return self.lm_head(self.norm(h))
