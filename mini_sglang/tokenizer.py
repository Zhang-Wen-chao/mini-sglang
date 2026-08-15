"""Minimal zero-dependency tokenizer for tests and the toy demo.

Byte-level: every UTF-8 byte is a token (ids 4..259), with PAD/BOS/EOS/UNK
reserved. Deterministic and dependency-free; real demos can plug in an HF
tokenizer instead (any object with `encode(text) -> list[int]` and
`decode(ids) -> str` works).
"""

from __future__ import annotations

from typing import List


class TinyTokenizer:
    PAD, BOS, EOS, UNK = 0, 1, 2, 3
    _BYTE_OFFSET = 4  # ids 4..259 map to bytes 0..255

    def __init__(self):
        self.bos_token_id = self.BOS
        self.eos_token_id = self.EOS

    @property
    def vocab_size(self) -> int:
        return 256 + self._BYTE_OFFSET

    def encode(self, text: str) -> List[int]:
        return [self.bos_token_id] + [b + self._BYTE_OFFSET for b in text.encode("utf-8")]

    def decode(self, ids: List[int]) -> str:
        raw = bytes(b - self._BYTE_OFFSET for b in ids if self._BYTE_OFFSET <= b < self._BYTE_OFFSET + 256)
        return raw.decode("utf-8", errors="replace")
