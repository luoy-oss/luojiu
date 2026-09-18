"""Small language-neutral sparse feature helpers.

There is no tokenizer, embedding model, gradient descent, or external service.
The replaceable brain uses these deterministic features for local routing.
"""
from __future__ import annotations

import hashlib
import math
from difflib import SequenceMatcher

DIMENSIONS = 4096


def normalize(text: str) -> str:
    # Unicode aware and language neutral.  Do not bake a language dictionary
    # into the brain: wording is acquired from the group's examples.
    return "".join(ch.lower() for ch in text if ch.isalnum() or ch == "_")[:300]


def features(text: str) -> dict[str, float]:
    value = normalize(text)
    counts: dict[str, float] = {}
    for n, weight in ((1, 0.45), (2, 1.0), (3, 0.65)):
        for i in range(len(value) - n + 1):
            token = value[i:i + n].encode("utf-8")
            key = str(int.from_bytes(hashlib.blake2s(token, digest_size=4).digest(), "little") % DIMENSIONS)
            counts[key] = counts.get(key, 0.0) + weight
    length = math.sqrt(sum(v * v for v in counts.values())) or 1.0
    result = {k: v / length for k, v in counts.items()}
    result["bias"] = 0.25
    return result


def lexical_similarity(left: str, right: str) -> float:
    a, b = normalize(left), normalize(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    x, y = features(a), features(b)
    cosine = sum(v * y.get(k, 0) for k, v in x.items() if k != "bias")
    sequence = SequenceMatcher(None, a, b, autojunk=False).ratio()
    return 0.6 * cosine + 0.4 * sequence
