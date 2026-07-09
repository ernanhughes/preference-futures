"""Dependency-free text utilities used by the NewsEdits adapter."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any


def normalise_space(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalise_sentence(value: str) -> str:
    text = normalise_space(value).lower()
    text = re.sub(r"[“”]", '"', text)
    return re.sub(r"[‘’]", "'", text)


def tokenise(value: str) -> list[str]:
    return re.findall(r"\b[\w'-]+\b", value.lower())


def lexical_jaccard(left: str, right: str) -> float:
    left_tokens = set(tokenise(left))
    right_tokens = set(tokenise(right))
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def sentence_split(text: str) -> list[str]:
    """Split article text while preserving paragraph boundaries before normalisation."""

    if not text or not text.strip():
        return []
    cleaned = re.sub(r"\r\n?", "\n", text)
    cleaned = re.sub(r"[\t\f\v ]+", " ", cleaned)
    pieces = re.split(
        r"(?<=[.!?])\s+(?=(?:[\"'“‘(\[]?[A-Z0-9]))|(?:\s*\n+\s*)",
        cleaned,
    )
    return [sentence for piece in pieces if (sentence := normalise_space(piece))]


def valid_sentence(sentence: str, *, min_chars: int, max_chars: int) -> bool:
    if not min_chars <= len(sentence) <= max_chars:
        return False
    return len(tokenise(sentence)) >= 3


def surrounding_context(
    sentences: Sequence[str],
    index: int,
    *,
    before: int,
    after: int,
) -> tuple[str, str]:
    before_start = max(0, index - before)
    after_end = min(len(sentences), index + 1 + after)
    return (
        " ".join(sentences[before_start:index]),
        " ".join(sentences[index + 1 : after_end]),
    )
