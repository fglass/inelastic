from typing import Iterator

STOP_WORDS = {"a", "and", "be", "have", "i", "in", "of", "that", "the", "to"}


def analyze(text: str) -> list[str]:
    tokens = _tokenize(text)
    tokens = _filter(tokens)
    return list(tokens)


def _tokenize(text: str) -> Iterator[str]:
    return (t for t in text.split())


def _filter(tokens: Iterator[str]) -> Iterator[str]:
    tokens = _normalization_filter(tokens)
    tokens = _stemmer_filter(tokens)
    tokens = _stop_word_filter(tokens)
    return tokens


#### Token filters


def _normalization_filter(tokens: Iterator[str]) -> Iterator[str]:
    return (t.replace(".", "").lower() for t in tokens)


def _stemmer_filter(tokens: Iterator[str]) -> Iterator[str]:
    return (t.replace("ing", "").replace("ed", "") for t in tokens)


def _stop_word_filter(tokens: Iterator[str]) -> Iterator[str]:
    return (t for t in tokens if t not in STOP_WORDS)
