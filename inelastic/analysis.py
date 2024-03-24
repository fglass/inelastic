from typing import Iterator

# https://github.com/apache/lucene/blob/main/lucene/analysis/common/src/java/org/apache/lucene/analysis/en/EnglishAnalyzer.java
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "if",
    "in",
    "into",
    "is",
    "it",
    "no",
    "not",
    "of",
    "on",
    "or",
    "such",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "was",
    "will",
    "with",
}


def analyze(text: str) -> list[str]:
    tokens = _tokenize(text)
    tokens = _filter(tokens)
    return list(tokens)


def _tokenize(text: str) -> Iterator[str]:
    return (t for t in text.replace("-", " ").split())


def _filter(tokens: Iterator[str]) -> Iterator[str]:
    tokens = _normalization_filter(tokens)
    tokens = _stop_word_filter(tokens)
    return tokens


#######################################################################################
# Token filters
#######################################################################################


def _normalization_filter(tokens: Iterator[str]) -> Iterator[str]:
    return (  # TODO: regex
        t.replace(".", "")
        .replace(":", "")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
        .lower()
        for t in tokens
    )


def _stemmer_filter(tokens: Iterator[str]) -> Iterator[str]:
    return (t.replace("ing", "").replace("ed", "") for t in tokens)


def _stop_word_filter(tokens: Iterator[str]) -> Iterator[str]:
    return (t for t in tokens if t not in STOP_WORDS)
