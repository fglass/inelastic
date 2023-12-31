from dataclasses import dataclass
from typing import Callable

from inelastic import score
from inelastic.analysis import analyze
from inelastic.index import Index


@dataclass
class QueryConfig:
    conjunctive: bool = False
    score_strategy: Callable[
        [Index, list[str], int], float
    ] = score.lucene_classic_strategy


@dataclass
class Result:
    score: float
    doc_id: int


def query(index: Index, q: str, config: QueryConfig = QueryConfig()) -> list[Result]:
    tokens = analyze(q)
    doc_ids = _retrieve(index, tokens, config.conjunctive)
    return _rank(
        doc_ids,
        score_strategy=lambda doc_id: config.score_strategy(index, tokens, doc_id),
    )


def _retrieve(index: Index, tokens: list[str], conjunctive: bool) -> list[int]:
    doc_ids: list[int] = []

    for i, token in enumerate(tokens):
        if token not in index:
            continue

        postings = list(index[token].keys())

        if conjunctive and len(doc_ids) == 0 and i == 0:
            doc_ids = postings
        else:
            doc_ids = _merge_postings(doc_ids, postings, conjunctive)

    return doc_ids


def _rank(doc_ids: list[int], score_strategy: Callable[[int], float]) -> list[Result]:
    results = [Result(score_strategy(d), d) for d in doc_ids]
    results.sort(key=lambda r: r.score, reverse=True)
    return results


def _merge_postings(a: list[int], b: list[int], conjunctive: bool) -> list[int]:
    return _intersect_postings(a, b) if conjunctive else _union_postings(a, b)


def _intersect_postings(a: list[int], b: list[int]) -> list[int]:
    merged = []
    i = j = 0

    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            merged.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1

    return merged


def _union_postings(a: list[int], b: list[int]) -> list[int]:
    merged = []
    i = j = 0

    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            merged.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            merged.append(a[i])
            i += 1
        else:
            merged.append(b[j])
            j += 1

    merged.extend(a[i:])
    merged.extend(b[j:])

    return merged
