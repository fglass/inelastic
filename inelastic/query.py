from dataclasses import dataclass
from typing import Callable

from inelastic import score
from inelastic.analysis import analyze
from inelastic.index import Index


@dataclass
class Config:
    conjunctive: bool = False
    score_strategy: Callable[[Index, list[str], int], float] = score.raw_tf_idf_strategy


@dataclass
class Result:
    score: float
    doc_id: int


def query(idx: Index, q: str, config: Config | None = None) -> list[Result]:
    query_terms = analyze(q)
    config = config or Config()

    doc_ids = _retrieve(idx, query_terms, config.conjunctive)
    return _rank(
        doc_ids,
        score_strategy=lambda doc_id: config.score_strategy(idx, query_terms, doc_id),
    )


def _retrieve(idx: Index, query_terms: list[str], conjunctive: bool) -> list[int]:
    doc_ids: list[int] = []

    for i, term in enumerate(query_terms):
        if term not in idx.inverted:
            continue

        postings_list = list(idx.inverted[term].keys())

        if conjunctive and len(doc_ids) == 0 and i == 0:
            doc_ids = postings_list
        else:
            doc_ids = _merge_postings_list(doc_ids, postings_list, conjunctive)

    return doc_ids


def _rank(doc_ids: list[int], score_strategy: Callable[[int], float]) -> list[Result]:
    results = [Result(score_strategy(d), d) for d in doc_ids]
    results.sort(key=lambda r: r.score, reverse=True)  # TODO: use heap
    return results


def _merge_postings_list(a: list[int], b: list[int], conjunctive: bool) -> list[int]:
    return _intersect_postings_list(a, b) if conjunctive else _union_postings_list(a, b)


def _intersect_postings_list(a: list[int], b: list[int]) -> list[int]:
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


def _union_postings_list(a: list[int], b: list[int]) -> list[int]:
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
