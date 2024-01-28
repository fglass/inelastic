import math
from typing import NamedTuple

from inelastic.analysis import analyze

InvertedIndex = dict[str, dict[int, int]]  # Term -> postings (sorted) -> tf
DocumentIndex = dict[int, float]  # Doc ID -> field norm
Index = NamedTuple("Index", [("inverted", InvertedIndex), ("docs", DocumentIndex)])


def index(docs: list[tuple[str, str]]) -> Index:
    inverted_idx: InvertedIndex = {}
    doc_idx, terms = _analyze_documents(docs)

    for term, doc_id in terms:
        if term not in inverted_idx:
            inverted_idx[term] = {}

        postings_list = inverted_idx[term]
        postings_list[doc_id] = postings_list.get(doc_id, 0) + 1

    return Index(inverted_idx, doc_idx)


def _analyze_documents(docs: list[tuple[str, str]]):
    doc_idx: DocumentIndex = {}
    terms = []

    for doc_id, (title, contents) in enumerate(docs):
        tokens = analyze(contents)  # TODO: multi-field
        field_length = len(tokens)

        if field_length == 0:
            continue

        field_norm = 1 / math.sqrt(field_length)
        doc_idx[doc_id] = field_norm  # No lossy encoding

        for token in tokens:
            terms.append((token, doc_id))

    terms.sort()  # Sort by term then doc ID

    return doc_idx, terms
