from typing import NamedTuple

from inelastic.analysis import analyze

InvertedIndex = dict[str, dict[int, int]]  # Term -> postings (sorted) -> tf
DocumentIndex = dict[int, float]  # Doc ID -> field length
Index = NamedTuple(
    "Index",
    [("inverted", InvertedIndex), ("docs", DocumentIndex), ("avg_field_length", float)],
)


def index(docs: list[tuple[str, str]]) -> Index:
    inverted_idx: InvertedIndex = {}
    doc_idx, terms, avg_field_length = _analyze_documents(docs)

    for term, doc_id in terms:
        if term not in inverted_idx:
            inverted_idx[term] = {}

        postings_list = inverted_idx[term]
        postings_list[doc_id] = postings_list.get(doc_id, 0) + 1

    return Index(inverted_idx, doc_idx, avg_field_length)


def _analyze_documents(
    docs: list[tuple[str, str]]
) -> tuple[DocumentIndex, list[tuple[str, int]], float]:
    doc_idx: DocumentIndex = {}
    terms = []
    total_field_length = 0.0

    for doc_id, (title, contents) in enumerate(docs):
        tokens = analyze(contents)
        field_length = len(tokens)

        if field_length == 0:
            continue

        doc_idx[doc_id] = field_length
        total_field_length += field_length

        for token in tokens:
            terms.append((token, doc_id))

    avg_field_length = total_field_length / len(docs)

    terms.sort()  # Sort by term, then by doc ID

    return doc_idx, terms, avg_field_length
