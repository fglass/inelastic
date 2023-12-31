from typing import NamedTuple

from inelastic.analysis import analyze

PostingEntry = NamedTuple("PostingEntry", [("tf", int), ("field_length", int)])
# Term -> postings (sorted) -> posting_entry(tf, field_length)
Index = dict[str, dict[int, PostingEntry]]


def index(docs: list[tuple[str, str]]) -> Index:
    idx: Index = {}

    for term, doc_id, field_length in _extract_terms(docs):
        if term not in idx:
            idx[term] = {}
        tf = idx[term].get(doc_id, PostingEntry(0, 0)).tf
        idx[term][doc_id] = PostingEntry(tf + 1, field_length)

    return idx


def _extract_terms(docs: list[tuple[str, str]]) -> list[tuple[str, int, int]]:
    terms = []

    for doc_id, (title, contents) in enumerate(docs):
        tokens = analyze(contents)
        for token in tokens:
            terms.append((token, doc_id, len(tokens)))

    return sorted(terms)  # Sort by term then doc ID
