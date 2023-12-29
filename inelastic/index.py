from inelastic.analysis import analyze

# Term -> postings (sorted) -> tf
Index = dict[str, dict[int, int]]


def index(docs: list[tuple[str, str]]) -> Index:
    idx: Index = {}

    # Add doc ID to postings and increment term frequency
    for term, doc_id in _extract_terms(docs):
        if term not in idx:
            idx[term] = {}
        idx[term][doc_id] = idx[term].get(doc_id, 0) + 1

    return idx


def _extract_terms(docs: list[tuple[str, str]]) -> list[tuple[str, int]]:
    terms = []

    for doc_id, (title, contents) in enumerate(docs):
        for term in analyze(contents):
            terms.append((term, doc_id))

    return sorted(terms)  # Sort by term then doc ID
