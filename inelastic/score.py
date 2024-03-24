import math

from inelastic.index import Index


def raw_tf_idf_strategy(idx: Index, query_terms: list[str], doc_id: int) -> float:
    score = 0.0

    for term in query_terms:
        postings_list = idx.inverted.get(term, {})
        tf = postings_list.get(doc_id)

        if tf is not None:
            df = len(postings_list)
            idf = 1 / (df * df)
            score += tf * idf

    return score


def lucene_classic_strategy(idx: Index, query_terms: list[str], doc_id: int) -> float:
    weight = 0.0

    n_docs = len(idx.docs) + 1
    sum_of_squared_weights = 0.0
    n_terms_found = 0

    for term in query_terms:
        postings_list = idx.inverted.get(term, {})

        df = len(postings_list)
        idf = 1 + math.log(n_docs / (df + 1))
        idf_squared = idf * idf
        sum_of_squared_weights += idf_squared

        if tf := postings_list.get(doc_id):
            field_length = idx.docs[doc_id]
            field_norm = 1 / math.sqrt(field_length)  # No lossy encoding
            weight += math.sqrt(tf) * idf_squared * field_norm
            n_terms_found += 1

    coord = n_terms_found / len(query_terms)
    query_norm = 1 / math.sqrt(sum_of_squared_weights)

    return weight * coord * query_norm


def lucene_bm25_strategy(
    idx: Index,
    query_terms: list[str],
    doc_id: int,
    boost: float = 2.2,
    k1: float = 1.2,
    b: float = 0.75,
) -> float:
    score = 0.0
    n_docs = len(idx.docs)

    for term in query_terms:
        postings_list = idx.inverted.get(term, {})
        freq = postings_list.get(doc_id)

        if freq is None:
            continue

        df = len(postings_list)
        idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))

        field_length = idx.docs[doc_id]
        b_factor = (1 - b) + b * (field_length / idx.avg_field_length)
        tf = freq / (freq + k1 * b_factor)

        score += boost * idf * tf

    return score
