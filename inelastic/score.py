import math

from inelastic.index import Index


def raw_tf_idf_strategy(index: Index, tokens: list[str], doc_id: int) -> float:
    score = 0.0

    for token in tokens:
        posting_entry = index.get(token, {}).get(doc_id)

        if posting_entry is None or posting_entry.tf == 0:
            continue

        df = len(index[token])
        tf_idf = posting_entry.tf / df
        score += tf_idf

    return score


def lucene_classic_strategy(index: Index, tokens: list[str], doc_id: int) -> float:
    score = 0.0
    n_docs = len(index)
    n_tokens_found = 0

    for token in tokens:
        posting_entry = index.get(token, {}).get(doc_id)

        if posting_entry is None or posting_entry.tf == 0:
            continue

        tf = math.sqrt(posting_entry.tf)

        df = len(index[token])
        idf = math.log(n_docs / (df + 1)) + 1

        field_length = posting_entry.field_length
        field_norm = 1 / math.sqrt(field_length)

        score += tf * idf * field_norm
        n_tokens_found += 1

    coord = n_tokens_found / len(tokens)
    return score * coord
