from inelastic.index import index
from inelastic.score import lucene_classic_strategy, raw_tf_idf_strategy


def test_raw_tf_idf_scoring():
    docs = [
        ("", "wild dog"),
        ("", "cat cat cat dog dog"),
        ("", "cat sat on a mat"),
    ]
    query_terms = ["wild", "cat"]

    idx = index(docs)
    scores = [
        raw_tf_idf_strategy(idx, query_terms, doc_id) for doc_id in range(len(docs))
    ]

    assert scores == [1.0, 0.75, 0.25]


def test_lucene_classic_scoring():
    docs = [
        ("", "wild dog"),
        ("", "cat cat cat dog dog"),
        ("", "cat sat on a mat"),
    ]
    query_terms = ["wild", "cat"]

    idx = index(docs)
    scores = [
        lucene_classic_strategy(idx, query_terms, doc_id) for doc_id in range(len(docs))
    ]

    assert scores == [0.47647624893784046, 0.3018976658796704, 0.1948741053684758]


def test_against_elasticsearch_single_term():
    docs = [("", "quick brown fox")]
    query_terms = ["fox"]

    idx = index(docs)
    scores = [
        lucene_classic_strategy(idx, query_terms, doc_id) for doc_id in range(len(docs))
    ]

    assert scores == [0.5773502691896258]


def test_against_elasticsearch_multi_term():
    docs = [("", "quick brown fox")]
    query_terms = ["quick", "brown", "fox"]

    idx = index(docs)
    scores = [
        lucene_classic_strategy(idx, query_terms, doc_id) for doc_id in range(len(docs))
    ]

    # NOTE: es=0.8660254037844388 due to lossy norm encoding
    assert scores == [1.0000000000000002]


def test_against_elasticsearch_missing_term():
    docs = [("", "quick brown")]
    query_terms = ["quick", "brown", "dead"]

    idx = index(docs)
    scores = [
        lucene_classic_strategy(idx, query_terms, doc_id) for doc_id in range(len(docs))
    ]

    # NOTE: es=0.3021964368185907 due to lossy norm encoding
    assert scores == [0.4273702994496751]
