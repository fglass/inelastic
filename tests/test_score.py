from inelastic.index import index
from inelastic.score import lucene_classic_strategy, raw_tf_idf_strategy


def test_raw_tf_idf_scoring():
    docs = [
        ("title0", "wild dog"),
        ("title1", "cat cat cat dog dog"),
        ("title2", "cat sat on a mat"),
    ]

    idx = index(docs)
    tokens = ["wild", "cat"]
    scores = [raw_tf_idf_strategy(idx, tokens, doc_id) for doc_id in range(len(docs))]

    assert scores == [1.0, 1.5, 0.5]


def test_lucene_classic_scoring():
    docs = [
        ("title0", "wild dog"),
        ("title1", "cat cat cat dog dog"),
        ("title2", "cat sat on a mat"),
    ]

    idx = index(docs)
    tokens = ["wild", "cat"]
    scores = [
        lucene_classic_strategy(idx, tokens, doc_id) for doc_id in range(len(docs))
    ]

    assert scores == [0.7419714901993204, 0.655753083298671, 0.42328679513998635]
