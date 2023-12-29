from inelastic.index import index
from inelastic.query import Result, _calc_raw_tf_idf, query


def test_exact_search():
    docs = [("doc0", "abc")]

    idx = index(docs)
    results = query(idx, q="abc")

    _assert_in_results(0, results)


def test_empty_search():
    docs = [("doc0", "abc")]

    idx = index(docs)
    results = query(idx, q="xyz")

    assert len(results) == 0


def test_conjunctive_search():
    docs = [
        ("title0", "The quick brown fox jumps over the lazy dog"),
        ("title1", "Pack my box with five dozen liquor jugs"),
    ]

    idx = index(docs)
    results = query(idx, q="fox box", conjunctive=True)

    assert len(results) == 0


def test_disjunctive_search():
    docs = [
        ("title0", "The quick brown fox jumps over the lazy dog"),
        ("title1", "Pack my box with five dozen liquor jugs"),
    ]

    idx = index(docs)
    results = query(idx, q="fox box", conjunctive=False)

    _assert_in_results(0, results)
    _assert_in_results(1, results)


def test_ranking():
    docs = [
        ("title0", "dog"),
        ("title1", "cat cat cat dog dog"),
    ]

    idx = index(docs)
    results = query(idx, q="dog")

    assert results[0].doc_id == 1
    assert results[1].doc_id == 0


def test_raw_tf_idf_scoring():
    docs = [
        ("title0", "wild dog"),
        ("title1", "cat cat cat dog dog"),
        ("title2", "cat sat on a mat"),
    ]

    idx = index(docs)
    tokens = ["wild", "cat"]
    scores = [_calc_raw_tf_idf(idx, tokens, doc_id) for doc_id in range(len(docs))]

    assert scores == [1.0, 1.5, 0.5]


def _assert_in_results(doc_id: int, results: list[Result]):
    assert doc_id in {r.doc_id for r in results}
