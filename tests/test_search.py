from inelastic.index import index
from inelastic.query import QueryConfig, Result, query
from inelastic.score import raw_tf_idf_strategy


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
    results = query(idx, q="fox box", config=QueryConfig(conjunctive=True))

    assert len(results) == 0


def test_disjunctive_search():
    docs = [
        ("title0", "The quick brown fox jumps over the lazy dog"),
        ("title1", "Pack my box with five dozen liquor jugs"),
    ]

    idx = index(docs)
    results = query(idx, q="fox box", config=QueryConfig(conjunctive=False))

    _assert_in_results(0, results)
    _assert_in_results(1, results)


def test_ranking():
    docs = [
        ("title0", "dog"),
        ("title1", "cat cat cat dog dog"),
    ]

    idx = index(docs)
    results = query(
        idx, q="dog", config=QueryConfig(score_strategy=raw_tf_idf_strategy)
    )

    assert results[0].doc_id == 1
    assert results[1].doc_id == 0


def _assert_in_results(doc_id: int, results: list[Result]):
    assert doc_id in {r.doc_id for r in results}
