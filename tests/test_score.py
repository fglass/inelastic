from typing import Callable
from inelastic.index import Index, index
from inelastic.score import (
    lucene_bm25_strategy,
    lucene_classic_strategy,
    raw_tf_idf_strategy,
)

SAMPLE_WIKI_DOCS: list[tuple[str, str]] = [
    ("Lynx (disambiguation)", "A lynx is a type of wild cat...."),
    ("Bobcat (disambiguation)", "Bobcat is a species of wild cat in North America...."),
    (
        "Wildcat",
        "The wildcat is a species complex comprising two small wild cat species: the European wildcat (Felis silvestris) and the African wildcat (F. lybica)....",
    ),
    (
        "Ocicat",
        "The Ocicat is an all-domestic breed of cat which resembles a wild cat but has no recent wild DNA in its gene pool. The breed is unusual in that it has a spotted tabby pattern, like a wild cat, but has the temperament of a domestic animal....",
    ),
    (
        "Random permutation",
        "A random permutation is a random ordering of a set of objects, that is, a permutation-valued random variable.  The use of random permutations is often fundamental to fields that use randomized algorithms such as coding theory, cryptography, and simulation....",
    ),
]


def test_raw_tf_idf_scoring():
    idx = index(SAMPLE_WIKI_DOCS)
    query_terms = ["small", "wild", "cat"]
    score_strategy = raw_tf_idf_strategy

    scores = _calculate_scores(idx, query_terms, score_strategy)

    assert scores == [
        ("Wildcat", 1.125),
        ("Ocicat", 0.375),
        ("Lynx (disambiguation)", 0.125),
        ("Bobcat (disambiguation)", 0.125),
        ("Random permutation", 0.0),
    ]


def test_lucene_classic_strategy():
    """
    Compared to Elasticsearch v5.6.16 with identical document set.
    Scores are not 1:1 due to Elasticsearch's lossy encoding of field norms
    """
    idx = index(SAMPLE_WIKI_DOCS)
    query_terms = ["small", "wild", "cat"]
    score_strategy = lucene_classic_strategy

    scores = _calculate_scores(idx, query_terms, score_strategy)

    assert scores == [
        ("Wildcat", 0.6507887551664275),  # ES=0.5869655 (fieldNorm=0.21875)
        ("Lynx (disambiguation)", 0.34730853488543767),
        (
            "Bobcat (disambiguation)",
            0.28357623126097775,  # ES=0.26048142 (fieldNorm=0.375)
        ),
        ("Ocicat", 0.22341230022409633),  # ES=0.18798625 (fieldNorm=0.15625)
        ("Random permutation", 0.0),
    ]


def test_lucene_bm25_strategy():
    """
    Compared to Elasticsearch v7.17.18 with identical document set.
    Scores match but Elasticsearch's scores have lower precision
    """
    idx = index(SAMPLE_WIKI_DOCS)
    query_terms = ["small", "wild", "cat"]
    score_strategy = lucene_bm25_strategy

    scores = _calculate_scores(idx, query_terms, score_strategy)

    assert scores == [
        ("Wildcat", 1.9025460287214067),
        ("Lynx (disambiguation)", 0.8284862335065372),
        ("Bobcat (disambiguation)", 0.7709968264012261),
        ("Ocicat", 0.7668580397564343),
        ("Random permutation", 0.0),
    ]


def _calculate_scores(
    idx: Index,
    query_terms: list[str],
    score_strategy: Callable[[Index, list[str], int], float],
) -> list[tuple[str, float]]:

    scores = [
        (title, score_strategy(idx, query_terms, doc_id))
        for doc_id, (title, _) in enumerate(SAMPLE_WIKI_DOCS)
    ]
    scores.sort(key=lambda x: x[1], reverse=True)

    return scores
