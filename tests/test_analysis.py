import pytest

from inelastic.analysis import analyze


@pytest.mark.parametrize(
    "text,tokens",
    [
        ("", []),
        ("abc", ["abc"]),
        ("small wild cat!", ["small", "wild", "cat!"]),
    ],
)
def test_analysis(text: str, tokens: list[str]):
    assert analyze(text) == tokens
