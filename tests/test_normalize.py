import pytest

from ttucs.slate.normalize import normalize_transcript, words_to_numbers


@pytest.mark.parametrize("spoken,expected", [
    ("take three", "take 3"),
    ("ninety six k twenty four bit", "96 k 24 bit"),
    ("forty eight k", "48 k"),
    ("mkh four sixteen", "mkh 416"),
    ("mkh eighty forty", "mkh 8040"),
    ("sennheiser m k eighty forty", "sennheiser mk 8040"),
    ("setup in a dual a b", "setup in a dual ab"),
    ("a b stereo pair", "ab stereo pair"),
])
def test_number_words(spoken, expected):
    assert normalize_transcript(spoken) == expected


def test_fillers_removed():
    out = normalize_transcript("um okay so uh this is a door creak")
    assert "um" not in out.split() and "uh" not in out.split()
    assert out.endswith("door creak")


def test_forty_four_one():
    assert "44.1 k" in normalize_transcript("forty four one k sixteen bit")


def test_words_to_numbers_leaves_plain_words():
    assert words_to_numbers("a wooden door") == "a wooden door"
