"""Speech-to-text slate parsing."""

from ttucs.slate.normalize import normalize_transcript, words_to_numbers
from ttucs.slate.parser import SlateFields, SlateParser, parse_slate

__all__ = ["SlateFields", "SlateParser", "normalize_transcript", "parse_slate", "words_to_numbers"]
