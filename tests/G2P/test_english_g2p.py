"""Smoke tests for English G2P pipeline.

Skipped automatically when G2P dependencies (nltk, g2p_en) are not installed.
"""
import pytest

nltk = pytest.importorskip("nltk", reason="nltk not installed; skipping English G2P tests")

from genie_tts.G2P.English.EnglishG2P import english_to_phones  # noqa: E402
from genie_tts.G2P.SymbolsV2 import symbol_to_id_v2  # noqa: E402


SAMPLE_SENTENCES = [
    "Hello, how are you?",
    "The quick brown fox jumps over the lazy dog.",
    "Speech synthesis is amazing!",
]


class TestEnglishG2P:
    def test_returns_list(self):
        result = english_to_phones("Hello world")
        assert isinstance(result, list)

    def test_non_empty(self):
        result = english_to_phones("Hello")
        assert len(result) > 0

    def test_all_ids_valid(self):
        """All returned phone IDs must be valid symbol_to_id_v2 values."""
        result = english_to_phones("Hello world")
        valid_ids = set(symbol_to_id_v2.values())
        for phone_id in result:
            assert phone_id in valid_ids, f"Invalid phone ID: {phone_id}"

    @pytest.mark.parametrize("sentence", SAMPLE_SENTENCES)
    def test_sample_sentences(self, sentence):
        result = english_to_phones(sentence)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_punctuation_only(self):
        # Should not crash on punctuation-only input
        result = english_to_phones("...")
        assert isinstance(result, list)

    def test_single_word(self):
        result = english_to_phones("cat")
        assert len(result) > 0
