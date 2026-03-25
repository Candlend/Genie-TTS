"""Smoke tests for Korean G2P pipeline.

Skipped automatically when G2P dependencies (g2pk2, jamo, ko_pron) are not installed.
"""
import pytest

g2pk2 = pytest.importorskip("g2pk2", reason="g2pk2 not installed; skipping Korean G2P tests")

from genie_tts.G2P.Korean.KoreanG2P import korean_to_phones  # noqa: E402
from genie_tts.G2P.SymbolsV2 import symbol_to_id_v2  # noqa: E402


SAMPLE_SENTENCES = [
    "안녕하세요.",
    "오늘 날씨가 좋네요.",
    "한국어 음성 합성입니다.",
    "저는 학생입니다.",
]


class TestKoreanG2P:
    def test_returns_list(self):
        result = korean_to_phones("안녕")
        assert isinstance(result, list)

    def test_non_empty(self):
        result = korean_to_phones("안녕")
        assert len(result) > 0

    def test_all_ids_valid(self):
        result = korean_to_phones("안녕")
        valid_ids = set(symbol_to_id_v2.values())
        for phone_id in result:
            assert phone_id in valid_ids, f"Invalid phone ID: {phone_id}"

    @pytest.mark.parametrize("sentence", SAMPLE_SENTENCES)
    def test_sample_sentences(self, sentence):
        result = korean_to_phones(sentence)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_single_word(self):
        result = korean_to_phones("사랑")
        assert len(result) > 0

    def test_with_punctuation(self):
        result = korean_to_phones("안녕하세요!")
        assert len(result) > 0
