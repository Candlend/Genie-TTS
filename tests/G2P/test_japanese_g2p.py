"""Smoke tests for Japanese G2P pipeline.

Skipped automatically when G2P dependencies (pyopenjtalk) are not installed.
"""
import pytest

pyopenjtalk = pytest.importorskip("pyopenjtalk", reason="pyopenjtalk not installed; skipping Japanese G2P tests")

from genie_tts.G2P.Japanese.JapaneseG2P import japanese_to_phones  # noqa: E402
from genie_tts.G2P.SymbolsV2 import symbol_to_id_v2  # noqa: E402


SAMPLE_SENTENCES = [
    "おはようございます。",
    "今日はいい天気ですね。",
    "音声合成の技術は素晴らしいです！",
    "日本語のテキスト読み上げです。",
]


class TestJapaneseG2P:
    def test_returns_list(self):
        result = japanese_to_phones("こんにちは")
        assert isinstance(result, list)

    def test_non_empty(self):
        result = japanese_to_phones("こんにちは")
        assert len(result) > 0

    def test_all_ids_valid(self):
        result = japanese_to_phones("こんにちは")
        valid_ids = set(symbol_to_id_v2.values())
        for phone_id in result:
            assert phone_id in valid_ids, f"Invalid phone ID: {phone_id}"

    @pytest.mark.parametrize("sentence", SAMPLE_SENTENCES)
    def test_sample_sentences(self, sentence):
        result = japanese_to_phones(sentence)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_katakana_input(self):
        result = japanese_to_phones("アイウエオ")
        assert len(result) > 0

    def test_mixed_kanji_kana(self):
        result = japanese_to_phones("東京は大きな都市です")
        assert len(result) > 0
