"""Smoke tests for Chinese (Mandarin) G2P pipeline.

Skipped automatically when G2P dependencies (pypinyin, g2pM, jieba_fast) are not installed.
"""
import pytest

pypinyin = pytest.importorskip("pypinyin", reason="pypinyin not installed; skipping Chinese G2P tests")

from genie_tts.G2P.Chinese.ChineseG2P import chinese_to_phones  # noqa: E402
from genie_tts.G2P.SymbolsV2 import symbol_to_id_v2  # noqa: E402


SAMPLE_SENTENCES = [
    "你好，今天天气怎么样？",
    "我喜欢学习中文。",
    "语音合成技术非常先进！",
    "他在北京工作。",
]


class TestChineseG2P:
    def test_returns_four_tuple(self):
        result = chinese_to_phones("你好")
        assert len(result) == 4  # (text_clean, norm_text, phones, word2ph)

    def test_phones_non_empty(self):
        _, _, phones, _ = chinese_to_phones("你好")
        assert len(phones) > 0

    def test_word2ph_matches_phones(self):
        _, _, phones, word2ph = chinese_to_phones("你好世界")
        assert sum(word2ph) == len(phones), (
            f"sum(word2ph)={sum(word2ph)} != len(phones)={len(phones)}"
        )

    def test_all_ids_valid(self):
        _, _, phones, _ = chinese_to_phones("你好")
        valid_ids = set(symbol_to_id_v2.values())
        for phone_id in phones:
            assert phone_id in valid_ids, f"Invalid phone ID: {phone_id}"

    @pytest.mark.parametrize("sentence", SAMPLE_SENTENCES)
    def test_sample_sentences(self, sentence):
        text_clean, norm_text, phones, word2ph = chinese_to_phones(sentence)
        assert len(phones) > 0
        assert sum(word2ph) == len(phones)

    def test_mixed_chinese_english(self):
        # Chinese G2P should handle embedded letters gracefully
        _, _, phones, word2ph = chinese_to_phones("我用Python编程")
        assert len(phones) > 0
