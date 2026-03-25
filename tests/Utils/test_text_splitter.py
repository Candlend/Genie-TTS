"""Tests for Utils/TextSplitter.py."""
import pytest
from genie_tts.Utils.TextSplitter import TextSplitter


class TestGetCharWidth:
    def test_ascii_is_1(self):
        assert TextSplitter.get_char_width('a') == 1
        assert TextSplitter.get_char_width('Z') == 1
        assert TextSplitter.get_char_width('0') == 1
        assert TextSplitter.get_char_width(' ') == 1

    def test_cjk_unified_is_2(self):
        assert TextSplitter.get_char_width('你') == 2
        assert TextSplitter.get_char_width('語') == 2
        assert TextSplitter.get_char_width('日') == 2

    def test_hiragana_is_2(self):
        assert TextSplitter.get_char_width('あ') == 2
        assert TextSplitter.get_char_width('ん') == 2

    def test_katakana_is_2(self):
        assert TextSplitter.get_char_width('ア') == 2
        assert TextSplitter.get_char_width('ン') == 2

    def test_hangul_syllable_is_2(self):
        # U+AC00 가, U+D7A3 힣
        assert TextSplitter.get_char_width('가') == 2
        assert TextSplitter.get_char_width('힣') == 2
        assert TextSplitter.get_char_width('한') == 2

    def test_korean_compat_jamo_is_1(self):
        # U+3130-U+318F compatibility jamo (individual letters used in phoneme output)
        assert TextSplitter.get_char_width('ㄱ') == 1  # U+3131
        assert TextSplitter.get_char_width('ㅏ') == 1  # U+314F
        assert TextSplitter.get_char_width('ㅎ') == 1  # U+314E


class TestGetEffectiveLen:
    def setup_method(self):
        self.splitter = TextSplitter()

    def test_pure_ascii(self):
        assert self.splitter.get_effective_len("hello") == 5

    def test_pure_chinese(self):
        # 3 CJK chars = 6
        assert self.splitter.get_effective_len("你好啊") == 6

    def test_punctuation_excluded(self):
        # "你好" is 4, punctuation ignored
        assert self.splitter.get_effective_len("你好。") == 4

    def test_korean_syllables(self):
        # 3 Hangul syllables = 6
        assert self.splitter.get_effective_len("안녕하") == 6

    def test_japanese_hiragana(self):
        # 3 hiragana = 6
        assert self.splitter.get_effective_len("おはよ") == 6

    def test_mixed_ascii_cjk(self):
        # "hi" (2) + "你好" (4) = 6
        assert self.splitter.get_effective_len("hi你好") == 6


class TestSplit:
    def setup_method(self):
        self.splitter = TextSplitter(max_len=10, min_len=3)

    def test_single_sentence(self):
        result = self.splitter.split("Hello world.")
        assert len(result) == 1
        assert "Hello world" in result[0]

    def test_splits_on_period(self):
        result = self.splitter.split("First sentence. Second sentence.")
        assert len(result) == 2

    def test_chinese_split(self):
        result = self.splitter.split("你好，我是测试句。这是第二句。")
        assert len(result) >= 1

    def test_empty_string(self):
        result = self.splitter.split("")
        assert result == []

    def test_only_punctuation(self):
        result = self.splitter.split("...")
        assert result == []
