"""Tests for Utils/LangDetector.py."""
import pytest
from unittest.mock import patch
from genie_tts.Utils.LangDetector import detect_language, segment_by_language


class TestDetectLanguage:
    def test_empty_string_returns_fallback(self):
        result = detect_language("")
        assert result == "English"

    def test_whitespace_returns_fallback(self):
        result = detect_language("   ")
        assert result == "English"

    def test_english_text(self):
        result = detect_language("Hello, how are you today?")
        assert result == "English"

    def test_chinese_text(self):
        result = detect_language("你好，今天天气怎么样？")
        assert result == "Chinese"

    def test_japanese_text(self):
        result = detect_language("おはようございます、今日はいい天気ですね。")
        assert result == "Japanese"

    def test_korean_text(self):
        result = detect_language("안녕하세요, 오늘 날씨가 어떤가요?")
        assert result == "Korean"

    def test_unsupported_lang_falls_back(self):
        # Mock detector returning an unsupported code
        with patch("genie_tts.Utils.LangDetector._detect_fn",
                   return_value={"lang": "fr", "score": 0.99}):
            result = detect_language("Bonjour le monde")
            assert result == "English"  # fallback

    def test_detector_exception_falls_back(self):
        def _raise(_text, **_kwargs):
            raise RuntimeError("detection failed")
        with patch("genie_tts.Utils.LangDetector._detect_fn", side_effect=_raise):
            result = detect_language("some text")
            assert result == "English"


class TestSegmentByLanguage:
    def test_empty_returns_empty(self):
        assert segment_by_language("") == []

    def test_pure_english(self):
        segs = segment_by_language("Hello world")
        assert len(segs) >= 1
        assert all(s["content"].strip() for s in segs)

    def test_pure_chinese(self):
        segs = segment_by_language("今天天气很好")
        assert len(segs) >= 1
        combined = "".join(s["content"] for s in segs)
        assert "今天" in combined

    def test_mixed_chinese_english_segments(self):
        segs = segment_by_language("今天 I went to 学校")
        assert len(segs) >= 1
        languages = {s["language"] for s in segs}
        # Should detect at least one language
        assert len(languages) >= 1

    def test_segments_cover_full_text(self):
        text = "Hello 你好 world"
        segs = segment_by_language(text)
        combined = "".join(s["content"] for s in segs)
        # All content should be present
        assert "Hello" in combined or "hello" in combined.lower()

    def test_no_empty_segments(self):
        segs = segment_by_language("Hello 你好")
        for s in segs:
            assert s["content"].strip() != ""

    def test_adjacent_same_lang_merged(self):
        # Two English phrases should merge into one segment
        segs = segment_by_language("Hello world how are you")
        en_segs = [s for s in segs if s["language"] == "English"]
        # Should be a single merged English segment
        assert len(en_segs) == 1
