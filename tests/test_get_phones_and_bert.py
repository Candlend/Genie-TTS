"""Tests for GetPhonesAndBert dispatch (auto, hybrid, single-language).

These tests exercise the routing logic only — G2P backends are mocked so the
suite runs without installing pyopenjtalk, g2p_en, pypinyin, etc.
"""
from __future__ import annotations

import sys
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Pre-stub G2P sub-modules that may not be installed (e.g. Korean requires g2pk2)
# This must happen before any import of GetPhonesAndBert so lazy imports resolve.
# ---------------------------------------------------------------------------

def _ensure_g2p_stubs():
    stubs = {
        "genie_tts.G2P.Korean": MagicMock(),
        "genie_tts.G2P.Korean.KoreanG2P": MagicMock(),
    }
    for name, mock in stubs.items():
        sys.modules.setdefault(name, mock)


_ensure_g2p_stubs()


# ---------------------------------------------------------------------------
# Patch target constants
# ---------------------------------------------------------------------------

_G2P_EN = "genie_tts.G2P.English.EnglishG2P.english_to_phones"
_G2P_ZH = "genie_tts.G2P.Chinese.ChineseG2P.chinese_to_phones"
_G2P_JA = "genie_tts.G2P.Japanese.JapaneseG2P.japanese_to_phones"
# Korean is stubbed in sys.modules; patch via the stub directly
_G2P_KO_MOD = "genie_tts.G2P.Korean.KoreanG2P"

# segment_by_language is lazily imported from LangDetector inside get_phones_and_bert
_SEGMENT = "genie_tts.Utils.LangDetector.segment_by_language"


# ---------------------------------------------------------------------------
# Fixture: mock model_manager so BERT branch (roberta_model) is skipped
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_model_manager():
    """Stub model_manager so load_roberta_model() returns False (no BERT)."""
    import genie_tts.GetPhonesAndBert as mod
    orig = mod.model_manager
    stub = MagicMock()
    stub.roberta_model = None
    stub.load_roberta_model = MagicMock(return_value=False)
    mod.model_manager = stub
    yield
    mod.model_manager = orig


# ---------------------------------------------------------------------------
# Tests: single-language dispatch
# ---------------------------------------------------------------------------

class TestSingleLanguageDispatch:

    def test_english_dispatch(self):
        from genie_tts.GetPhonesAndBert import get_phones_and_bert
        fake = (np.array([[1, 2, 3]], dtype=np.int64), np.zeros((3, 1024), dtype=np.float32))
        with patch("genie_tts.GetPhonesAndBert._get_phones_and_bert_single", return_value=fake) as mock:
            phones, bert = get_phones_and_bert("Hello world", language="English")
        mock.assert_called_once_with("Hello world", "English")
        assert phones.shape == (1, 3)
        assert bert.shape == (3, 1024)
        assert np.all(bert == 0)

    def test_japanese_dispatch(self):
        from genie_tts.GetPhonesAndBert import get_phones_and_bert
        fake = (np.array([[10, 20]], dtype=np.int64), np.zeros((2, 1024), dtype=np.float32))
        with patch("genie_tts.GetPhonesAndBert._get_phones_and_bert_single", return_value=fake) as mock:
            phones, bert = get_phones_and_bert("おはよう", language="Japanese")
        mock.assert_called_once_with("おはよう", "Japanese")
        assert phones.shape == (1, 2)

    def test_korean_dispatch(self):
        from genie_tts.GetPhonesAndBert import get_phones_and_bert
        fake = (np.array([[5, 6, 7, 8]], dtype=np.int64), np.zeros((4, 1024), dtype=np.float32))
        with patch("genie_tts.GetPhonesAndBert._get_phones_and_bert_single", return_value=fake) as mock:
            phones, bert = get_phones_and_bert("안녕하세요", language="Korean")
        mock.assert_called_once_with("안녕하세요", "Korean")
        assert phones.shape == (1, 4)

    def test_chinese_dispatch_no_bert(self):
        """Single-language Chinese dispatch routes through the Chinese branch."""
        from genie_tts.GetPhonesAndBert import get_phones_and_bert
        fake = (np.array([[1, 2, 3]], dtype=np.int64), np.zeros((3, 1024), dtype=np.float32))
        with patch("genie_tts.GetPhonesAndBert._get_phones_and_bert_single", return_value=fake) as mock:
            phones, bert = get_phones_and_bert("你好", language="Chinese")
        mock.assert_called_once_with("你好", "Chinese")
        assert phones.shape == (1, 3)
        assert np.all(bert == 0)


# ---------------------------------------------------------------------------
# Tests: Hybrid-Chinese-English
# ---------------------------------------------------------------------------

class TestHybridDispatch:

    def test_hybrid_splits_and_concatenates(self):
        """Hybrid mode splits on Latin chars and concatenates phone sequences."""
        from genie_tts.GetPhonesAndBert import get_phones_and_bert
        side_effect = [
            (np.array([[10, 11]], dtype=np.int64), np.zeros((2, 1024), dtype=np.float32)),
            (np.array([[20, 21, 22]], dtype=np.int64), np.zeros((3, 1024), dtype=np.float32)),
        ]
        with patch("genie_tts.GetPhonesAndBert._get_phones_and_bert_single", side_effect=side_effect) as mock_single:
            phones, bert = get_phones_and_bert("你好 hello", language="Hybrid-Chinese-English")
        assert mock_single.call_args_list == [
            (("你好 ", "chinese"),),
            (("hello", "english"),),
        ]
        assert phones.ndim == 2
        assert phones.shape[0] == 1
        assert phones.shape[1] == 5

    def test_hybrid_pure_chinese(self):
        """Pure Chinese text in hybrid mode only calls Chinese routing."""
        from genie_tts.GetPhonesAndBert import get_phones_and_bert
        fake = (np.array([[1, 2]], dtype=np.int64), np.zeros((2, 1024), dtype=np.float32))
        with patch("genie_tts.GetPhonesAndBert._get_phones_and_bert_single", return_value=fake) as mock_single:
            phones, bert = get_phones_and_bert("纯中文", language="Hybrid-Chinese-English")
        mock_single.assert_called_once_with("纯中文", "chinese")
        assert phones.shape == (1, 2)


# ---------------------------------------------------------------------------
# Tests: Auto mode
# ---------------------------------------------------------------------------

class TestAutoDispatch:

    def test_auto_single_english_segment(self):
        """Single English segment routes to English path."""
        from genie_tts.GetPhonesAndBert import get_phones_and_bert
        segments = [{"language": "English", "content": "Hello world"}]
        fake = (np.array([[1, 2, 3]], dtype=np.int64), np.zeros((3, 1024), dtype=np.float32))
        with (
            patch(_SEGMENT, return_value=segments),
            patch("genie_tts.GetPhonesAndBert._get_phones_and_bert_single", return_value=fake) as mock_single,
        ):
            phones, bert = get_phones_and_bert("Hello world", language="auto")
        mock_single.assert_called_once_with("Hello world", "English")
        assert phones.shape == (1, 3)

    def test_auto_single_japanese_segment(self):
        """Single Japanese segment routes to Japanese path."""
        from genie_tts.GetPhonesAndBert import get_phones_and_bert
        segments = [{"language": "Japanese", "content": "おはよう"}]
        fake = (np.array([[10, 20, 30]], dtype=np.int64), np.zeros((3, 1024), dtype=np.float32))
        with (
            patch(_SEGMENT, return_value=segments),
            patch("genie_tts.GetPhonesAndBert._get_phones_and_bert_single", return_value=fake) as mock_single,
        ):
            phones, bert = get_phones_and_bert("おはよう", language="auto")
        mock_single.assert_called_once_with("おはよう", "Japanese")
        assert phones.shape == (1, 3)

    def test_auto_multi_segment_concatenates(self):
        """Two-segment auto mode concatenates phone sequences from both languages."""
        from genie_tts.GetPhonesAndBert import get_phones_and_bert
        segments = [
            {"language": "Chinese", "content": "你好"},
            {"language": "English", "content": " hello"},
        ]
        side_effect = [
            (np.array([[10, 11]], dtype=np.int64), np.zeros((2, 1024), dtype=np.float32)),
            (np.array([[20, 21, 22]], dtype=np.int64), np.zeros((3, 1024), dtype=np.float32)),
        ]
        with (
            patch(_SEGMENT, return_value=segments),
            patch("genie_tts.GetPhonesAndBert._get_phones_and_bert_single", side_effect=side_effect),
        ):
            phones, bert = get_phones_and_bert("你好 hello", language="auto")
        assert phones.shape == (1, 5)
        assert bert.shape == (5, 1024)

    def test_auto_empty_segments_fallback_to_japanese(self):
        """Empty segmentation falls back to Japanese path."""
        from genie_tts.GetPhonesAndBert import get_phones_and_bert
        fake = (np.array([[5, 6]], dtype=np.int64), np.zeros((2, 1024), dtype=np.float32))
        with (
            patch(_SEGMENT, return_value=[]),
            patch("genie_tts.GetPhonesAndBert._get_phones_and_bert_single", return_value=fake) as mock_single,
        ):
            phones, bert = get_phones_and_bert("???", language="auto")
        mock_single.assert_called_once_with("???", "japanese")
        assert phones.shape == (1, 2)

    def test_auto_mixed_chinese_japanese_routes_each_chunk(self):
        from genie_tts.GetPhonesAndBert import get_phones_and_bert
        segments = [
            {"language": "Chinese", "content": "我喜欢東京！"},
            {"language": "Japanese", "content": "でも今日は忙しい。"},
        ]
        side_effect = [
            (np.array([[10, 11]], dtype=np.int64), np.zeros((2, 1024), dtype=np.float32)),
            (np.array([[20, 21, 22]], dtype=np.int64), np.zeros((3, 1024), dtype=np.float32)),
        ]
        with (
            patch(_SEGMENT, return_value=segments),
            patch("genie_tts.GetPhonesAndBert._get_phones_and_bert_single", side_effect=side_effect) as mock_single,
        ):
            phones, bert = get_phones_and_bert("我喜欢東京！でも今日は忙しい。", language="auto")
        assert mock_single.call_args_list == [
            (("我喜欢東京！", "Chinese"),),
            (("でも今日は忙しい。", "Japanese"),),
        ]
        assert phones.shape == (1, 5)
        assert bert.shape == (5, 1024)
