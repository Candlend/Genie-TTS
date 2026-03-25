"""Tests for Utils/Language.py normalize_language()."""
import pytest
from genie_tts.Utils.Language import normalize_language, SUPPORTED_LANGUAGES


class TestNormalizeLanguage:
    # --- Japanese ---
    def test_ja(self):
        assert normalize_language("ja") == "Japanese"

    def test_jp(self):
        assert normalize_language("jp") == "Japanese"

    def test_jpn_iso639(self):
        assert normalize_language("jpn") == "Japanese"

    def test_japanese_full(self):
        assert normalize_language("Japanese") == "Japanese"

    def test_japanese_lower(self):
        assert normalize_language("japanese") == "Japanese"

    # --- English ---
    def test_en(self):
        assert normalize_language("en") == "English"

    def test_eng_iso639(self):
        assert normalize_language("eng") == "English"

    def test_en_gb(self):
        assert normalize_language("en-GB") == "English"

    # --- Chinese ---
    def test_zh(self):
        assert normalize_language("zh") == "Chinese"

    def test_cmn_iso639(self):
        assert normalize_language("cmn") == "Chinese"

    def test_zho_iso639(self):
        assert normalize_language("zho") == "Chinese"

    def test_zh_hans(self):
        assert normalize_language("zh-hans") == "Chinese"

    # --- Korean ---
    def test_ko(self):
        assert normalize_language("ko") == "Korean"

    def test_kor_iso639(self):
        assert normalize_language("kor") == "Korean"

    # --- Hybrid ---
    def test_hybrid(self):
        assert normalize_language("hybrid") == "auto"

    def test_hybrid_zh_en(self):
        assert normalize_language("hybrid-zh-en") == "Hybrid-Chinese-English"

    # --- Auto ---
    def test_auto(self):
        assert normalize_language("auto") == "auto"

    # --- Cantonese (reserved, logs warning) ---
    def test_yue_returns_cantonese(self):
        result = normalize_language("yue")
        assert result == "Cantonese"

    def test_cantonese_alias(self):
        assert normalize_language("cantonese") == "Cantonese"

    # --- Case insensitivity ---
    def test_uppercase_EN(self):
        assert normalize_language("EN") == "English"

    def test_mixed_case_Japanese(self):
        assert normalize_language("JAPANESE") == "Japanese"

    # --- Unknown code raises ValueError ---
    def test_unknown_code_raises(self):
        with pytest.raises(ValueError, match="Unknown language code"):
            normalize_language("xx")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            normalize_language("")

    def test_gibberish_raises(self):
        with pytest.raises(ValueError):
            normalize_language("foobar")


class TestSupportedLanguages:
    def test_supported_set_contains_core(self):
        for lang in ("Japanese", "English", "Chinese", "Korean", "auto"):
            assert lang in SUPPORTED_LANGUAGES

    def test_hybrid_in_supported(self):
        assert "Hybrid-Chinese-English" in SUPPORTED_LANGUAGES
