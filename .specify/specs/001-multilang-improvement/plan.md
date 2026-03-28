# Plan: Multilingual Support Improvement

**Spec ID:** 001
**Created:** 2026-03-25

---

## Architecture

### New Module: `Utils/LangDetector.py`
Provides two functions:
- `detect_language(text: str) -> str` — returns a canonical language name
- `segment_by_language(text: str) -> list[dict]` — splits text into `{language, content}` chunks

Uses `fast_langdetect` (already used in GPT-SoVITS reference). Lazy-imported to avoid startup cost.
Fallback: if detection confidence is low or language unsupported, logs warning and falls back to English.

### Updated: `Utils/Language.py`
- Add ISO 639-3 codes (`cmn`, `jpn`, `kor`, `eng`)
- Add `yue` -> `Cantonese` (with warning)
- Add `auto` -> `auto` passthrough
- `normalize_language()` raises `ValueError` for truly unknown codes

### Updated: `GetPhonesAndBert.py`
- `get_phones_and_bert()` handles `language == "auto"` by calling `segment_by_language()`
- Existing `Hybrid-Chinese-English` mode preserved for backward compatibility
- Both `auto` and `Hybrid-Chinese-English` delegate to same per-chunk dispatch logic (extracted helper)

### Updated: `Utils/TextSplitter.py`
- Fix `get_char_width()` to correctly handle Korean Hangul (U+AC00-U+D7A3) and jamo (U+3130-U+318F)
- Already handles CJK Unified Ideographs; verify Japanese kana coverage

### New: `tests/` directory
- `tests/conftest.py` — pytest configuration
- `tests/Utils/test_language.py` — normalize_language tests
- `tests/Utils/test_lang_detector.py` — LangDetector tests
- `tests/Utils/test_text_splitter.py` — TextSplitter char width tests
- `tests/G2P/test_chinese_g2p.py` — Chinese G2P smoke tests
- `tests/G2P/test_english_g2p.py` — English G2P smoke tests
- `tests/G2P/test_japanese_g2p.py` — Japanese G2P smoke tests
- `tests/G2P/test_korean_g2p.py` — Korean G2P smoke tests

## Dependencies
- `fast_langdetect` — add to `pyproject.toml`

## No Breaking Changes
- All existing `language` string values continue to work
- `Hybrid-Chinese-English` mode preserved
- Public API signatures unchanged
