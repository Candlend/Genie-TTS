# Tasks: Multilingual Support Improvement

**Spec ID:** 001

---

## US-3: Language Normalisation Completeness

- [x] T1: Add ISO 639-3 codes, `auto`, `yue` to `Language.py`; raise ValueError for unknown codes
- [x] T2: Write `tests/Utils/test_language.py`

## US-1 + US-2: Auto Detection & Mixed Segmentation

- [x] T3: Add `fast_langdetect` to `pyproject.toml`
- [x] T4: Create `Utils/LangDetector.py` with `detect_language()` and `segment_by_language()`
- [x] T5: Update `GetPhonesAndBert.py` to handle `language == "auto"`
- [x] T6: Write `tests/Utils/test_lang_detector.py`

## US-4: TextSplitter Language Awareness

- [x] T7: Fix `get_char_width()` in `TextSplitter.py` for Korean Hangul and jamo
- [x] T8: Write `tests/Utils/test_text_splitter.py`

## US-5: Unit Test Suite

- [x] T9: Create `tests/conftest.py` and `pyproject.toml` pytest config
- [x] T10: Write G2P smoke tests for Chinese, English, Japanese, Korean
- [x] T11: Run full test suite and fix failures

## Documentation

- [x] T12: Update README.md supported languages section
- [x] T13: Update spec status to Done
