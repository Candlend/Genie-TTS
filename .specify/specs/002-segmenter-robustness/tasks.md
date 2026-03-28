# Tasks: Segmenter Robustness & Mixed CJK Disambiguation

**Spec ID:** 002

---

## US-1: Punctuation attached to the preceding segment

- [x] T1: Keep `segment_by_language()` punctuation-only chunks attached to the preceding segment
- [x] T2: Keep unit tests in `tests/Utils/test_lang_detector.py` for punctuation handling

## US-2: Configurable minimum segment length

- [x] T3: Keep `min_len` parameter on `segment_by_language()` (default=2)
- [x] T4: Keep unit tests for `min_len=1` and `min_len=2` behaviour

## US-3: Kana is a hard Japanese signal

- [x] T5: Add script-first heuristics so kana-containing chunks resolve to `Japanese`
- [x] T6: Add regression tests that force wrong detector output and verify kana still wins

## US-4: Chinese-first handling for mixed Chinese/Japanese text

- [x] T7: Keep Chinese as the default main context for mixed Chinese/Japanese text unless a stronger Japanese structural anchor appears
- [x] T8: Add regression tests for short kana inside Chinese sentences and embedded Japanese chunks inside Chinese sentences

## US-5: Auto routing stays stable after segmentation

- [x] T9: Add auto-dispatch tests for mixed Chinese/Japanese chunk routing order
- [ ] T10: Run the full `pytest tests/` suite and confirm no regressions
