# Plan: Segmenter Robustness & Mixed CJK Disambiguation

**Spec ID:** 002

---

## T1 — Restore spec coverage on `features/multilang`

Bring back the multilang-related spec files so the branch once again carries its own behavioral contract.

## T2 — Keep punctuation and `min_len` guarantees intact

Preserve the existing punctuation-attachment and configurable minimum-segment-length behavior in `segment_by_language()`.
This work must not regress the previously fixed punctuation leak or short-chunk controls.

## T3 — Add mixed Chinese/Japanese disambiguation heuristics

Extend `src/genie_tts/Utils/LangDetector.py` with a Chinese-first, script-first decision layer:
- Hiragana / katakana => Japanese, except very short kana embedded inside a Chinese sentence
- Hangul => Korean
- Latin-only chunks continue through the existing detector path
- Han-heavy chunks default to the Chinese main context unless a stronger Japanese structural anchor is present

## T4 — Lock the behavior with regression tests

Expand `tests/Utils/test_lang_detector.py` with regression cases for:
- kana-containing Japanese text surviving wrong detector output
- Chinese-dominant context surviving wrong detector output on shared Han chunks
- punctuation + mixed CJK boundaries

## T5 — Verify auto-mode dispatch

Expand `tests/test_get_phones_and_bert.py` so mixed Chinese/Japanese chunk outputs from the segmenter are routed to the expected backend order and preserve concatenation shapes.

## T6 — Align tasks/spec state with implementation

Update `tasks.md` so the branch clearly records the mixed Chinese/Japanese segmentation work alongside the earlier punctuation/min_len improvements.
