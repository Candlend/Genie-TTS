# Spec: Segmenter Robustness & Mixed CJK Disambiguation

**Spec ID:** 002
**Status:** In Progress
**Branch:** features/multilang
**Created:** 2026-03-25

---

## Problem Statement

After shipping auto-detection (spec 001), the segmenter still has a high-risk weakness around mixed Chinese/Japanese text.

The current `segment_by_language()` flow pre-splits on CJK/non-CJK boundaries, attaches punctuation, merges short chunks, and then relies on `fast_langdetect` for chunk labels. This is good enough for many Chinese/English cases, but it breaks down for mixed Chinese/Japanese text because both languages share Han characters.

Three robustness gaps matter most:

1. **Detector overreach on kana-containing Japanese** — A chunk that clearly contains hiragana/katakana can still be mislabeled if the detector output is wrong, even though kana should be a hard Japanese signal.

2. **Shared Han chunks can drift without structural anchors** — Mixed Chinese/Japanese text often contains Han-only spans that are ambiguous by themselves. The segmenter needs a structural rule for how such spans inherit the surrounding language, without relying on hardcoded character lists.

3. **Spec/tests do not lock the behavior** — There is not yet a written contract for how `auto` should resolve mixed Chinese/Japanese boundaries, so regressions are easy to reintroduce.

This spec extends the segmenter contract so `auto` remains stable on punctuation, short chunks, and mixed Chinese/Japanese text without changing the existing Hybrid-Chinese-English flow. In mixed Chinese/Japanese text, the guiding policy is **Chinese-first with local Japanese extraction**: keep the Chinese main context stable, and only cut out Japanese chunks when there is strong structural evidence such as kana runs.

---

## User Stories

### US-1: Punctuation attached to the preceding segment
As a developer using `language="auto"`,
I want punctuation characters (CJK and ASCII) to attach to the preceding language segment
rather than forming their own segment labelled 'English',
so that G2P processes punctuation together with the text it belongs to.

**Acceptance criteria:**
- `segment_by_language("你好，world")` does not produce a standalone punctuation segment
- Punctuation between same-language segments merges into that language segment
- Punctuation between different-language segments attaches to the preceding segment
- Unit tests in `tests/Utils/test_lang_detector.py` cover these cases

### US-2: Configurable minimum segment length
As a developer,
I want to pass `min_len` to `segment_by_language()` to override the default minimum segment size,
so that single-character CJK words are not silently merged into the previous segment.

**Acceptance criteria:**
- `segment_by_language(text, min_len=1)` respects segments of length 1
- Default behaviour (`min_len=2`) is unchanged unless mixed CJK heuristics explicitly override label choice
- Parameter is documented in the function docstring
- Unit tests verify both `min_len=1` and `min_len=2` behaviour

### US-3: Kana is a hard Japanese signal
As a developer using `language="auto"`,
I want chunks containing hiragana or katakana to resolve to Japanese even if the detector is wrong,
so that obvious Japanese text is not routed through Chinese G2P.

**Acceptance criteria:**
- A chunk containing kana is labeled `Japanese`
- This rule takes precedence over `fast_langdetect` output
- Regression tests cover detector outputs that incorrectly return `zh`

### US-4: Chinese-first handling for mixed Chinese/Japanese text
As a developer using `language="auto"`,
I want Chinese-dominant text to stay Chinese by default while clearly Japanese chunks are extracted locally,
so that short kana and embedded Japanese words do not flip the whole sentence.

**Acceptance criteria:**
- Short kana inside a Chinese sentence does not flip the full sentence to `Japanese`
- A clearly Japanese chunk inside a Chinese sentence can still be extracted locally as `Japanese`
- Han-only chunks follow structural context and default to the Chinese main context unless there is a stronger Japanese structural anchor
- Regression tests use structural signals only and do not depend on hardcoded character tables

### US-5: Auto routing stays stable after segmentation
As a maintainer,
I want `GetPhonesAndBert` auto mode to route each mixed Chinese/Japanese chunk to the expected G2P backend,
so that segmentation fixes actually change the downstream phoneme path.

**Acceptance criteria:**
- Auto mode routes Chinese chunks to Chinese G2P and Japanese chunks to Japanese G2P in the original order
- Concatenated `phones` / `bert` shapes remain correct
- Dispatch tests use mocks/stubs and do not require real model data

---

## Out of Scope
- Dictionary-grade semantic disambiguation for every Han-only Chinese/Japanese word
- Changes to `Hybrid-Chinese-English`
- Cantonese G2P implementation
- GPU inference
- Multi-speaker caching
