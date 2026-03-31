"""Language detection and mixed-language segmentation utilities.

Uses fast_langdetect for per-segment language detection.
Supported output languages: Japanese, English, Chinese, Korean.
Unknown or low-confidence detections fall back to English with a warning.
"""
from __future__ import annotations

import logging
import re
from typing import TypedDict

logger = logging.getLogger(__name__)

# Mapping from fast_langdetect ISO 639-1 codes -> Genie canonical names
_FASTLANG_MAP: dict[str, str] = {
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "en": "English",
    # Additional codes fast_langdetect may return
    "zh-cn": "Chinese",
    "zh-tw": "Chinese",
}

_FALLBACK_LANG = "English"
_DEFAULT_MIN_SEGMENT_LEN = 2  # default minimum chars to keep a segment separate

# Punctuation-only pattern: CJK punctuation + ASCII punctuation + whitespace.
# Chunks matching this are attached to the preceding segment rather than
# being sent to detect_language() (which cannot infer language from punctuation).
_RE_PUNCT_ONLY = re.compile(
    r"^["
    r"\s"                          # whitespace
    r"\u3000-\u303f"               # CJK symbols & punctuation (。「」、・…)
    r"\uff00-\uffef"               # Fullwidth & halfwidth forms
    r"\u2000-\u206f"               # General punctuation
    r"!-/:-@\[-`{-~"              # ASCII punctuation (printable non-alnum)
    r"]+$"
)


def _split_edge_punctuation(part: str) -> tuple[str, str, str]:
    """Split *part* into (leading_punct, core_text, trailing_punct).

    This prevents mixed chunks like 'new things。' from being misdetected as
    Chinese just because they end with a CJK full-stop.
    """
    start = 0
    end = len(part)
    while start < end and _RE_PUNCT_ONLY.match(part[start]):
        start += 1
    while end > start and _RE_PUNCT_ONLY.match(part[end - 1]):
        end -= 1
    return part[:start], part[start:end], part[end:]


class LangSegment(TypedDict):
    language: str
    content: str


def _load_detector():
    """Lazy-load fast_langdetect to avoid startup cost."""
    try:
        from fast_langdetect import detect
        return detect
    except ImportError:
        logger.warning(
            "fast_langdetect not installed. "
            "Install it with: pip install fast_langdetect. "
            "Falling back to English for auto-detection."
        )
        return None


_detect_fn = None


def detect_language(text: str) -> str:
    """Detect the dominant language of *text* and return a canonical language name.

    Returns one of: 'Japanese', 'English', 'Chinese', 'Korean'.
    Falls back to 'English' with a warning if detection fails or language is unsupported.
    """
    global _detect_fn
    if _detect_fn is None:
        _detect_fn = _load_detector()

    if not text or not text.strip():
        return _FALLBACK_LANG

    if _detect_fn is None:
        return _FALLBACK_LANG

    try:
        result = _detect_fn(text)
        # fast_langdetect returns a list of dicts: [{'lang': 'zh', 'score': 0.99}, ...]
        if isinstance(result, list):
            code = result[0].get("lang", "").lower() if result else ""
        else:
            code = result.get("lang", "").lower()
        canonical = _FASTLANG_MAP.get(code)
        if canonical is None:
            logger.warning(
                "Auto language detection: unsupported language code '%s' for text %r. "
                "Falling back to %s.",
                code, text[:40], _FALLBACK_LANG,
            )
            return _FALLBACK_LANG
        return canonical
    except Exception as exc:
        logger.warning(
            "Auto language detection failed for text %r: %s. Falling back to %s.",
            text[:40], exc, _FALLBACK_LANG,
        )
        return _FALLBACK_LANG


# Regex: split into script runs so Han, kana, hangul, latin, punctuation can be
# reasoned about separately instead of collapsing all CJK into one chunk.
_RE_CJK_SPLIT = re.compile(
    r"([\u3040-\u30ff]+|[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+|[\uac00-\ud7a3\u3130-\u318f]+)"
)

_RE_HIRAGANA_KATAKANA = re.compile(r"[\u3040-\u30ff]")
_RE_HANGUL = re.compile(r"[\uac00-\ud7a3\u3130-\u318f]")
_RE_HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_RE_LATIN = re.compile(r"[A-Za-z]")
_RE_STRONG_BOUNDARY = re.compile(r"[。！？!?\n]")


def _classify_script_run(part: str) -> str:
    if _RE_HIRAGANA_KATAKANA.search(part):
        return "kana"
    if _RE_HANGUL.search(part):
        return "hangul"
    if _RE_HAN.search(part):
        return "han"
    if _RE_LATIN.search(part):
        return "latin"
    return "other"


def _score_clause_main_context(parts: list[str]) -> tuple[int, int]:
    runs = [part for part in parts if part and not _RE_PUNCT_ONLY.match(part)]
    if not runs:
        return 0, 0

    scripts = [_classify_script_run(part) for part in runs]
    zh_score = 0
    jp_score = 0

    han_runs = [part for part, script in zip(runs, scripts) if script == "han"]
    kana_runs = [part for part, script in zip(runs, scripts) if script == "kana"]

    zh_score += len(han_runs)
    if scripts[0] == "han":
        zh_score += 2
    if scripts[-1] == "han":
        zh_score += 2

    if kana_runs:
        jp_score += sum(len(run.strip()) for run in kana_runs)
        jp_score += sum(2 for run in kana_runs if len(run.strip()) >= 2)
        if scripts[0] == "kana":
            jp_score += 3
        if scripts[-1] == "kana":
            jp_score += 2
        if len(kana_runs) >= 2:
            jp_score += 2

    for idx in range(1, len(runs) - 1):
        if scripts[idx - 1] == "han" and scripts[idx] == "kana" and scripts[idx + 1] == "han":
            if len(runs[idx].strip()) == 1:
                zh_score += 3
            else:
                zh_score += 2
                jp_score += 1

    for idx in range(1, len(runs) - 1):
        if scripts[idx - 1] == "kana" and scripts[idx] == "han" and scripts[idx + 1] == "kana":
            jp_score += 3

    return zh_score, jp_score


def _detect_clause_main_context(parts: list[str]) -> str | None:
    zh_score, jp_score = _score_clause_main_context(parts)
    if zh_score >= jp_score + 2:
        return "Chinese"
    if jp_score >= zh_score + 2:
        return "Japanese"
    return None


def _build_clause_contexts(parts: list[str]) -> list[str | None]:
    contexts: list[str | None] = [None] * len(parts)
    start = 0
    for idx, part in enumerate(parts):
        if not part:
            continue
        if _RE_PUNCT_ONLY.match(part) and _RE_STRONG_BOUNDARY.search(part):
            clause_parts = parts[start:idx]
            context = _detect_clause_main_context(clause_parts)
            for fill_idx in range(start, idx + 1):
                contexts[fill_idx] = context
            start = idx + 1
    if start < len(parts):
        context = _detect_clause_main_context(parts[start:])
        for fill_idx in range(start, len(parts)):
            contexts[fill_idx] = context
    return contexts


def _detect_segment_language(
    core_part: str,
    prev_language: str | None = None,
    next_part: str | None = None,
    prev_content: str | None = None,
    clause_context: str | None = None,
) -> str:
    """Detect a segment language with script-first heuristics for mixed CJK text."""
    if _RE_HIRAGANA_KATAKANA.search(core_part):
        if clause_context == "Chinese" and len(core_part.strip()) < 2 and prev_language == "Chinese" and next_part and _RE_HAN.search(next_part):
            return "Chinese"
        return "Japanese"
    if _RE_HANGUL.search(core_part):
        return "Korean"
    if _RE_LATIN.search(core_part) and not _RE_HAN.search(core_part):
        return detect_language(core_part)

    han_chars = [char for char in core_part if _RE_HAN.match(char)]
    if han_chars and len(han_chars) == len(core_part.strip()):
        if prev_language == "Japanese":
            if prev_content and _RE_HIRAGANA_KATAKANA.search(prev_content):
                prev_kana_count = len(_RE_HIRAGANA_KATAKANA.findall(prev_content))
                if prev_kana_count >= 2 and len(han_chars) >= 3:
                    return "Chinese"
                return "Japanese"
            if next_part and _RE_HIRAGANA_KATAKANA.search(next_part):
                return "Japanese"
        if clause_context == "Japanese":
            return "Japanese"
        if clause_context == "Chinese":
            return "Chinese"
        if prev_language == "Chinese":
            return "Chinese"
        return "Chinese"

    return detect_language(core_part)


def segment_by_language(text: str, min_len: int = _DEFAULT_MIN_SEGMENT_LEN) -> list[LangSegment]:
    """Split *text* into segments, each labelled with a detected language.

    Strategy:
    1. Pre-split on CJK/non-CJK script boundaries.
    2. Punctuation-only chunks are attached to the preceding segment (not detected).
    3. Chunks shorter than *min_len* are attached to the preceding segment.
    4. Detect language for each remaining chunk using fast_langdetect.
    5. Merge adjacent same-language chunks.
    6. Drop empty segments.

    Args:
        text: Input text to segment.
        min_len: Minimum stripped length of a chunk before it gets its own language
            label. Chunks shorter than this are merged with the previous segment.
            Default is 2. Use 1 to keep single-character CJK words separate.

    Returns:
        A list of LangSegment dicts: {language: str, content: str}.
    """
    if not text or not text.strip():
        return []

    # Split into alternating CJK / non-CJK runs, keeping delimiters
    parts = _RE_CJK_SPLIT.split(text)
    clause_contexts = _build_clause_contexts(parts)

    raw: list[LangSegment] = []
    for idx, part in enumerate(parts):
        if not part:
            continue
        stripped = part.strip()
        # Attach punctuation-only chunks to the previous segment
        if _RE_PUNCT_ONLY.match(part):
            if raw:
                raw[-1]["content"] += part
            else:
                # Leading punctuation: defer — will be prepended when next segment appears
                raw.append({"language": _FALLBACK_LANG, "content": part})
            continue

        leading_punct, core_part, trailing_punct = _split_edge_punctuation(part)
        core_stripped = core_part.strip()
        if not core_stripped:
            if raw:
                raw[-1]["content"] += part
            else:
                raw.append({"language": _FALLBACK_LANG, "content": part})
            continue
        if len(core_stripped) < min_len:
            # Too short to detect reliably — attach to previous if possible
            if raw:
                raw[-1]["content"] += part
            else:
                raw.append({"language": _FALLBACK_LANG, "content": part})
            continue

        next_nonempty_part = None
        for future_part in parts[idx + 1:]:
            if future_part and not _RE_PUNCT_ONLY.match(future_part):
                next_nonempty_part = future_part
                break

        lang = _detect_segment_language(
            core_part,
            raw[-1]["language"] if raw else None,
            next_nonempty_part,
            raw[-1]["content"] if raw else None,
            clause_contexts[idx],
        )
        content = f"{leading_punct}{core_part}{trailing_punct}"
        raw.append({"language": lang, "content": content})

    if not raw:
        return [{"language": detect_language(text), "content": text}]

    # Merge adjacent same-language segments
    merged: list[LangSegment] = [raw[0]]
    for seg in raw[1:]:
        if seg["language"] == merged[-1]["language"]:
            merged[-1]["content"] += seg["content"]
        else:
            merged.append(seg)

    # Filter out empty segments
    return [s for s in merged if s["content"].strip()]
