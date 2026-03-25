"""Multi-language pipeline test script.

Demonstrates how Genie-TTS handles mixed-language text with:
  1. Single-language mode
  2. Hybrid-Chinese-English mode (legacy regex split)
  3. Auto mode (fast_langdetect-based segmentation)

Runs WITHOUT loading inference models — only tests the text->phones pipeline.
"""
import os
import sys
import logging

os.environ["GENIE_SKIP_RESOURCE_CHECK"] = "1"
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from genie_tts.Utils.Language import normalize_language
from genie_tts.Utils.LangDetector import detect_language, segment_by_language
from genie_tts.Utils.TextSplitter import TextSplitter


SEP = "=" * 60


def show_segments(text: str):
    """Show how segment_by_language splits a text."""
    segs = segment_by_language(text)
    print(f"  Input : {text!r}")
    for i, s in enumerate(segs):
        print(f"  Seg {i} : [{s['language']:8s}] {s['content']!r}")
    return segs


def show_detect(text: str):
    lang = detect_language(text)
    print(f"  detect_language({text!r}) -> {lang!r}")
    return lang


def section(title: str):
    print(f"\n{SEP}\n  {title}\n{SEP}")


# -----------------------------------------------------------------------
# 1. Language normalisation
# -----------------------------------------------------------------------
section("1. normalize_language() — alias resolution")
cases = [
    ("ja", "Japanese"), ("jpn", "Japanese"),
    ("zh", "Chinese"),  ("cmn", "Chinese"), ("zho", "Chinese"),
    ("en", "English"),  ("eng", "English"),
    ("ko", "Korean"),   ("kor", "Korean"),
    ("hybrid", "Hybrid-Chinese-English"),
    ("auto", "auto"),
    ("yue", "Cantonese"),
]
for code, expected in cases:
    try:
        result = normalize_language(code)
        status = "OK" if result == expected else f"MISMATCH (expected {expected})"
    except ValueError as e:
        result, status = "ValueError", str(e)[:60]
    print(f"  {code:25s} -> {result:25s}  [{status}]")

try:
    normalize_language("xx")
except ValueError as e:
    print(f"  {'xx':25s} -> ValueError: {str(e)[:60]}...")


# -----------------------------------------------------------------------
# 2. Auto language detection (single text)
# -----------------------------------------------------------------------
section("2. detect_language() — single-text detection")
texts = [
    "Hello, how are you?",
    "你好，今天天气怎么样？",
    "おはようございます、今日はいい天気ですね。",
    "안녕하세요, 오늘도 좋은 하루 되세요.",
    "Bonjour le monde",       # unsupported -> fallback
    "こんにちは、I love Tokyo!",  # mixed
]
for t in texts:
    show_detect(t)


# -----------------------------------------------------------------------
# 3. Mixed-language segmentation
# -----------------------------------------------------------------------
section("3. segment_by_language() — mixed-language splitting")
mixed_cases = [
    # Chinese + English
    "我今天去了school，学了很多new things。",
    # Japanese + English
    "こんにちは、I am studying Japanese every day!",
    # Chinese + Japanese (no Latin)
    "你好、こんにちは、我叫田中。",
    # Korean + English
    "안녕하세요, my name is 김철수.",
    # Pure Chinese
    "今天天气非常好，我们去公园散步吧。",
    # Pure English
    "The quick brown fox jumps over the lazy dog.",
]
for text in mixed_cases:
    print()
    show_segments(text)


# -----------------------------------------------------------------------
# 4. Legacy Hybrid-Chinese-English split (regex)
# -----------------------------------------------------------------------
section("4. Legacy Hybrid-Chinese-English (regex Latin split)")
from genie_tts.GetPhonesAndBert import _split_chinese_english

hybrid_cases = [
    "我今天去了school，学了很多new things。",
    "Hello 你好 world 世界",
    "纯中文没有英文",
    "Pure English no Chinese",
]
for text in hybrid_cases:
    parts = _split_chinese_english(text)
    print(f"  Input : {text!r}")
    for p in parts:
        print(f"    [{p['language']:8s}] {p['content']!r}")
    print()


# -----------------------------------------------------------------------
# 5. TextSplitter with multi-language content
# -----------------------------------------------------------------------
section("5. TextSplitter — sentence splitting (max_len=20)")
splitter = TextSplitter(max_len=20, min_len=4)
long_texts = [
    "今天天气很好，我们去公园散步吧。公园里有很多花，非常漂亮！",
    "Hello world, this is a test. Another sentence here! And one more.",
    "안녕하세요. 오늘도 좋은 하루 되세요. 날씨가 참 좋네요!",
    "おはようございます。今日もいい天気ですね。頑張りましょう！",
]
for text in long_texts:
    sentences = splitter.split(text)
    print(f"  Input : {text[:50]!r}{'...' if len(text)>50 else ''}")
    for i, s in enumerate(sentences):
        print(f"    [{i}] {s!r}")
    print()


print(f"\n{SEP}")
print("  All tests complete.")
print(SEP)
