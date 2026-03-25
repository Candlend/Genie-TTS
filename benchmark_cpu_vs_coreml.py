"""Benchmark: CPU vs CoreML (Apple Silicon GPU/ANE) for Genie-TTS inference.

Runs each test sentence through both providers and prints timing summary.
Outputs wav files to ./benchmark_output/

Usage:
    python benchmark_cpu_vs_coreml.py
"""
import os
import sys
import time
import logging

os.environ["GENIE_DATA_DIR"] = os.path.join(os.path.dirname(__file__), "GenieData")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import genie_tts as genie

MODELS_DIR = os.path.join(os.path.dirname(__file__), "CharacterModels", "v2ProPlus")
OUT_DIR = os.path.join(os.path.dirname(__file__), "benchmark_output")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Test cases: (character, model_subdir, prompt_wav, prompt_text, language, sentences)
# ---------------------------------------------------------------------------
TEST_CASES = [
    (
        "thirtyseven",
        "thirtyseven",
        os.path.join(MODELS_DIR, "thirtyseven", "prompt_wav", "En_play_hero3066_mainvoc_6.wav"),
        "And now, I belong to this set.",
        "English",
        [
            "The quick brown fox jumps over the lazy dog.",
            "Hello world, this is a CoreML benchmark test.",
            "Speech synthesis performance comparison on Apple Silicon.",
        ],
    ),
    (
        "mika",
        "mika",
        os.path.join(MODELS_DIR, "mika", "prompt_wav", "917575.wav"),
        "私も昔、これと似たようなの持ってたなぁ…。",
        "auto",
        [
            "おはようございます、今日はいい天気ですね。",
            "こんにちは、I love Tokyo!",
        ],
    ),
]
PROVIDERS = {
    "CPU": {
        "providers": ["CPUExecutionProvider"],
    },
    "CoreML": {
        "providers": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
    },
}


def run_benchmark():
    results = []  # list of {character, sentence, provider, elapsed}

    for provider_label, runtime_config in PROVIDERS.items():
        print(f"\n{'='*60}")
        print(f"  Provider: {provider_label}")
        print(f"{'='*60}")

        for char_name, model_subdir, prompt_wav, prompt_text, language, sentences in TEST_CASES:
            char_tag = f"{char_name}_{provider_label}"
            print(f"  Loading {char_name} ...")

            genie.load_character(
                character_name=char_tag,
                onnx_model_dir=os.path.join(MODELS_DIR, model_subdir, "tts_models"),
                language=language,
                runtime_config=runtime_config,
            )
            genie.set_reference_audio(
                character_name=char_tag,
                audio_path=prompt_wav,
                audio_text=prompt_text,
            )

            for i, text in enumerate(sentences):
                out_path = os.path.join(OUT_DIR, f"{char_tag}_{i+1:02d}.wav")
                t0 = time.perf_counter()
                try:
                    genie.tts(
                        character_name=char_tag,
                        text=text,
                        save_path=out_path,
                        play=False,
                    )
                    elapsed = time.perf_counter() - t0
                    size = os.path.getsize(out_path) if os.path.exists(out_path) else 0
                    print(f"    [{i+1}] {elapsed:.3f}s  {text[:40]!r}  -> {size//1024}KB")
                    results.append({
                        "character": char_name,
                        "provider": provider_label,
                        "sentence": text,
                        "elapsed": elapsed,
                    })
                except Exception as e:
                    print(f"    [{i+1}] ERROR: {e}")
                    results.append({
                        "character": char_name,
                        "provider": provider_label,
                        "sentence": text,
                        "elapsed": None,
                    })

            genie.unload_character(char_tag)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"{'Character':<14} {'Provider':<10} {'Sentences':>9} {'Total(s)':>10} {'Avg(s)':>8}")
    print("-" * 60)

    from itertools import groupby
    key = lambda r: (r["character"], r["provider"])
    for (char, prov), group in groupby(sorted(results, key=key), key=key):
        timings = [r["elapsed"] for r in group if r["elapsed"] is not None]
        n = len(timings)
        total = sum(timings)
        avg = total / n if n else float("nan")
        print(f"{char:<14} {prov:<10} {n:>9} {total:>10.3f} {avg:>8.3f}")

    print()
    # CPU vs CoreML comparison per character
    by_char = {}
    for r in results:
        by_char.setdefault(r["character"], {}).setdefault(r["provider"], [])
        if r["elapsed"] is not None:
            by_char[r["character"]][r["provider"]].append(r["elapsed"])

    print("  Speedup (CPU avg / CoreML avg):")
    for char, provs in sorted(by_char.items()):
        cpu_avg = sum(provs.get("CPU", [])) / len(provs["CPU"]) if provs.get("CPU") else None
        cml_avg = sum(provs.get("CoreML", [])) / len(provs["CoreML"]) if provs.get("CoreML") else None
        if cpu_avg and cml_avg:
            speedup = cpu_avg / cml_avg
            print(f"    {char:<14}  CPU={cpu_avg:.3f}s  CoreML={cml_avg:.3f}s  speedup={speedup:.2f}x")
        else:
            print(f"    {char:<14}  (incomplete data)")

    print(f"\nOutput files in: {OUT_DIR}/")


if __name__ == "__main__":
    run_benchmark()

