"""Benchmark: intra_op_num_threads scaling for Genie-TTS CPU inference.

Tests thread counts from 1 to physical core count and measures per-sentence
latency to find the optimal thread configuration.

Usage:
    python benchmark_threads.py
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

# Sentences that exercise both short and long sequences
SENTENCES = [
    ("thirtyseven", "thirtyseven",
     os.path.join(MODELS_DIR, "thirtyseven", "prompt_wav", "En_play_hero3066_mainvoc_6.wav"),
     "And now, I belong to this set.",
     "English",
     [
         "The quick brown fox jumps over the lazy dog.",
         "Hello world, this is a thread scaling benchmark test.",
         "Speech synthesis performance depends heavily on CPU thread configuration.",
     ]),
    ("mika", "mika",
     os.path.join(MODELS_DIR, "mika", "prompt_wav", "917575.wav"),
     "私も昔、これと似たようなの持ってたなぁ…。",
     "auto",
     [
         "おはようございます、今日はいい天気ですね。",
         "こんにちは、I love Tokyo!",
     ]),
]

THREAD_COUNTS = [1, 2, 4, 6, 8, 10]
def run_benchmark():
    results = []  # {char, threads, sentence, elapsed}

    for n_threads in THREAD_COUNTS:
        print(f"\n{'='*60}")
        print(f"  intra_op_num_threads = {n_threads}")
        print(f"{'='*60}")

        runtime_config = {
            "providers": ["CPUExecutionProvider"],
            "intra_op_num_threads": n_threads,
            "inter_op_num_threads": 1,
        }

        for char_name, model_subdir, prompt_wav, prompt_text, language, sentences in SENTENCES:
            char_tag = f"{char_name}_t{n_threads}"
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

            print(f"  [{char_name}]")
            for i, text in enumerate(sentences):
                out_path = os.path.join(OUT_DIR, f"{char_tag}_{i+1:02d}.wav")
                t0 = time.perf_counter()
                try:
                    genie.tts(character_name=char_tag, text=text, save_path=out_path, play=False)
                    elapsed = time.perf_counter() - t0
                    print(f"    [{i+1}] {elapsed:.3f}s  {text[:45]!r}")
                    results.append({"char": char_name, "threads": n_threads, "sentence": text, "elapsed": elapsed})
                except Exception as e:
                    print(f"    [{i+1}] ERROR: {e}")
                    results.append({"char": char_name, "threads": n_threads, "sentence": text, "elapsed": None})

            genie.unload_character(char_tag)

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  SUMMARY — avg latency per sentence (seconds)")
    print(f"{'='*60}")

    # Collect avg per (char, threads)
    from itertools import groupby
    key = lambda r: (r["char"], r["threads"])
    agg = {}
    for (char, threads), group in groupby(sorted(results, key=key), key=key):
        timings = [r["elapsed"] for r in group if r["elapsed"] is not None]
        agg[(char, threads)] = sum(timings) / len(timings) if timings else None

    chars = sorted({r["char"] for r in results})
    header = f"{'threads':>8}" + "".join(f"  {c:>14}" for c in chars)
    print(header)
    print("-" * len(header))
    for n in THREAD_COUNTS:
        row = f"{n:>8}"
        for c in chars:
            val = agg.get((c, n))
            row += f"  {val:>13.3f}s" if val is not None else f"  {'N/A':>14}"
        print(row)

    # Speedup vs single thread
    print(f"\n  Speedup vs 1 thread:")
    print(f"{'threads':>8}" + "".join(f"  {c:>14}" for c in chars))
    print("-" * len(header))
    for n in THREAD_COUNTS:
        row = f"{n:>8}"
        for c in chars:
            base = agg.get((c, 1))
            val = agg.get((c, n))
            if base and val:
                row += f"  {base/val:>13.2f}x"
            else:
                row += f"  {'N/A':>14}"
        print(row)
    print()


if __name__ == "__main__":
    run_benchmark()

