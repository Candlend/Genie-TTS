"""TTS inference test — generates wav files for mixed-language sentences.

Language is set per test case via set_reference_audio().
Outputs wav files to ./test_output/
"""
import os
import sys
import logging

os.environ["GENIE_DATA_DIR"] = os.path.join(os.path.dirname(__file__), "GenieData")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import genie_tts as genie

OUT_DIR = os.path.join(os.path.dirname(__file__), "test_output")
os.makedirs(OUT_DIR, exist_ok=True)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "CharacterModels", "v2ProPlus")


def test_char(name, model_subdir, prompt_wav, prompt_text, default_language, test_cases):
    print(f"\n{'='*60}")
    print(f"  Character: {name} | default_lang={default_language}")
    print(f"{'='*60}")

    genie.load_character(
        character_name=name,
        onnx_model_dir=os.path.join(MODELS_DIR, model_subdir, "tts_models"),
        language=default_language,
    )

    genie.set_reference_audio(
        character_name=name,
        audio_path=prompt_wav,
        audio_text=prompt_text,
    )

    for i, text in enumerate(test_cases):
        out_path = os.path.join(OUT_DIR, f"{name}_{i+1:02d}.wav")
        print(f"  [{i+1}] lang={default_language!r:25s} text={text!r}")

        try:
            genie.tts(
                character_name=name,
                text=text,
                save_path=out_path,
                play=False,
            )
            size = os.path.getsize(out_path) if os.path.exists(out_path) else 0
            print(f"       -> {os.path.basename(out_path)} ({size/1024:.0f} KB)")
        except Exception as e:
            print(f"       -> ERROR: {e}")

    genie.unload_character(name)


# -----------------------------------------------------------------------
# 1. feibi (Chinese model loaded as auto)
# -----------------------------------------------------------------------
feibi_wav = os.path.join(MODELS_DIR, "feibi", "prompt_wav", "zh_vo_Main_Linaxita_2_1_10_26.wav")
test_char(
    name="feibi",
    model_subdir="feibi",
    prompt_wav=feibi_wav,
    prompt_text="在此之前，请您务必继续享受旅居拉古那的时光。",
    default_language="auto",
    test_cases=[
        "今天天气非常好，我们去公园散步吧。",          # pure Chinese
        "我今天去了school，学了很多new things。",      # Chinese+English
        "你好，I love 北京！",                      # short mixed
        "你好，こんにちは，I love music!",           # Chinese+Japanese+English
    ],
)

# -----------------------------------------------------------------------
# 2. thirtyseven (English model loaded as English)
# -----------------------------------------------------------------------
thirtyseven_wav = os.path.join(MODELS_DIR, "thirtyseven", "prompt_wav", "En_play_hero3066_fightingvoc_19.wav")
test_char(
    name="thirtyseven",
    model_subdir="thirtyseven",
    prompt_wav=thirtyseven_wav,
    prompt_text="And now, I belong to this set.",
    default_language="English",
    test_cases=[
        "The quick brown fox jumps over the lazy dog.",
        "Hello world, auto language detection test.",
    ],
)

# -----------------------------------------------------------------------
# 3. mika (Japanese model loaded as auto)
# -----------------------------------------------------------------------
mika_wav = os.path.join(MODELS_DIR, "mika", "prompt_wav", "917575.wav")
test_char(
    name="mika",
    model_subdir="mika",
    prompt_wav=mika_wav,
    prompt_text="私も昔、これと似たようなの持ってたなぁ…。",
    default_language="auto",
    test_cases=[
        "おはようございます、今日はいい天気ですね。",
        "こんにちは、I love Tokyo!",
        "你好，こんにちは，I love music!",
    ],
)

print(f"\n\nDone. Output files in: {OUT_DIR}/")
for f in sorted(os.listdir(OUT_DIR)):
    path = os.path.join(OUT_DIR, f)
    print(f"  {f}  ({os.path.getsize(path)/1024:.0f} KB)")
