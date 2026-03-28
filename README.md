<div align="center">
<pre>
██████╗  ███████╗███╗   ██╗██╗███████╗
██╔════╝ ██╔════╝████╗  ██║██║██╔════╝
██║  ███╗█████╗  ██╔██╗ ██║██║█████╗  
██║   ██║██╔══╝  ██║╚██╗██║██║██╔══╝  
╚██████╔╝███████╗██║ ╚████║██║███████╗
 ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚═╝╚══════╝
</pre>
</div>

<div align="center">

# 🔮 GENIE: [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) Lightweight Inference Engine

**Experience near-instantaneous speech synthesis on your CPU**

[简体中文](./README_zh.md) | [English](./README.md)

</div>

---

**GENIE** is a lightweight inference engine built on the open-source TTS
project [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS). It integrates TTS inference, ONNX model conversion, API
server, and other core features, aiming to provide ultimate performance and convenience.

* **✅ Supported Model Version:** GPT-SoVITS V2, V2ProPlus
* **✅ Supported Language:** Japanese, English, Chinese, Korean, Auto-detect (`language="auto"`)
* **✅ Supported Python Version:** >= 3.10

---

## 🎬 Demo Video

- **[➡️ Watch the demo video (Chinese)](https://www.bilibili.com/video/BV1d2hHzJEz9)**

---

## 🚀 Performance Advantages

GENIE optimizes the original model for outstanding CPU performance.

| Feature                     |  🔮 GENIE   | Official PyTorch Model | Official ONNX Model |
|:----------------------------|:-----------:|:----------------------:|:-------------------:|
| **First Inference Latency** |  **1.13s**  |         1.35s          |        3.57s        |
| **Runtime Size**            | **\~200MB** |      \~several GB      |  Similar to GENIE   |
| **Model Size**              | **\~230MB** |    Similar to GENIE    |       \~750MB       |

> 📝 **Latency Test Info:** All latency data is based on a test set of 100 Japanese sentences (\~20 characters each),
> averaged. Tested on CPU i7-13620H.

---

## 🏁 QuickStart

> **⚠️ Important:** It is recommended to run GENIE in **Administrator mode** to avoid potential performance degradation.

### 📦 Installation

Install via pip:

```bash
pip install genie-tts
```

## 📥 Pretrained Models

When running GENIE for the first time, it requires downloading resource files (**~391MB**). You can follow the library's
prompts to download them automatically.

> Alternatively, you can manually download the files
> from [HuggingFace](https://huggingface.co/High-Logic/Genie/tree/main/GenieData)
> and place them in a local folder. Then set the `GENIE_DATA_DIR` environment variable **before** importing the library:

```python
import os

# Set the path to your manually downloaded resource files
# Note: Do this BEFORE importing genie_tts
os.environ["GENIE_DATA_DIR"] = r"C:\path\to\your\GenieData"

import genie_tts as genie

# The library will now load resources from the specified directory
```

### ⚡️ Quick Tryout

No GPT-SoVITS model yet? No problem!
GENIE includes several predefined speaker characters you can use immediately —
for example:

* **Mika (聖園ミカ)** — *Blue Archive* (Japanese)
* **ThirtySeven (37)** — *Reverse: 1999* (English)
* **Feibi (菲比)** — *Wuthering Waves* (Chinese)

You can browse all available characters here:
**[https://huggingface.co/High-Logic/Genie/tree/main/CharacterModels](
https://huggingface.co/High-Logic/Genie/tree/main/CharacterModels)**

Try it out with the example below:

```python
import genie_tts as genie
import time

# Automatically downloads required files on first run
genie.load_predefined_character('mika')

genie.tts(
    character_name='mika',
    text='どうしようかな……やっぱりやりたいかも……！',
    play=True,  # Play the generated audio directly
)

genie.wait_for_playback_done()  # Ensure audio playback completes
```

### 🎤 TTS Best Practices

A simple TTS inference example:

```python
import genie_tts as genie

# Step 1: Load character voice model
genie.load_character(
    character_name='<CHARACTER_NAME>',  # Replace with your character name
    onnx_model_dir=r"<PATH_TO_CHARACTER_ONNX_MODEL_DIR>",  # Folder containing ONNX model
    language='<LANGUAGE_CODE>',  # Replace with language code, e.g., 'en', 'zh', 'jp'
    runtime_config={
        "providers": ["CPUExecutionProvider"],
        # Optimal thread count on Apple Silicon (10-core): 4 threads gives ~3x speedup
        # vs single-threaded; beyond 4 threads, scheduling overhead reduces gains.
        # Tune this to your physical core count — typically floor(cores / 2) to cores.
        "intra_op_num_threads": 4,
        "inter_op_num_threads": 1,
        # You can also set defaults via environment variables and keep your code unchanged:
        # GENIE_ORT_PROVIDERS=CPUExecutionProvider
        # GENIE_ORT_INTRA_OP_NUM_THREADS=4
        # GENIE_ORT_INTER_OP_NUM_THREADS=1
        # GENIE_ORT_EXECUTION_MODE=ORT_SEQUENTIAL
        # Explicit runtime_config values override environment variables.
        # Example for CUDA (Linux/Windows with CUDA-enabled onnxruntime):
        # "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        # "provider_options": {"CUDAExecutionProvider": {"device_id": "0"}},
        # Example for Apple Silicon Mac (CoreML — GPU / Neural Engine):
        # "providers": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        # Note: CoreML is NOT recommended for TTS — the autoregressive T2S decoder
        # calls the ONNX session hundreds of times per sentence, and CoreML's
        # per-call GPU kernel overhead makes it 4-8x slower than CPU for this workload.
    },
)

# Step 2: Set reference audio (for emotion and intonation cloning)
genie.set_reference_audio(
    character_name='<CHARACTER_NAME>',  # Must match loaded character name
    audio_path=r"<PATH_TO_REFERENCE_AUDIO>",  # Path to reference audio
    audio_text="<REFERENCE_AUDIO_TEXT>",  # Corresponding text
)

# Step 3: Run TTS inference and generate audio
genie.tts(
    character_name='<CHARACTER_NAME>',  # Must match loaded character
    text="<TEXT_TO_SYNTHESIZE>",  # Text to synthesize
    play=True,  # Play audio directly
    save_path="<OUTPUT_AUDIO_PATH>",  # Output audio file path
)

genie.wait_for_playback_done()  # Ensure audio playback completes

print("🎉 Audio generation complete!")
```

---

## 🔧 Model Conversion

To convert original GPT-SoVITS models for GENIE, ensure `torch` is installed:

```bash
pip install torch
```

Use the built-in conversion tool:

> **Tip:** `convert_to_onnx` currently supports V2 and V2ProPlus models.

```python
import genie_tts as genie

genie.convert_to_onnx(
    torch_pth_path=r"<YOUR .PTH MODEL FILE>",  # Replace with your .pth file
    torch_ckpt_path=r"<YOUR .CKPT CHECKPOINT FILE>",  # Replace with your .ckpt file
    output_dir=r"<ONNX MODEL OUTPUT DIRECTORY>"  # Directory to save ONNX model
)
```

---

## 🌐 Launch FastAPI Server

GENIE includes a lightweight FastAPI server:

```python
import genie_tts as genie

# Start server
genie.start_server(
    host="0.0.0.0",  # Host address
    port=8000,  # Port
    workers=4,  # Number of workers for process-based scaling
)

# Single-process mode with bounded queueing (workers=1)
# genie.start_server(
#     host="0.0.0.0",
#     port=8000,
#     workers=1,
#     max_concurrency=1,
#     queue_maxsize=8,
# )
```

`start_server()` supports two deployment modes, selected automatically by `workers`:

- `workers > 1` (multi-process): uvicorn forks N worker processes for higher throughput.
  Each process keeps its own model and reference-audio cache.
- `workers=1` (single-process): bounded in-process request accounting is enabled automatically.
  Use `max_concurrency` and `queue_maxsize` to limit concurrent and queued requests.
  Requests beyond `queue_maxsize` are rejected with HTTP 429.

You can also keep the Python call unchanged and configure defaults via environment variables:

```bash
export GENIE_SERVER_WORKERS=4
export GENIE_SERVER_MAX_CONCURRENCY=1
export GENIE_SERVER_QUEUE_MAXSIZE=8
```

Explicit `start_server(...)` arguments override environment variables.

> For request formats and API details, see our [API Server Tutorial](./Tutorial/English/API%20Server%20Tutorial.py).


---

## 📝 Roadmap

* [x] **🌐 Language Expansion**

    * [x] Add support for **Chinese** and **English**.

* [x] **🚀 Model Compatibility**

    * [x] Support for `V2Proplus`.
    * [ ] Support for `V3`, `V4`, and more.

* [x] **📦 Easy Deployment**

    * [ ] Release **Official Docker images**.
    * [x] Provide out-of-the-box **Windows bundles**.

---
