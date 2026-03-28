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

# 🔮 GENIE: [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 轻量级推理引擎

**在 CPU 上体验近乎即时的语音合成**

[简体中文](./README_zh.md) | [English](./README.md)

</div>

---

**GENIE** 是一个基于开源 TTS 项目 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 构建的轻量级推理引擎。它集成了 TTS
推理、ONNX 模型转换、API 服务端以及其他核心功能，旨在提供极致的性能和便利性。

* **✅ 支持的模型版本：** GPT-SoVITS V2, V2ProPlus
* **✅ 支持的语言：** 日语、英语、中文、韩语、自动检测（`language="auto"`）
* **✅ 支持的 Python 版本：** >= 3.10

---

## 🎬 演示视频

- **[➡️ 观看演示视频（中文）](https://www.bilibili.com/video/BV1d2hHzJEz9)**

---

## 🚀 性能优势

GENIE 针对原始模型进行了优化，以实现出色的 CPU 性能。

| 特性         |  🔮 GENIE   | 官方 PyTorch 模型 | 官方 ONNX 模型 |
|:-----------|:-----------:|:-------------:|:----------:|
| **首次推理延迟** |  **1.13s**  |     1.35s     |   3.57s    |
| **运行时大小**  | **\~200MB** |    \~数 GB     | 与 GENIE 相似 |
| **模型大小**   | **\~230MB** |  与 GENIE 相似   |  \~750MB   |

> 📝 **延迟测试说明：** 所有延迟数据均基于 100 个日语句子（每句约 20 个字符）的测试集取平均值。测试环境为 CPU i7-13620H。

---

## 🏁 快速开始

> **⚠️ 重要提示：** 建议在 **管理员模式** 下运行 GENIE，以避免潜在的性能下降。

### 📦 安装

通过 pip 安装：

```bash
pip install genie-tts
```

## 📥 预训练模型

首次运行 GENIE 时，需要下载资源文件（**~391MB**）。您可以按照库的提示自动下载。

> 或者，您可以从 [HuggingFace](https://huggingface.co/High-Logic/Genie/tree/main/GenieData) 手动下载文件并将其放置在本地文件夹中。然后在导入库
**之前** 设置 `GENIE_DATA_DIR` 环境变量：

```python
import os

# 设置手动下载的资源文件路径
# 注意：请在导入 genie_tts 之前执行此操作
os.environ["GENIE_DATA_DIR"] = r"C:\path\to\your\GenieData"

import genie_tts as genie

# 库现在将从指定目录加载资源
```

### ⚡️ 快速试用

还没有 GPT-SoVITS 模型？没问题！
GENIE 包含几个预定义的说话人角色，您可以立即使用 —— 例如：

* **Mika (聖園ミカ)** — *蔚蓝档案 (Blue Archive)* (日语)
* **ThirtySeven (37)** — *重返未来：1999 (Reverse: 1999)* (英语)
* **Feibi (菲比)** — *鸣潮 (Wuthering Waves)* (中文)

您可以在此处浏览所有可用角色：
**[https://huggingface.co/High-Logic/Genie/tree/main/CharacterModels](
https://huggingface.co/High-Logic/Genie/tree/main/CharacterModels)**

使用以下示例进行尝试：

```python
import genie_tts as genie
import time

# 首次运行时自动下载所需文件
genie.load_predefined_character('mika')

genie.tts(
    character_name='mika',
    text='どうしようかな……やっぱりやりたいかも……！',
    play=True,  # 直接播放生成的音频
)

genie.wait_for_playback_done()  # 确保音频播放完成
```

### 🎤 TTS 最佳实践

一个简单的 TTS 推理示例：

```python
import genie_tts as genie

# 第一步：加载角色语音模型
genie.load_character(
    character_name='<CHARACTER_NAME>',  # 替换为您的角色名称
    onnx_model_dir=r"<PATH_TO_CHARACTER_ONNX_MODEL_DIR>",  # 包含 ONNX 模型的文件夹
    language='<LANGUAGE_CODE>',  # 替换为语言代码，例如 'en', 'zh', 'jp'
    runtime_config={
        "providers": ["CPUExecutionProvider"],
        # Apple Silicon 10 核实测：4 线程约有 3x 加速；超过 4 线程后调度开销反而使性能下降。
        # 建议设置为物理核心数的一半到全部之间，按实际机器调整。
        "intra_op_num_threads": 4,
        "inter_op_num_threads": 1,
        # 也可以通过环境变量设置默认值，这样代码接口可以保持不变：
        # GENIE_ORT_PROVIDERS=CPUExecutionProvider
        # GENIE_ORT_INTRA_OP_NUM_THREADS=4
        # GENIE_ORT_INTER_OP_NUM_THREADS=1
        # GENIE_ORT_EXECUTION_MODE=ORT_SEQUENTIAL
        # 显式传入的 runtime_config 优先级高于环境变量。
        # CUDA 构建示例（Linux/Windows，需要 CUDA 版 onnxruntime）：
        # "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        # "provider_options": {"CUDAExecutionProvider": {"device_id": "0"}},
        # Apple Silicon Mac（CoreML，走 GPU / 神经引擎）：
        # "providers": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        # 注意：不推荐对 TTS 使用 CoreML。T2S 自回归解码器每句要调用 ONNX session
        # 数百次，CoreML 每次 GPU kernel 启动开销较高，实测比 CPU 慢 4-8 倍。
    },
)

# 第二步：设置参考音频（用于情感和语调克隆）
genie.set_reference_audio(
    character_name='<CHARACTER_NAME>',  # 必须与加载的角色名称匹配
    audio_path=r"<PATH_TO_REFERENCE_AUDIO>",  # 参考音频的路径
    audio_text="<REFERENCE_AUDIO_TEXT>",  # 对应的文本
)

# 第三步：运行 TTS 推理并生成音频
genie.tts(
    character_name='<CHARACTER_NAME>',  # 必须与加载的角色匹配
    text="<TEXT_TO_SYNTHESIZE>",  # 要合成的文本
    play=True,  # 直接播放音频
    save_path="<OUTPUT_AUDIO_PATH>",  # 输出音频文件路径
)

genie.wait_for_playback_done()  # 确保音频播放完成

print("🎉 Audio generation complete!")
```

---

## 🔧 模型转换

要将原始 GPT-SoVITS 模型转换为 GENIE 格式，请确保已安装 `torch`：

```bash
pip install torch
```

使用内置的转换工具：

> **提示：** `convert_to_onnx` 目前支持 V2 和 V2ProPlus 模型。

```python
import genie_tts as genie

genie.convert_to_onnx(
    torch_pth_path=r"<YOUR .PTH MODEL FILE>",  # 替换为您的 .pth 文件
    torch_ckpt_path=r"<YOUR .CKPT CHECKPOINT FILE>",  # 替换为您的 .ckpt 文件
    output_dir=r"<ONNX MODEL OUTPUT DIRECTORY>"  # 保存 ONNX 模型的目录
)
```

---

## 🌐 启动 FastAPI 服务

GENIE 包含一个轻量级的 FastAPI 服务器：

```python
import genie_tts as genie

# 启动服务
genie.start_server(
    host="0.0.0.0",  # 主机地址
    port=8000,  # 端口
    workers=4,  # 基于多进程扩展的工作进程数
)

# 单进程 + 有界排队模式（workers=1 时自动启用）
# genie.start_server(
#     host="0.0.0.0",
#     port=8000,
#     workers=1,
#     max_concurrency=1,
#     queue_maxsize=8,
# )
```

`start_server()` 根据 `workers` 自动选择部署模式：

- `workers > 1`（多进程）：uvicorn fork 出 N 个 worker 进程，提升并发吞吐。
  每个进程各自持有独立的模型缓存和参考音频缓存。
- `workers=1`（单进程）：自动启用进程内有界请求控制。
  通过 `max_concurrency` 和 `queue_maxsize` 限制同时处理和排队的请求数。
  超出 `queue_maxsize` 的请求返回 HTTP 429。

也可以保持 Python 调用不变，通过环境变量配置默认值：

```bash
export GENIE_SERVER_WORKERS=4
export GENIE_SERVER_MAX_CONCURRENCY=1
export GENIE_SERVER_QUEUE_MAXSIZE=8
```

显式传入的 `start_server(...)` 参数优先级高于环境变量。

> 关于请求格式和 API 详情，请参阅我们的 [API 服务教程](./Tutorial/English/API%20Server%20Tutorial.py)。


---

## 📝 路线图

* [x] **🌐 语言扩展**

    * [x] 添加对 **中文** 和 **英文** 的支持。

* [x] **🚀 模型兼容性**

    * [x] 支持 `V2Proplus`。
    * [ ] 支持 `V3`、`V4` 等更多版本。

* [x] **📦 简易部署**

    * [ ] 发布 **官方 Docker 镜像**。
    * [x] 提供开箱即用的 **Windows 整合包**。

---