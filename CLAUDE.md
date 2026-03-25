# Genie-TTS — Claude Code Guidance

## Project Overview

Genie-TTS is a lightweight ONNX inference engine built on GPT-SoVITS, targeting CPU performance. It
supports TTS inference, ONNX model conversion, a FastAPI HTTP server, and a PySide6 GUI.

**Package:** `genie_tts` (src layout under `src/`)
**Python:** >= 3.10
**Supported languages:** Japanese, English, Chinese (Mandarin), Korean, Auto-detect

---

## Repository Layout

```
src/genie_tts/
  __init__.py          # Public API
  Internal.py          # User-facing function implementations
  GetPhonesAndBert.py  # Language dispatch: text -> phones + BERT features
  ModelManager.py      # ONNX session management
  Server.py            # FastAPI server
  Utils/
    Language.py        # Language code normalization
    LangDetector.py    # Automatic language detection & segmentation (new)
    TextSplitter.py    # Sentence splitting
    Shared.py          # Shared context singleton
    Constants.py       # BERT_FEATURE_DIM etc.
  G2P/
    SymbolsV2.py       # Unified phoneme symbol set (symbol_to_id_v2)
    Chinese/           # Mandarin G2P pipeline
    English/           # ARPAbet G2P pipeline
    Japanese/          # Open JTalk G2P
    Korean/            # g2pk2 + jamo G2P
  Core/
    Inference.py       # GENIE class: T2S + VITS
    Resources.py       # Path constants, HF download
    TTSPlayer.py       # Threaded TTS player
  Audio/
    Audio.py           # Audio utilities
    ReferenceAudio.py  # Reference audio + SSL/HuBERT extraction
  Converter/           # PyTorch -> ONNX converters (v2, v2ProPlus)
  Data/                # Bundled ONNX model stubs
.specify/              # Spec-kit specs and templates
tests/                 # Pytest unit tests
```

---

## Development Conventions

### Branch Naming
Follow spec-kit convention: `NNN-short-description` (e.g., `001-spec-kit-multilang`).

### Commits
Use Conventional Commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`.
Commit frequently after each logical unit of work.

### Testing
- Framework: **pytest** (run `pytest tests/`)
- Unit tests live in `tests/` mirroring the `src/genie_tts/` structure
- Never mock G2P internals; test with real inputs and verify phone ID sequences or strings
- Run tests before every commit

### Language Pipeline
The canonical language identifiers (after `normalize_language()`) are:

| Canonical Name           | Description                                      |
|--------------------------|--------------------------------------------------|
| `Japanese`               | pyopenjtalk-plus G2P, no BERT                    |
| `English`                | g2p_en ARPAbet, no BERT                          |
| `Chinese`                | pypinyin + g2pM + jieba, RoBERTa BERT            |
| `Korean`                 | g2pk2 + jamo, no BERT                            |
| `Hybrid-Chinese-English` | regex split by Latin chars, per-chunk G2P        |
| `auto`                   | fast_langdetect detection + segment-level dispatch|

Adding a new language requires:
1. G2P module in `src/genie_tts/G2P/<Language>/`
2. Symbol set entries in `SymbolsV2.py`
3. Language code mapping in `Utils/Language.py`
4. Dispatch branch in `GetPhonesAndBert.py`
5. Tests in `tests/G2P/`

### Spec-Kit Workflow
- Specs live in `.specify/specs/NNN-feature-name/`
- Constitution: `.specify/memory/constitution.md`
- Keep specs up to date with implementation; update when behaviour changes
- `tasks.md` in each spec tracks implementation status

### Dependencies
Add new dependencies to `pyproject.toml` `[project.dependencies]`.
Do not use `requirements.txt` for runtime deps (it mirrors pyproject for compatibility only).

### Rules
- Do not use PyTorch at inference time — ONNX only
- Do not break the public API in `__init__.py` without a version bump
- Cantonese symbols already exist in `SymbolsV2.py`; add G2P before enabling
- `auto` language mode must never silently fall back without logging a warning

