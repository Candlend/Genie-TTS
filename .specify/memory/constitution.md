# Genie-TTS Constitution

## Core Principles

### I. ONNX-Only Inference
All inference (T2S autoregressive decode, VITS vocoder, BERT, SSL/HuBERT) must use ONNX Runtime.
No PyTorch imports at runtime. PyTorch is allowed only in `Converter/` for model export.

### II. Lightweight by Default
Minimise runtime dependencies. Prefer pure-Python or ONNX-based solutions over large ML frameworks.
Target CPU performance — no GPU assumption. Memory footprint matters.

### III. Language-First Design
Every text-processing concern (G2P, normalisation, segmentation) is per-language.
New language support follows the established module pattern:
`G2P/<Language>/`, `SymbolsV2.py` entries, `Language.py` mapping, `GetPhonesAndBert.py` dispatch.
Mixed/auto language modes are first-class, not afterthoughts.

### IV. Test-Driven Quality
All new features and bug fixes must be accompanied by pytest unit tests in `tests/`.
Tests must pass before merging. G2P modules are tested with representative sentences per language.
No mocking of G2P internals — test real outputs.

### V. Stable Public API
The public surface (`src/genie_tts/__init__.py`) is stable.
Breaking changes require a semver MAJOR/MINOR bump and changelog entry.
Internal modules may evolve freely.

### VI. Spec-Driven Development
Non-trivial features start with a spec in `.specify/specs/NNN-feature-name/`.
Specs are kept in sync with implementation. Outdated specs must be updated, not ignored.
Constitution supersedes all other practices.

### VII. Conventional Commits & Frequent Commits
All commits use Conventional Commits format. Commit after each logical unit of work.
Branch names follow `NNN-short-description` (spec-kit sequential convention).

## Governance
This constitution supersedes all other practices and preferences.
Amendments require updating this file with rationale and updating `Last Amended`.

**Version**: 1.0.0 | **Ratified**: 2026-03-25 | **Last Amended**: 2026-03-25
