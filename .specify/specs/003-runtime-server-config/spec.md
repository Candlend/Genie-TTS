# Spec: Runtime Configurability & Server Scaling

**Spec ID:** 003
**Status:** In Progress
**Branch:** 003-runtime-server-config
**Created:** 2026-03-26

---

## Problem Statement

Genie-TTS currently hardcodes ONNX Runtime execution to CPU, does not expose provider or threading
controls through the public APIs, and relies on shared mutable server state in the `/tts` path. This
makes it hard to tune CPU usage, enable GPU execution in controlled environments, or scale the server
safely when multiple requests arrive concurrently.

---

## User Stories

### US-1: Configurable runtime selection
As a developer integrating Genie-TTS,
I want to choose ONNX Runtime providers and provider-specific options when loading a character,
so that I can run inference on CPU or GPU with explicit configuration.

**Acceptance criteria:**
- `load_character()` accepts an optional `runtime_config`
- `/load_character` accepts an optional `runtime_config`
- `runtime_config.providers` is passed to ONNX Runtime session creation
- `runtime_config.provider_options` is passed through to ONNX Runtime session creation
- Default behaviour remains CPU-only when config is omitted

### US-2: Configurable runtime threading
As a developer,
I want to configure ONNX Runtime thread counts,
so that I can tune CPU utilization for my deployment target.

**Acceptance criteria:**
- `runtime_config.intra_op_num_threads` is applied to session options
- `runtime_config.inter_op_num_threads` is applied to session options
- Existing default behaviour is preserved when thread counts are omitted

### US-3: Explicit server scaling modes
As a deployer,
I want `start_server()` to distinguish between multi-process scaling and single-process queueing,
so that I can pick a safe scaling strategy for my environment.

**Acceptance criteria:**
- `start_server()` accepts `scaling_mode`
- `scaling_mode="single-process"` rejects `workers > 1`
- `start_server()` accepts `max_concurrency` and `queue_maxsize`
- Server runtime state stores the effective scaling configuration

### US-4: Single-process queue protection
As a deployer using single-process mode,
I want the server to reject excess TTS requests once the queue is full,
so that shared in-process state does not grow without bound.

**Acceptance criteria:**
- single-process mode tracks active and waiting requests
- excess requests return HTTP 429 when queue limit is reached
- accepted streaming requests release their slot when the stream finishes
- unit tests cover queue rejection and slot release behaviour

---

## Out of Scope
- Full multi-GPU scheduling
- Automatic provider fallback from invalid explicit configs
- Cross-process shared character/reference caches
- Refactoring `tts_player` into a fully parallel request engine
