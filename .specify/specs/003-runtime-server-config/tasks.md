# Tasks: Runtime Configurability & Server Scaling

**Spec ID:** 003

---

## US-1: Configurable runtime selection

- [x] T1: Add `runtime_config` support to `load_character()` and `/load_character`
- [x] T2: Pass `providers` and `provider_options` into ONNX Runtime session creation

## US-2: Configurable runtime threading

- [x] T3: Apply `intra_op_num_threads` and `inter_op_num_threads` to session options
- [ ] T4: Extend helper usage to auxiliary model sessions where applicable

## US-3: Explicit server scaling modes

- [x] T5: Extend `start_server()` with `scaling_mode`, `max_concurrency`, and `queue_maxsize`
- [x] T6: Persist effective server runtime settings in module-level state

## US-4: Single-process queue protection

- [x] T7: Add bounded single-process request accounting and 429 rejection
- [x] T8: Release accepted single-process slots when streaming responses finish

## US-5: Docs and regression coverage

- [x] T9: Add regression tests for runtime config plumbing and queue guards
- [ ] T10: Update README / README_zh with runtime and scaling configuration examples
