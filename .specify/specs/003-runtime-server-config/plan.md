# Plan: Runtime Configurability & Server Scaling

**Spec ID:** 003

---

## T1 — RuntimeConfig plumbing

Introduce a reusable runtime configuration structure in `ModelManager.py` and thread it through the
public Python API and the HTTP `/load_character` endpoint.

## T2 — Session creation wiring

Centralize session creation inputs so providers, provider options, and session options are built from
`runtime_config` and applied consistently to character model sessions.

## T3 — Server scaling configuration

Extend `start_server()` with explicit scaling controls (`scaling_mode`, `max_concurrency`,
`queue_maxsize`) and persist the effective values in module-level server runtime state.

## T4 — Single-process queue protection

Add request accounting helpers for single-process mode. Reject excess requests with 429 when the queue
limit is exceeded, and ensure successful streamed requests release their slot on stream completion.

## T5 — Docs and regression coverage

Document the new runtime/server options in README files and add regression tests covering provider
options, thread settings, single-process limits, and slot release behavior.
