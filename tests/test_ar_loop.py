"""Regression tests for AR loop off-by-one fix and truncation warning (spec 004)."""
from __future__ import annotations

import logging
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def _make_stage_decoder(stop_at_call: int, total_calls: int, token_val: int = 7):
    """Return a mock stage_decoder that emits stop_condition_tensor=True at stop_at_call.

    Each call appends one new column to y so the slice math is realistic.
    """
    call_count = [0]
    # Start y with a prompt of 4 tokens (shape 1×4)
    y_state = [np.zeros((1, 4), dtype=np.int64)]

    def run(output_names, input_feed):
        call_count[0] += 1
        new_col = np.array([[token_val]], dtype=np.int64)
        y_state[0] = np.concatenate([y_state[0], new_col], axis=1)
        y = y_state[0]
        y_emb = np.zeros((1, y.shape[1], 512), dtype=np.float32)
        stop = np.array(call_count[0] >= stop_at_call)
        pkv = [np.zeros((1, 1), dtype=np.float32)]
        return [y, y_emb, stop, *pkv]

    mock = MagicMock()
    mock.run.side_effect = run
    mock.get_inputs.return_value = [
        MagicMock(name="y"), MagicMock(name="y_emb"), MagicMock(name="pkv")
    ]
    return mock, call_count


def _make_encoder_and_first_stage():
    """Minimal mocks for encoder and first_stage_decoder."""
    encoder = MagicMock()
    encoder.run.return_value = [
        np.zeros((1, 8, 512), dtype=np.float32),   # x
        np.zeros((1, 4, 512), dtype=np.float32),   # prompts
    ]
    first_stage = MagicMock()
    y0 = np.zeros((1, 4), dtype=np.int64)
    y_emb0 = np.zeros((1, 4, 512), dtype=np.float32)
    pkv0 = np.zeros((1, 1), dtype=np.float32)
    first_stage.run.return_value = [y0, y_emb0, pkv0]
    return encoder, first_stage


def _run_t2s(stage_decoder):
    """Wire up GENIE.t2s_cpu with minimal inputs."""
    from genie_tts.Core.Inference import GENIE
    genie = GENIE()
    encoder, first_stage = _make_encoder_and_first_stage()
    return genie.t2s_cpu(
        ref_seq=np.zeros((1, 4), dtype=np.int64),
        ref_bert=np.zeros((1, 4, 1024), dtype=np.float32),
        text_seq=np.zeros((1, 6), dtype=np.int64),
        text_bert=np.zeros((1, 6, 1024), dtype=np.float32),
        ssl_content=np.zeros((1, 8, 768), dtype=np.float32),
        encoder=encoder,
        first_stage_decoder=first_stage,
        stage_decoder=stage_decoder,
    )


class TestARLoopOffByOne:

    def test_stop_at_idx_1_returns_exactly_one_token(self):
        """AC-4.1: stopping at idx=1 must return exactly 1 generated token, not all of y."""
        stage_decoder, _ = _make_stage_decoder(stop_at_call=1, total_calls=1)
        result = _run_t2s(stage_decoder)
        assert result is not None
        assert result.shape[2] == 1, (
            f"Expected 1 token, got {result.shape[2]}. "
            "off-by-one: y[:, -1:] returned wrong slice."
        )

    def test_stop_at_idx_3_returns_exactly_three_tokens(self):
        """Stopping at idx=3 must return exactly 3 generated tokens."""
        stage_decoder, _ = _make_stage_decoder(stop_at_call=3, total_calls=3)
        result = _run_t2s(stage_decoder)
        assert result is not None
        assert result.shape[2] == 3

    def test_result_does_not_include_prompt_tokens(self):
        """The returned slice must never be as long as prompt+generated combined."""
        stage_decoder, _ = _make_stage_decoder(stop_at_call=2, total_calls=2)
        result = _run_t2s(stage_decoder)
        assert result is not None
        # prompt is 4 tokens; only 2 were generated — result must be < 4
        assert result.shape[2] < 4


class TestARLoopTruncationWarning:

    def test_warning_emitted_when_loop_exhausts(self, caplog):
        """AC-4.2: exhausting MAX_T2S_LEN iterations emits a WARNING."""
        from genie_tts.Core.Inference import MAX_T2S_LEN
        # Make stop never fire — run exactly MAX_T2S_LEN calls
        stage_decoder, _ = _make_stage_decoder(
            stop_at_call=MAX_T2S_LEN + 1,  # never stops within the loop
            total_calls=MAX_T2S_LEN,
        )
        with caplog.at_level(logging.WARNING, logger="genie_tts.Core.Inference"):
            _run_t2s(stage_decoder)
        assert any(
            "exhausted" in rec.message and "MAX_T2S_LEN" in rec.message
            for rec in caplog.records
        ), "Expected truncation warning was not emitted."

    def test_no_warning_when_stop_token_received(self, caplog):
        """No warning when the loop terminates normally via stop token."""
        stage_decoder, _ = _make_stage_decoder(stop_at_call=2, total_calls=2)
        with caplog.at_level(logging.WARNING, logger="genie_tts.Core.Inference"):
            _run_t2s(stage_decoder)
        assert not any(
            "exhausted" in rec.message
            for rec in caplog.records
        )
