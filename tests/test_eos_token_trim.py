"""Regression tests for EOS token zeroing bug fix (spec 012)."""
from __future__ import annotations

import numpy as np
from unittest.mock import MagicMock, patch


def _make_stage_decoder_with_eos(stop_at_call: int = 2):
    """Stage decoder that fires stop at stop_at_call and appends EOS (1024) on stop."""
    call_count = [0]
    y_state = [np.zeros((1, 4), dtype=np.int64)]

    def run(output_names, input_feed):
        call_count[0] += 1
        token = 1024 if call_count[0] >= stop_at_call else 7
        new_col = np.array([[token]], dtype=np.int64)
        y_state[0] = np.concatenate([y_state[0], new_col], axis=1)
        stop = np.array(call_count[0] >= stop_at_call)
        y_emb = np.zeros((1, y_state[0].shape[1], 512), dtype=np.float32)
        return [y_state[0], y_emb, stop, np.zeros((1, 1), dtype=np.float32)]

    mock = MagicMock()
    mock.run.side_effect = run
    mock.get_inputs.return_value = [
        MagicMock(name='y'), MagicMock(name='y_emb'), MagicMock(name='pkv')
    ]
    return mock


def _run_t2s(stage_decoder):
    from genie_tts.Core.Inference import GENIE
    genie = GENIE()
    encoder = MagicMock()
    encoder.run.return_value = [
        np.zeros((1, 8, 512), dtype=np.float32),
        np.zeros((1, 4, 512), dtype=np.float32),
    ]
    first_stage = MagicMock()
    first_stage.run.return_value = [
        np.zeros((1, 4), dtype=np.int64),
        np.zeros((1, 4, 512), dtype=np.float32),
        np.zeros((1, 1), dtype=np.float32),
    ]
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


class TestEOSTokenTrim:

    def test_t2s_output_does_not_contain_zero_from_eos_zeroing(self):
        """t2s_cpu must not zero out the EOS token to 0 before returning."""
        stage_decoder = _make_stage_decoder_with_eos(stop_at_call=2)
        result = _run_t2s(stage_decoder)
        assert result is not None
        # With the old bug: last token was zeroed to 0; with the fix: EOS (1024) is present
        # The EOS filter in tts() handles trimming — t2s_cpu should return raw tokens
        last_token = result[0, 0, -1]
        assert last_token != 0 or result.shape[2] == 0, (
            f"Last token was zeroed to 0, which defeats the EOS filter. Got: {result}"
        )

    def test_eos_filter_in_tts_trims_correctly(self):
        """tts() EOS filter must trim tokens >= 1024 from t2s_cpu output."""
        from genie_tts.Core.Inference import GENIE

        genie = GENIE()
        # Inject a t2s_cpu that returns tokens [7, 7, 1024] (EOS at index 2)
        fake_semantic = np.array([[[7, 7, 1024]]], dtype=np.int64)

        prompt_audio = MagicMock()
        prompt_audio.phonemes_seq = np.zeros((1, 4), dtype=np.int64)
        prompt_audio.text_bert = np.zeros((1, 4, 1024), dtype=np.float32)
        prompt_audio.ssl_content = np.zeros((1, 8, 768), dtype=np.float32)
        prompt_audio.audio_32k = np.zeros((1, 3200), dtype=np.float32)

        encoder = MagicMock()
        encoder.run.return_value = [
            np.zeros((1, 8, 512), dtype=np.float32),
            np.zeros((1, 4, 512), dtype=np.float32),
        ]
        first_stage = MagicMock()
        first_stage.run.return_value = [
            np.zeros((1, 4), dtype=np.int64),
            np.zeros((1, 4, 512), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
        ]
        stage_decoder = MagicMock()
        stage_decoder.get_inputs.return_value = [
            MagicMock(name='y'), MagicMock(name='y_emb'), MagicMock(name='pkv')
        ]
        vocoder = MagicMock()
        vocoder.run.return_value = [np.zeros((1, 100), dtype=np.float32)]

        with patch.object(genie, 't2s_cpu', return_value=fake_semantic), \
             patch('genie_tts.Core.Inference.get_phones_and_bert',
                   return_value=(np.zeros((1, 6), dtype=np.int64),
                                 np.zeros((6, 1024), dtype=np.float32))):
            genie.tts(
                text='テスト。',
                prompt_audio=prompt_audio,
                encoder=encoder,
                first_stage_decoder=first_stage,
                stage_decoder=stage_decoder,
                vocoder=vocoder,
                prompt_encoder=None,
                language='Japanese',
            )

        # Check vocoder was called with trimmed tokens (only [7, 7], not [7, 7, 1024])
        assert vocoder.run.called
        call_kwargs = vocoder.run.call_args[0][1]
        pred_semantic = call_kwargs['pred_semantic']
        assert pred_semantic.shape[-1] == 2, (
            f"Expected 2 tokens after EOS trim, got {pred_semantic.shape[-1]}"
        )
        assert not np.any(pred_semantic >= 1024), (
            "EOS token (>= 1024) must not reach the vocoder"
        )
