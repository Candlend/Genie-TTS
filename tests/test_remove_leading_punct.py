"""Regression tests for removal of unconditional leading 。 prepend (spec 008)."""
from __future__ import annotations

import numpy as np
from unittest.mock import patch, MagicMock


def _make_tts_mocks():
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
    y_state = [np.zeros((1, 5), dtype=np.int64)]
    def _stage_run(_, feed):
        y_state[0] = np.concatenate([y_state[0], np.array([[1]], dtype=np.int64)], axis=1)
        return [y_state[0], np.zeros((1, y_state[0].shape[1], 512), dtype=np.float32),
                np.array(True), np.zeros((1, 1), dtype=np.float32)]
    stage_decoder = MagicMock()
    stage_decoder.run.side_effect = _stage_run
    stage_decoder.get_inputs.return_value = [MagicMock(name=n) for n in ("y", "y_emb", "pkv")]
    vocoder = MagicMock()
    vocoder.run.return_value = [np.zeros((1, 100), dtype=np.float32)]
    return prompt_audio, encoder, first_stage, stage_decoder, vocoder


class TestRemoveLeadingPunct:

    def test_get_phones_not_called_with_leading_kuten(self):
        """tts() must NOT prepend 。 to the input before calling get_phones_and_bert."""
        from genie_tts.Core.Inference import GENIE

        captured = []
        prompt_audio, encoder, first_stage, stage_decoder, vocoder = _make_tts_mocks()

        with patch('genie_tts.Core.Inference.get_phones_and_bert',
                   side_effect=lambda text, **kw: captured.append(text) or
                   (np.zeros((1, 6), dtype=np.int64),
                    np.zeros((6, 1024), dtype=np.float32))):
            genie = GENIE()
            genie.tts(
                text='こんにちは',
                prompt_audio=prompt_audio,
                encoder=encoder,
                first_stage_decoder=first_stage,
                stage_decoder=stage_decoder,
                vocoder=vocoder,
                prompt_encoder=None,
                language='Japanese',
            )

        assert captured, "get_phones_and_bert was never called"
        called_text = captured[0]
        assert not called_text.startswith('。'), (
            f"Leading 。 must not be prepended; got: {called_text!r}"
        )

    def test_trailing_punct_still_appended(self):
        """The trailing punct guard must still be applied."""
        from genie_tts.Core.Inference import GENIE

        captured = []
        prompt_audio, encoder, first_stage, stage_decoder, vocoder = _make_tts_mocks()

        with patch('genie_tts.Core.Inference.get_phones_and_bert',
                   side_effect=lambda text, **kw: captured.append(text) or
                   (np.zeros((1, 6), dtype=np.int64),
                    np.zeros((6, 1024), dtype=np.float32))):
            genie = GENIE()
            genie.tts(
                text='こんにちは',
                prompt_audio=prompt_audio,
                encoder=encoder,
                first_stage_decoder=first_stage,
                stage_decoder=stage_decoder,
                vocoder=vocoder,
                prompt_encoder=None,
                language='Japanese',
            )

        assert captured
        assert captured[0].endswith('。'), (
            f"Trailing 。 must still be appended; got: {captured[0]!r}"
        )

    def test_text_already_starting_with_punct_not_doubled(self):
        """Text that already begins with 。 must not get another 。 prepended."""
        from genie_tts.Core.Inference import GENIE

        captured = []
        prompt_audio, encoder, first_stage, stage_decoder, vocoder = _make_tts_mocks()

        with patch('genie_tts.Core.Inference.get_phones_and_bert',
                   side_effect=lambda text, **kw: captured.append(text) or
                   (np.zeros((1, 6), dtype=np.int64),
                    np.zeros((6, 1024), dtype=np.float32))):
            genie = GENIE()
            genie.tts(
                text='。こんにちは',
                prompt_audio=prompt_audio,
                encoder=encoder,
                first_stage_decoder=first_stage,
                stage_decoder=stage_decoder,
                vocoder=vocoder,
                prompt_encoder=None,
                language='Japanese',
            )

        assert captured
        assert not captured[0].startswith('。。'), (
            f"Double leading 。 must not occur; got: {captured[0]!r}"
        )
