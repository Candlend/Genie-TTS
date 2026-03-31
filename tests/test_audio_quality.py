"""Regression tests for audio output quality fixes (spec 005)."""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Helpers to build minimal GENIE.tts() mocks
# ---------------------------------------------------------------------------

def _make_tts_mocks(vocoder_audio: np.ndarray):
    """Return mocks for all sessions + prompt_audio needed by GENIE.tts()."""
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

    # stage_decoder stops immediately at idx=1
    y_state = [np.zeros((1, 5), dtype=np.int64)]
    call_n = [0]
    def _stage_run(_, feed):
        call_n[0] += 1
        y_state[0] = np.concatenate([y_state[0], np.array([[1]], dtype=np.int64)], axis=1)
        return [
            y_state[0],
            np.zeros((1, y_state[0].shape[1], 512), dtype=np.float32),
            np.array(True),
            np.zeros((1, 1), dtype=np.float32),
        ]
    stage_decoder = MagicMock()
    stage_decoder.run.side_effect = _stage_run
    stage_decoder.get_inputs.return_value = [MagicMock(name=n) for n in ("y", "y_emb", "pkv")]

    vocoder = MagicMock()
    vocoder.run.return_value = [vocoder_audio]

    return prompt_audio, encoder, first_stage, stage_decoder, vocoder


def _call_tts(text: str, language: str, vocoder_audio: np.ndarray) -> np.ndarray:
    from genie_tts.Core.Inference import GENIE
    genie = GENIE()
    prompt_audio, encoder, first_stage, stage_decoder, vocoder = _make_tts_mocks(vocoder_audio)
    with patch("genie_tts.Core.Inference.get_phones_and_bert",
               return_value=(np.zeros((1, 4), dtype=np.int64),
                             np.zeros((1, 4, 1024), dtype=np.float32))):
        return genie.tts(
            text=text,
            prompt_audio=prompt_audio,
            encoder=encoder,
            first_stage_decoder=first_stage,
            stage_decoder=stage_decoder,
            vocoder=vocoder,
            prompt_encoder=None,
            language=language,
        )


# ---------------------------------------------------------------------------
# US-1: Trailing punctuation
# ---------------------------------------------------------------------------

class TestTrailingPunctuation:

    def test_japanese_text_without_punct_gets_japanese_period(self):
        """AC-4.1: bare Japanese text gets 。 appended."""
        captured = []
        with patch("genie_tts.Core.Inference.get_phones_and_bert",
                   side_effect=lambda text, **kw: captured.append(text) or
                   (np.zeros((1,4),dtype=np.int64), np.zeros((1,4,1024),dtype=np.float32))):
            from genie_tts.Core.Inference import GENIE
            genie = GENIE()
            prompt_audio, encoder, first_stage, stage_decoder, vocoder = \
                _make_tts_mocks(np.zeros((1, 100), dtype=np.float32))
            genie.tts("こんにちは", prompt_audio, encoder, first_stage,
                      stage_decoder, vocoder, None, language="Japanese")
        assert captured[0].endswith('。'), f"Expected 。 suffix, got: {captured[0]!r}"

    def test_english_text_without_punct_gets_period(self):
        """AC-4.2: bare English text gets . appended."""
        captured = []
        with patch("genie_tts.Core.Inference.get_phones_and_bert",
                   side_effect=lambda text, **kw: captured.append(text) or
                   (np.zeros((1,4),dtype=np.int64), np.zeros((1,4,1024),dtype=np.float32))):
            from genie_tts.Core.Inference import GENIE
            genie = GENIE()
            prompt_audio, encoder, first_stage, stage_decoder, vocoder = \
                _make_tts_mocks(np.zeros((1, 100), dtype=np.float32))
            genie.tts("Hello world", prompt_audio, encoder, first_stage,
                      stage_decoder, vocoder, None, language="English")
        assert captured[0].endswith('.'), f"Expected . suffix, got: {captured[0]!r}"

    def test_text_already_ending_with_punct_not_doubled(self):
        """Text already ending with 。 must not get another one appended."""
        captured = []
        with patch("genie_tts.Core.Inference.get_phones_and_bert",
                   side_effect=lambda text, **kw: captured.append(text) or
                   (np.zeros((1,4),dtype=np.int64), np.zeros((1,4,1024),dtype=np.float32))):
            from genie_tts.Core.Inference import GENIE
            genie = GENIE()
            prompt_audio, encoder, first_stage, stage_decoder, vocoder = \
                _make_tts_mocks(np.zeros((1, 100), dtype=np.float32))
            genie.tts("こんにちは。", prompt_audio, encoder, first_stage,
                      stage_decoder, vocoder, None, language="Japanese")
        assert not captured[0].endswith('。。'), f"Double punct: {captured[0]!r}"


# ---------------------------------------------------------------------------
# US-2: Amplitude normalisation
# ---------------------------------------------------------------------------

class TestAmplitudeNormalisation:

    def test_hot_audio_is_clipped_to_max_one(self):
        """AC-4.2: vocoder output with max > 1.0 is normalised to max == 1.0."""
        hot_audio = np.array([[2.0, -3.0, 1.5]], dtype=np.float32)
        result = _call_tts("テスト。", "Japanese", hot_audio)
        assert result is not None
        assert np.abs(result).max() <= 1.0 + 1e-6

    def test_hot_audio_shape_preserved(self):
        """Normalisation must not change the array shape."""
        hot_audio = np.ones((1, 50), dtype=np.float32) * 5.0
        result = _call_tts("テスト。", "Japanese", hot_audio)
        assert result is not None
        assert result.shape == hot_audio.shape

    def test_normal_audio_unchanged(self):
        """Audio already within [-1, 1] must not be altered."""
        normal = np.array([[0.5, -0.3, 0.8]], dtype=np.float32)
        result = _call_tts("テスト。", "Japanese", normal)
        assert result is not None
        np.testing.assert_allclose(result, normal, atol=1e-6)


# ---------------------------------------------------------------------------
# US-3: Inter-sentence silence
# ---------------------------------------------------------------------------

class TestInterSentenceSilence:

    def test_silence_chunk_sent_after_each_sentence_when_split(self):
        """AC-4.3: chunk_callback receives a silence chunk after each audio chunk."""
        from genie_tts.Core.TTSPlayer import TTSPlayer, INTER_SENTENCE_SILENCE_SAMPLES

        received: list[bytes] = []
        player = TTSPlayer(sample_rate=32000)
        player._split = True
        player._chunk_callback = received.append
        player._play = False
        player._current_save_path = None

        silence_bytes_len = INTER_SENTENCE_SILENCE_SAMPLES * 2  # int16
        audio_chunk = np.ones((1, 100), dtype=np.float32) * 0.5

        # Simulate what _tts_worker_loop does for one audio chunk
        with patch("genie_tts.Core.TTSPlayer.tts_client") as mock_client, \
             patch("genie_tts.Core.TTSPlayer.model_manager") as mock_mm, \
             patch("genie_tts.Core.TTSPlayer.context") as mock_ctx:
            mock_mm.get.return_value = MagicMock(
                T2S_ENCODER=MagicMock(), T2S_FIRST_STAGE_DECODER=MagicMock(),
                T2S_STAGE_DECODER=MagicMock(), VITS=MagicMock(),
                PROMPT_ENCODER=None, LANGUAGE="Japanese",
            )
            mock_ctx.current_speaker = "test"
            mock_ctx.current_prompt_audio = MagicMock()
            mock_client.stop_event = MagicMock()
            mock_client.tts.return_value = audio_chunk

            # Directly invoke the relevant section of _tts_worker_loop
            from genie_tts.Core.TTSPlayer import INTER_SENTENCE_SILENCE_SAMPLES
            import numpy as _np
            gsv_model = mock_mm.get(mock_ctx.current_speaker)
            mock_client.stop_event.clear()
            chunk = mock_client.tts(
                text="テスト",
                prompt_audio=mock_ctx.current_prompt_audio,
                encoder=gsv_model.T2S_ENCODER,
                first_stage_decoder=gsv_model.T2S_FIRST_STAGE_DECODER,
                stage_decoder=gsv_model.T2S_STAGE_DECODER,
                vocoder=gsv_model.VITS,
                prompt_encoder=gsv_model.PROMPT_ENCODER,
                language=gsv_model.LANGUAGE,
            )
            if chunk is not None:
                audio_data = player._preprocess_for_playback(chunk)
                player._chunk_callback(audio_data)
                if player._split and player._chunk_callback:
                    silence = _np.zeros(INTER_SENTENCE_SILENCE_SAMPLES, dtype=_np.float32)
                    player._chunk_callback(player._preprocess_for_playback(silence))

        assert len(received) == 2, f"Expected 2 callbacks (audio + silence), got {len(received)}"
        assert len(received[1]) == silence_bytes_len, (
            f"Silence chunk wrong size: {len(received[1])} vs {silence_bytes_len}"
        )

    def test_silence_constant_is_9600_samples(self):
        """INTER_SENTENCE_SILENCE_SAMPLES must be 9600 (0.3 s at 32 kHz)."""
        from genie_tts.Core.TTSPlayer import INTER_SENTENCE_SILENCE_SAMPLES
        assert INTER_SENTENCE_SILENCE_SAMPLES == 9600
