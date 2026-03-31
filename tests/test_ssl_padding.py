"""Regression tests for SSL tail padding (spec 006)."""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call


SSL_TAIL_PADDING_SAMPLES_EXPECTED = 4800  # 0.3 s at 16 kHz


class TestSSLTailPadding:

    def _make_mocks(self, audio_len: int = 16000):
        """Return mocked model_manager and a fake audio_16k array."""
        mock_mm = MagicMock()
        mock_mm.cn_hubert = MagicMock()
        mock_mm.cn_hubert.run.return_value = [np.zeros((1, 50, 768), dtype=np.float32)]
        return mock_mm

    def test_hubert_called_with_padded_input(self):
        """HuBERT must receive input longer than the original audio."""
        from genie_tts.Audio.ReferenceAudio import SSL_TAIL_PADDING_SAMPLES

        original_len = 16000
        audio_32k = np.zeros(32000, dtype=np.float32)
        audio_16k = np.zeros((1, original_len), dtype=np.float32)

        mock_mm = self._make_mocks(original_len)

        with patch('genie_tts.Audio.ReferenceAudio.model_manager', mock_mm), \
             patch('genie_tts.Audio.ReferenceAudio.load_audio', return_value=np.zeros(32000, dtype=np.float32)), \
             patch('genie_tts.Audio.ReferenceAudio.soxr') as mock_soxr, \
             patch('genie_tts.Audio.ReferenceAudio.get_phones_and_bert',
                   return_value=(np.zeros((1, 4), dtype=np.int64),
                                 np.zeros((1, 4, 1024), dtype=np.float32))):
            mock_soxr.resample.return_value = np.zeros(original_len, dtype=np.float32)

            from genie_tts.Audio.ReferenceAudio import ReferenceAudio
            ReferenceAudio._prompt_cache.clear()
            ReferenceAudio('fake.wav', 'test text', 'Chinese')

        # Check HuBERT was called
        assert mock_mm.cn_hubert.run.called
        call_args = mock_mm.cn_hubert.run.call_args
        input_values = call_args[0][1]['input_values']
        assert input_values.shape[-1] == original_len + SSL_TAIL_PADDING_SAMPLES, (
            f"Expected padded length {original_len + SSL_TAIL_PADDING_SAMPLES}, "
            f"got {input_values.shape[-1]}"
        )

    def test_audio_16k_shape_unchanged(self):
        """self.audio_16k must retain original shape (padding not stored)."""
        original_len = 16000

        mock_mm = self._make_mocks(original_len)

        with patch('genie_tts.Audio.ReferenceAudio.model_manager', mock_mm), \
             patch('genie_tts.Audio.ReferenceAudio.load_audio', return_value=np.zeros(32000, dtype=np.float32)), \
             patch('genie_tts.Audio.ReferenceAudio.soxr') as mock_soxr, \
             patch('genie_tts.Audio.ReferenceAudio.get_phones_and_bert',
                   return_value=(np.zeros((1, 4), dtype=np.int64),
                                 np.zeros((1, 4, 1024), dtype=np.float32))):
            mock_soxr.resample.return_value = np.zeros(original_len, dtype=np.float32)

            from genie_tts.Audio.ReferenceAudio import ReferenceAudio
            ReferenceAudio._prompt_cache.clear()
            ref = ReferenceAudio('fake2.wav', 'test text', 'Chinese')

        # audio_16k should be (1, original_len) — unpadded
        assert ref.audio_16k.shape == (1, original_len), (
            f"audio_16k shape changed: {ref.audio_16k.shape}"
        )

    def test_ssl_tail_padding_samples_constant(self):
        """SSL_TAIL_PADDING_SAMPLES must equal 4800 (0.3 s at 16 kHz)."""
        from genie_tts.Audio.ReferenceAudio import SSL_TAIL_PADDING_SAMPLES
        assert SSL_TAIL_PADDING_SAMPLES == SSL_TAIL_PADDING_SAMPLES_EXPECTED
