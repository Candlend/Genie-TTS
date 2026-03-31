"""Regression tests for word2ph / phone alignment bugs in ChineseG2P (spec 014)."""
from __future__ import annotations

import pytest

# These tests require the full Chinese G2P stack (jieba_fast, g2pM, pypinyin).
# Skip if not available.
pytest.importorskip('jieba_fast')
pytest.importorskip('g2pM')
pytest.importorskip('pypinyin')


class TestWord2PhAlignment:

    def _get_g2p(self):
        from genie_tts.G2P.Chinese.ChineseG2P import ChineseG2P
        return ChineseG2P()

    def test_sum_word2ph_equals_len_phones_normal_text(self):
        """For normal Chinese text, sum(word2ph) must equal len(phones)."""
        g2p = self._get_g2p()
        _, phones, _, word2ph = g2p.process('你好世界')
        assert sum(word2ph) == len(phones), (
            f"Invariant violated: sum(word2ph)={sum(word2ph)}, len(phones)={len(phones)}"
        )

    def test_sum_word2ph_equals_len_phones_with_punctuation(self):
        """word2ph/phones invariant holds even when punctuation is present."""
        g2p = self._get_g2p()
        _, phones, _, word2ph = g2p.process('你好，世界！')
        assert sum(word2ph) == len(phones), (
            f"Invariant violated: sum(word2ph)={sum(word2ph)}, len(phones)={len(phones)}"
        )

    def test_unknown_phone_substituted_with_unk_not_dropped(self):
        """Bug 1 fix: unknown phones use UNK substitution, preserving word2ph invariant."""
        from genie_tts.G2P.Chinese.ChineseG2P import ChineseG2P
        from genie_tts.G2P.SymbolsV2 import symbols_v2
        from unittest.mock import patch

        g2p = ChineseG2P()

        # Patch g2p to return a phone that is NOT in symbols_v2
        original_g2p = g2p.g2p

        def patched_g2p(text):
            phones, word2ph = original_g2p(text)
            # Inject a fake unknown phone into the middle
            if phones:
                phones[0] = '__UNKNOWN_PHONE__'
            return phones, word2ph

        g2p.g2p = patched_g2p
        _, phones, _, word2ph = g2p.process('你好')

        assert sum(word2ph) == len(phones), (
            f"word2ph/phones mismatch after unknown phone: "
            f"sum={sum(word2ph)}, len={len(phones)}"
        )
        # The unknown phone should be replaced with UNK
        assert 'UNK' in phones, "Unknown phone must be replaced with UNK"
        assert '__UNKNOWN_PHONE__' not in phones

    def test_keyerror_in_pinyin_conversion_uses_unk(self):
        """Bug 2 fix: KeyError in _pinyin_to_opencpop_phones emits UNK + word2ph entry."""
        from genie_tts.G2P.Chinese.ChineseG2P import ChineseG2P
        from unittest.mock import patch

        g2p = ChineseG2P()

        original_convert = g2p._pinyin_to_opencpop_phones

        call_count = [0]
        def patched_convert(c, v):
            call_count[0] += 1
            if call_count[0] == 1:
                raise KeyError(f'Unknown pinyin: {c}{v}')
            return original_convert(c, v)

        g2p._pinyin_to_opencpop_phones = patched_convert

        # Should not raise and invariant must hold
        phones, word2ph = g2p.g2p('你好')
        assert sum(word2ph) == len(phones), (
            f"Invariant violated after KeyError: sum={sum(word2ph)}, len={len(phones)}"
        )
