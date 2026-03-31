"""Regression tests for short-phones guard in get_phones_and_bert (spec 007)."""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import patch, MagicMock


def _make_phones_result(n_phones: int):
    """Return (phones_seq, text_bert) with exactly n_phones phone tokens."""
    return (
        np.zeros((1, n_phones), dtype=np.int64),
        np.zeros((n_phones, 1024), dtype=np.float32),
    )


class TestShortPhonesGuard:

    def test_min_phone_len_constant(self):
        """MIN_PHONE_LEN must equal 6."""
        from genie_tts.GetPhonesAndBert import MIN_PHONE_LEN
        assert MIN_PHONE_LEN == 6

    def test_long_sequence_returned_as_is(self):
        """Text producing >= 6 phones is returned without retry."""
        from genie_tts.GetPhonesAndBert import get_phones_and_bert

        long_result = _make_phones_result(8)

        with patch('genie_tts.GetPhonesAndBert._get_phones_and_bert_single',
                   return_value=long_result) as mock_single:
            phones_seq, _ = get_phones_and_bert('Hello world', 'English')

        # Should be called exactly once — no retry
        assert mock_single.call_count == 1
        assert phones_seq.shape[-1] == 8

    def test_short_sequence_triggers_retry_with_dot_prefix(self):
        """Text producing < 6 phones triggers a retry with '.' prepended."""
        from genie_tts.GetPhonesAndBert import get_phones_and_bert

        short_result = _make_phones_result(3)
        long_result = _make_phones_result(7)

        call_texts = []

        def fake_single(text, language):
            call_texts.append(text)
            if text.startswith('.'):
                return long_result
            return short_result

        with patch('genie_tts.GetPhonesAndBert._get_phones_and_bert_single',
                   side_effect=fake_single):
            phones_seq, _ = get_phones_and_bert('Hi', 'English')

        assert len(call_texts) == 2, f"Expected 2 calls, got {len(call_texts)}: {call_texts}"
        assert call_texts[1].startswith('.'), (
            f"Retry text must start with '.', got: {call_texts[1]!r}"
        )
        assert phones_seq.shape[-1] == 7

    def test_retry_does_not_recurse_infinitely(self):
        """_final=True prevents infinite recursion even if retry also short."""
        from genie_tts.GetPhonesAndBert import get_phones_and_bert

        short_result = _make_phones_result(2)

        call_count = [0]

        def fake_single(text, language):
            call_count[0] += 1
            return short_result

        with patch('genie_tts.GetPhonesAndBert._get_phones_and_bert_single',
                   side_effect=fake_single):
            # Should not raise RecursionError
            phones_seq, _ = get_phones_and_bert('Hi', 'English')

        # At most 2 calls: original + one retry
        assert call_count[0] <= 2, f"Too many calls: {call_count[0]}"
