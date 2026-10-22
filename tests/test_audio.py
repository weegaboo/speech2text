import shutil

import numpy as np
import pytest

from speech2text.audio import CONTEXT_SAMPLES, CORE_SAMPLES, Chunk, chunks
from speech2text.backends import Transcript, Word, assemble, owned_text

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")


def test_chunks_bounded_and_resume_sample_exact(wav_file):
    path = wav_file(seconds=80)
    original = list(chunks(path))
    assert len(original) == 4
    assert original[0].core_start == 0
    assert original[-1].core_end == 80 * 16000
    assert all(len(c.samples) <= CORE_SAMPLES + 2 * CONTEXT_SAMPLES for c in original)
    assert all(a.core_end == b.core_start for a, b in zip(original, original[1:], strict=False))
    resumed = list(chunks(path, original[1].core_end))
    assert len(resumed) == 2
    for expected, actual in zip(original[2:], resumed, strict=True):
        assert (expected.core_start, expected.core_end) == (actual.core_start, actual.core_end)
        np.testing.assert_array_equal(expected.samples, actual.samples)


def test_silence_keeps_full_timeline(wav_file):
    path = wav_file(seconds=60, silence=True)
    result = list(chunks(path, detector=lambda _: []))
    assert all(not c.has_speech for c in result)
    assert result[-1].core_end == 60 * 16000


def test_prefers_pause(wav_file):
    path = wav_file(seconds=30)
    result = list(chunks(path, detector=lambda a: [(0, 16 * 16000), (17 * 16000, len(a))]))
    assert result[0].core_end == int(16.5 * 16000)


def test_timestamp_ownership_preserves_real_repeated_words():
    result = Transcript("да да да", [Word("да", 0, 0.4), Word("да", 0.6, 1), Word("да", 1.2, 1.6)])
    left = Chunk(np.zeros(32000), 0, 0, 16000, True, False)
    right = Chunk(np.zeros(32000), 0, 16000, 32000, True, True)
    assert owned_text(result, left)[0] == "да да"
    assert owned_text(result, right)[0] == "да"


def test_missing_word_timestamps_fail_instead_of_dropping_text():
    from speech2text.config import UserError

    with pytest.raises(UserError, match="временных меток"):
        owned_text(Transcript("Слова", []), Chunk(np.zeros(100), 0, 0, 100, True, True))


def test_overlap_deduplicates_only_the_same_acoustic_word():
    left = Chunk(np.zeros(32000), 0, 0, 16000, True, False)
    right = Chunk(np.zeros(32000), 0, 16000, 32000, True, True)
    a = assemble(Transcript("отдельную", [Word("отдельную", 0.8, 1.16)]), left, [])
    b = assemble(
        Transcript("отдельную папку", [Word("отдельную", 0.84, 1.24), Word("папку", 1.3, 1.7)]),
        right,
        a[3],
    )
    assert a[0] == "отдельную"
    assert b[0] == "папку"


def test_overlap_recovers_word_whose_midpoint_crossed_cut():
    left = Chunk(np.zeros(32000), 0, 0, 16000, True, False)
    right = Chunk(np.zeros(32000), 0, 16000, 32000, True, True)
    a = assemble(Transcript("отдельную", [Word("отдельную", 0.84, 1.24)]), left, [])
    b = assemble(
        Transcript("отдельную папку", [Word("отдельную", 0.8, 1.16), Word("папку", 1.3, 1.7)]),
        right,
        a[3],
    )
    assert a[0] == ""
    assert b[0] == "отдельную папку"


def test_overlap_does_not_remove_actual_repetition():
    left = Chunk(np.zeros(32000), 0, 0, 16000, True, False)
    right = Chunk(np.zeros(32000), 0, 16000, 32000, True, True)
    a = assemble(Transcript("да", [Word("да", 0.6, 0.9)]), left, [])
    b = assemble(Transcript("да да", [Word("да", 0.65, 0.95), Word("да", 1.1, 1.4)]), right, a[3])
    assert a[0] == "да" and b[0] == "да"


def test_context_anchors_compensate_for_whisper_timestamp_drift():
    left = Chunk(np.zeros(32000), 0, 0, 16000, True, False)
    right = Chunk(np.zeros(32000), 0, 16000, 32000, True, True)
    a = assemble(
        Transcript(
            "в отдельную папку",
            [Word("в", 0.12, 0.12), Word("отдельную", 0.14, 0.75), Word("папку", 0.83, 1.58)],
        ),
        left,
        [],
    )
    b = assemble(
        Transcript("отдельную папку", [Word("отдельную", 0.64, 1.24), Word("папку", 1.25, 1.9)]),
        right,
        a[3],
    )
    assert a[0] == "в отдельную"
    assert b[0] == "папку"
