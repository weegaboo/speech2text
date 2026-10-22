import wave

import numpy as np
import pytest


@pytest.fixture
def wav_file(tmp_path):
    def create(name="audio.wav", seconds=2.0, silence=False):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        count = int(seconds * 16000)
        samples = (
            np.zeros(count)
            if silence
            else np.sin(np.arange(count) * 2 * np.pi * 440 / 16000) * 0.15
        )
        with wave.open(str(path), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(16000)
            out.writeframes((samples * 32767).astype("<i2").tobytes())
        return path

    return create
