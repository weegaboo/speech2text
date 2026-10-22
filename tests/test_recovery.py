import io
import sqlite3

from rich.console import Console

from speech2text import pipeline
from speech2text.backends import Transcript, Word
from speech2text.config import Settings


def test_inference_failure_resumes_without_repeating_completed_chunk(
    wav_file, tmp_path, monkeypatch
):
    source = wav_file(seconds=55)
    call_count = 0
    closed = []

    class Backend:
        def __init__(self, *args):
            pass

        def transcribe(self, samples):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("simulated crash")
            return Transcript("Слово.", [Word("Слово.", 1, 1.5)])

        def close(self):
            closed.append(1)

    monkeypatch.setattr(pipeline, "verify", lambda *args: "test-model")
    monkeypatch.setattr(pipeline.backends, "GigaAM", Backend)
    output = tmp_path / "texts"
    options = dict(
        source=source,
        output=output,
        settings=Settings(vad=False),
        models_dir=tmp_path,
        console=Console(file=io.StringIO()),
    )
    assert pipeline.run(**options).failed == 1
    assert not (output / "audio.txt").exists()
    with sqlite3.connect(output / ".speech2text/progress.sqlite3") as db:
        assert db.execute("select count(*) from chunks").fetchone()[0] == 1
    assert pipeline.run(**options).completed == 1
    assert (output / "audio.txt").read_text().split() == ["Слово."] * 3
    assert call_count == 4
    assert len(closed) == 2


def test_silent_audio_never_loads_recognizer(wav_file, tmp_path, monkeypatch):
    source = wav_file(silence=True)
    monkeypatch.setattr(pipeline, "verify", lambda *args: "test-model")
    monkeypatch.setattr(pipeline, "SpeechDetector", lambda: lambda _: [])
    monkeypatch.setattr(
        pipeline.backends, "GigaAM", lambda *args: (_ for _ in ()).throw(AssertionError())
    )
    result = pipeline.run(
        source, tmp_path / "texts", Settings(), tmp_path, console=Console(file=io.StringIO())
    )
    assert result.completed == 1
    assert (tmp_path / "texts/audio.txt").read_text() == ""
