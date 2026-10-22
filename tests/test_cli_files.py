import io

import pytest
from rich.console import Console

from speech2text import cli
from speech2text.config import Settings, UserError
from speech2text.pipeline import discover, targets


def test_english_giga_rejected_before_io(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("must not start file processing")

    monkeypatch.setattr("speech2text.pipeline.run", forbidden)
    assert cli.main(["missing.wav", "--language", "en"]) == 2


@pytest.mark.parametrize(
    "model,language",
    [
        ("gigaam-v3", "ru"),
        ("whisper-large-v3", "ru"),
        ("whisper-large-v3", "en"),
    ],
)
def test_valid_settings(model, language):
    Settings(model, language).validate()


def test_recursive_discovery_and_collisions(wav_file, tmp_path):
    first = wav_file("inputs/запись.wav")
    second = first.with_suffix(".mp3")
    second.write_bytes(b"fake")
    nested = wav_file("inputs/sub/запись.WAV")
    (first.parent / "ignore.txt").write_text("ignore")
    (first.parent / "linked.wav").symlink_to(first)
    files = discover(first.parent, True)
    assert set(files) == {first, second, nested}
    assert len(discover(first.parent, False)) == 2
    outputs = targets(files, first.parent, tmp_path / "texts")
    assert len({str(p).casefold() for p in outputs.values()}) == 3
    assert outputs[nested] == tmp_path / "texts/sub/запись.txt"
    assert ".wav." in outputs[first].name


def test_empty_input_is_clear_error(tmp_path):
    with pytest.raises(UserError, match="не найдены"):
        discover(tmp_path, False)


def test_extensionless_audio_is_detected_and_gets_txt_name(wav_file, tmp_path):
    from speech2text.audio import probe

    source = wav_file("audio", seconds=2)
    assert discover(source, False) == [source]
    assert probe(source) == pytest.approx(2)
    assert targets([source], source, tmp_path)[source] == tmp_path / "audio.txt"


def test_extensionless_non_audio_is_rejected_by_probe(tmp_path):
    from speech2text.audio import probe

    source = tmp_path / "audio"
    source.write_text("This is not an audio recording.")
    with pytest.raises(UserError, match="аудио"):
        probe(source)


def test_resume_skip_and_language_change(wav_file, tmp_path, monkeypatch):
    from speech2text import pipeline
    from speech2text.backends import Transcript, Word

    source = wav_file()
    calls = []

    class FakeBackend:
        def __init__(self, *args):
            pass

        def transcribe(self, samples):
            calls.append(len(samples))
            return Transcript("Привет, мир!", [Word("Привет,", 0.1, 0.5), Word("мир!", 0.6, 1.0)])

        def close(self):
            pass

    monkeypatch.setattr(pipeline, "verify", lambda *a: "model-sha")
    monkeypatch.setattr(pipeline.backends, "Whisper", FakeBackend)
    output = tmp_path / "out"
    console = Console(file=io.StringIO())
    options = dict(
        source=source,
        output=output,
        models_dir=tmp_path,
        settings=Settings("whisper-large-v3", "ru", vad=False),
        console=console,
    )
    assert pipeline.run(**options).completed == 1
    assert pipeline.run(**options).skipped == 1
    assert len(calls) == 1
    options["settings"] = Settings("whisper-large-v3", "en", vad=False)
    assert pipeline.run(**options).failed == 1
    assert len(calls) == 1
    assert pipeline.run(**options, overwrite=True).completed == 1
    assert len(calls) == 2


def test_failed_file_does_not_stop_batch(wav_file, tmp_path, monkeypatch):
    from speech2text import pipeline
    from speech2text.backends import Transcript, Word

    good = wav_file("inputs/b_good.wav")
    (good.parent / "a_broken.mp3").write_text("not audio")
    wav_file("inputs/c_good.wav")
    loads = []

    class FakeBackend:
        def __init__(self, *args):
            loads.append(1)

        def transcribe(self, samples):
            return Transcript("Тест.", [Word("Тест.", 0.1, 0.9)])

        def close(self):
            pass

    monkeypatch.setattr(pipeline, "verify", lambda *a: "sha")
    monkeypatch.setattr(pipeline.backends, "GigaAM", FakeBackend)
    result = pipeline.run(
        good.parent,
        tmp_path / "out",
        Settings(vad=False),
        tmp_path,
        console=Console(file=io.StringIO()),
    )
    assert result.completed == 2 and result.failed == 1 and result.exit_code == 1
    assert len(loads) == 1
