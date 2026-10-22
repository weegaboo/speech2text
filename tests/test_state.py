import pytest

from speech2text.config import UserError
from speech2text.state import State, directory_lock


def make_job(state, root, settings=None, fingerprint="audio-sha"):
    return state.job(
        root / "audio.wav", root / "audio.txt", fingerprint, settings or {"language": "ru"}
    )


def test_checkpoint_survives_restart_and_export_is_complete(tmp_path):
    state = State(tmp_path / "db")
    job = make_job(state, tmp_path)
    state.save(job, 0, 16000, "Привет.", 0.0, 0.8)
    state.close()
    assert not (tmp_path / "audio.txt").exists()
    state = State(tmp_path / "db")
    assert state.position(job) == 16000
    state.save(job, 16000, 48000, "Вторая фраза.", 2.4, 2.9)
    state.export(job, tmp_path / "audio.txt", False)
    assert (tmp_path / "audio.txt").read_text() == "Привет.\n\nВторая фраза.\n"
    assert state.completed(job, tmp_path / "audio.txt")
    state.close()


def test_changed_audio_or_config_never_reuses_chunks(tmp_path):
    state = State(tmp_path / "db")
    first = make_job(state, tmp_path)
    state.save(first, 0, 16000, "Один.", 0, 0.5)
    changed = make_job(state, tmp_path, fingerprint="changed")
    english = make_job(state, tmp_path, {"language": "en"})
    assert len({first, changed, english}) == 3
    assert state.position(changed) == state.position(english) == 0
    state.close()


def test_atomic_no_overwrite_and_user_edits(tmp_path):
    state = State(tmp_path / "db")
    job = make_job(state, tmp_path)
    state.save(job, 0, 100, "Текст.", 0, 0.01)
    target = tmp_path / "audio.txt"
    target.write_text("Моя редактура")
    with pytest.raises(UserError, match="уже существует"):
        state.export(job, target, False)
    assert target.read_text() == "Моя редактура"
    assert not list(tmp_path.glob("*.tmp"))
    state.export(job, target, True)
    assert state.completed(job, target)
    target.write_text("Моя редактура")
    assert not state.completed(job, target)
    state.close()


def test_parallel_writers_rejected(tmp_path):
    with directory_lock(tmp_path):
        with pytest.raises(UserError, match="другим запуском"), directory_lock(tmp_path):
            pass


def test_out_of_order_checkpoint_rejected(tmp_path):
    state = State(tmp_path / "db")
    job = make_job(state, tmp_path)
    with pytest.raises(UserError, match="последовательность"):
        state.save(job, 100, 200, "", None, None)
    state.close()


def test_export_published_before_crash_is_recognized_on_restart(tmp_path):
    from speech2text.models import digest

    state = State(tmp_path / "db")
    job = make_job(state, tmp_path)
    target = tmp_path / "audio.txt"
    target.write_text("Полный текст.\n")
    with state.db:
        state.db.execute(
            "UPDATE jobs SET status='exporting',output_hash=? WHERE id=?", (digest(target), job)
        )
    state.fail(job, "interrupted after publish")
    state.close()
    state = State(tmp_path / "db")
    assert state.completed(job, target)
    assert state.status(job) == "complete"
    state.close()
