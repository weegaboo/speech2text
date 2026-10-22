import hashlib
import json

import pytest

from speech2text.config import UserError
from speech2text.models import download, verify


def test_verification_never_downloads_missing_assets(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("network access")

    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    with pytest.raises(UserError, match="models download"):
        verify("gigaam-v3", tmp_path)


def test_corrupt_model_rejected(tmp_path):
    folder = tmp_path / "gigaam-v3"
    folder.mkdir()
    names = ["v3_e2e_rnnt.ckpt", "v3_e2e_rnnt_tokenizer.model"]
    manifest = {}
    for name in names:
        (folder / name).write_bytes(b"original")
        manifest[name] = hashlib.sha256(b"original").hexdigest()
    (folder / "manifest.json").write_text(json.dumps(manifest))
    assert verify("gigaam-v3", tmp_path)
    (folder / names[0]).write_bytes(b"corrupt")
    with pytest.raises(UserError, match="повреждена"):
        verify("gigaam-v3", tmp_path)


def test_failed_download_does_not_publish_partial_model(tmp_path, monkeypatch):
    import io

    class Response(io.BytesIO):
        headers = {"Content-Length": "3"}

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Response(b"bad"))
    target = tmp_path / "model.bin"
    with pytest.raises(UserError, match="Контрольная сумма"):
        download("https://example.invalid/model", target, "not-the-sha")
    assert not target.exists()
    assert not target.with_name("model.bin.part").exists()
