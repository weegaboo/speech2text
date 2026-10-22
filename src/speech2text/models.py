"""Explicit online preparation; inference only reads verified local assets."""

import hashlib
import json
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

from rich.progress import Progress

from .config import WHISPER_REVISION, UserError

GIGA_URL = "https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/"
WHISPER_SHA = "64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2"
GIGA_MD5 = "2730de7545ac43ad256485a462b0a27a"
GIGA_TOKENIZER_SHA = "828c12c991019eef952a960661f25a92d6ad279591e2ea466b4aeddf1d20a18a"


def digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, path: Path, checksum: str | None = None, algorithm="sha256"):
    if path.is_file() and (checksum is None or digest(path, algorithm) == checksum):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:
            total = int(response.headers.get("Content-Length", 0)) or None
            with Progress() as progress:
                task = progress.add_task(path.name, total=total)
                while block := response.read(1024 * 1024):
                    out.write(block)
                    progress.update(task, advance=len(block))
            out.flush()
            os.fsync(out.fileno())
        if checksum and digest(partial, algorithm) != checksum:
            raise UserError(f"Контрольная сумма не совпала: {path.name}. Повторите загрузку.")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def build_whisper(root: Path) -> Path:
    for tool in ("git", "cmake", "clang"):
        if not shutil.which(tool):
            raise UserError(
                f"Не найден {tool}. Установите Command Line Tools и brew install cmake."
            )
    native = Path(__file__).parent / "native"
    source = root / "whisper.cpp"
    if not source.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                "v1.9.5",
                "https://github.com/ggml-org/whisper.cpp.git",
                str(source),
            ],
            check=True,
        )
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != WHISPER_REVISION:
        raise UserError(f"Неожиданная версия whisper.cpp в {source}.")
    build = root / "build"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(native),
            "-B",
            str(build),
            f"-DWHISPER_SOURCE={source}",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        check=True,
    )
    subprocess.run(["cmake", "--build", str(build), "--config", "Release", "-j", "4"], check=True)
    return build / "speech2text-whisper"


def prepare(model: str, root: Path):
    root = root.resolve()
    folder = root / model
    folder.mkdir(parents=True, exist_ok=True)
    if model == "gigaam-v3":
        names = ["v3_e2e_rnnt.ckpt", "v3_e2e_rnnt_tokenizer.model"]
        download(GIGA_URL + names[0], folder / names[0], GIGA_MD5, "md5")
        download(GIGA_URL + names[1], folder / names[1], GIGA_TOKENIZER_SHA)
    else:
        name = "ggml-large-v3.bin"
        download(
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/" + name,
            folder / name,
            WHISPER_SHA,
        )
        build_whisper(folder)
        names = [name, "build/speech2text-whisper"]
    manifest = {name: digest(folder / name) for name in names}
    temporary = folder / "manifest.json.part"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(folder / "manifest.json")


def verify(model: str, root: Path) -> str:
    folder = root / model
    try:
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        required = (
            {"v3_e2e_rnnt.ckpt", "v3_e2e_rnnt_tokenizer.model"}
            if model == "gigaam-v3"
            else {"ggml-large-v3.bin", "build/speech2text-whisper"}
        )
        if set(manifest) != required:
            raise ValueError("invalid manifest")
        for name, expected in manifest.items():
            if digest(folder / name) != expected:
                raise ValueError(f"checksum: {name}")
        return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    except (OSError, ValueError, TypeError) as exc:
        raise UserError(
            f"Модель {model} отсутствует или повреждена. Выполните:\n"
            f"speech2text models download --model {model} --models-dir '{root}'"
        ) from exc
