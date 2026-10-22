import hashlib
import importlib.metadata
import logging
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from . import backends
from .audio import SpeechDetector, chunks, probe, require_audio_tools
from .config import (
    GIGA_REVISION,
    PIPELINE_VERSION,
    SAMPLE_RATE,
    WHISPER_REVISION,
    Settings,
    UserError,
)
from .models import digest, verify
from .state import State, directory_lock

EXTENSIONS = {
    ".mp3",
    ".m4a",
    ".wav",
    ".ogg",
    ".opus",
    ".flac",
    ".aac",
    ".aiff",
    ".aif",
    ".wma",
    ".webm",
}


def discover(source: Path, recursive: bool) -> list[Path]:
    if source.is_file():
        # Messenger exports may have no extension; ffprobe validates their contents later.
        if source.suffix and source.suffix.lower() not in EXTENSIONS:
            raise UserError(f"Неподдерживаемое расширение {source.suffix}.")
        return [source]
    if not source.is_dir():
        raise UserError(f"Путь не существует: {source}")
    paths = source.rglob("*") if recursive else source.iterdir()
    files = sorted(
        p
        for p in paths
        if p.is_file()
        and not p.is_symlink()
        and p.suffix.lower() in EXTENSIONS
        and ".speech2text" not in p.relative_to(source).parts
    )
    if not files:
        raise UserError("Аудиофайлы не найдены. Для подпапок добавьте --recursive.")
    return files


def targets(files: list[Path], source: Path, output: Path) -> dict[Path, Path]:
    base = source if source.is_dir() else source.parent
    proposed = {p: output / p.relative_to(base).with_suffix(".txt") for p in files}
    groups: dict[str, list[Path]] = {}
    for p, target in proposed.items():
        groups.setdefault(str(target).casefold(), []).append(p)
    reserved = {str(t).casefold() for t in proposed.values()}
    for group in groups.values():
        if len(group) > 1:
            for p in group:
                token = hashlib.sha256(str(p.relative_to(base)).encode()).hexdigest()[:10]
                candidate = proposed[p].with_name(f"{p.name}.{token}.txt")
                while str(candidate).casefold() in reserved:
                    candidate = candidate.with_name(candidate.stem + "_" + ".txt")
                proposed[p] = candidate
                reserved.add(str(candidate).casefold())
    return proposed


@dataclass
class Summary:
    completed: int = 0
    skipped: int = 0
    failed: int = 0

    @property
    def exit_code(self):
        return (1 if self.completed + self.skipped else 2) if self.failed else 0


def run(
    source: Path,
    output: Path | None,
    settings: Settings,
    models_dir: Path,
    recursive=False,
    overwrite=False,
    console: Console | None = None,
) -> Summary:
    console = console or Console(stderr=True, markup=False)
    settings.validate()
    require_audio_tools()
    source = source.expanduser().resolve()
    files = discover(source, recursive)
    output = (
        output.expanduser().resolve()
        if output
        else source.parent
        if source.is_file()
        else source.with_name(source.name + "_text")
    )
    if output.exists() and not output.is_dir():
        raise UserError("--output должен указывать на папку, а не на файл.")
    mapping = targets(files, source, output)
    console.print(
        f"Модель: {settings.model} | язык: {settings.language} | CPU: {settings.threads} потока"
    )
    console.print(f"Файлов: {len(files)} | результаты: {output}")
    console.print("Проверка локальной модели…")
    model_identity = verify(settings.model, models_dir)
    identity = {
        **asdict(settings),
        "pipeline": PIPELINE_VERSION,
        "model_hash": model_identity,
        "engine_revision": GIGA_REVISION if settings.model == "gigaam-v3" else WHISPER_REVISION,
        "vad_version": importlib.metadata.version("silero-vad"),
        "torch_version": importlib.metadata.version("torch"),
        "ffmpeg_version": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[
            0
        ],
    }
    state_dir = output / ".speech2text"
    summary = Summary()
    backend = None
    detector = None
    started = time.monotonic()
    with directory_lock(state_dir):
        state = State(state_dir / "progress.sqlite3")
        logger = logging.getLogger("speech2text")
        handler = logging.FileHandler(state_dir / "run.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        try:
            for index, source_file in enumerate(files, 1):
                job = None
                processing = False
                try:
                    target = mapping[source_file]
                    console.print(f"[{index}/{len(files)}] {source_file.name}")
                    fingerprint = digest(source_file)
                    job = state.job(source_file, target, fingerprint, identity)
                    if not overwrite and state.completed(job, target):
                        summary.skipped += 1
                        console.print("  Уже готово, пропускаю.")
                        continue
                    if (target.exists() or target.is_symlink()) and not overwrite:
                        raise UserError(
                            f"Файл уже существует: {target}. Выберите другую папку или --overwrite."
                        )
                    processing = True
                    if overwrite and state.status(job) == "complete":
                        state.reset(job)
                    duration = probe(source_file)
                    position = state.position(job)
                    previous_tail = state.tail(job)
                    if position:
                        console.print(f"  Продолжаю с {position / SAMPLE_RATE:.1f} с.")
                    if settings.vad and detector is None:
                        detector = SpeechDetector()
                    with Progress(
                        TextColumn("{task.description}", markup=False),
                        BarColumn(),
                        TaskProgressColumn(),
                        TimeElapsedColumn(),
                        console=console,
                        disable=not console.is_terminal,
                    ) as progress:
                        task = progress.add_task(
                            "Распознавание", total=duration, completed=position / SAMPLE_RATE
                        )
                        for chunk in chunks(source_file, position, detector):
                            text, speech_start, speech_end = "", None, None
                            tail = []
                            if chunk.has_speech:
                                if backend is None:
                                    console.print("  Загрузка модели в память…")
                                    factory = (
                                        backends.GigaAM
                                        if settings.model == "gigaam-v3"
                                        else backends.Whisper
                                    )
                                    backend = factory(
                                        settings, models_dir, state_dir / "whisper.log"
                                    )
                                result = backend.transcribe(chunk.samples)
                                text, speech_start, speech_end, tail = backends.assemble(
                                    result, chunk, previous_tail
                                )
                            state.save(
                                job,
                                chunk.core_start,
                                chunk.core_end,
                                text,
                                speech_start,
                                speech_end,
                                tail,
                            )
                            previous_tail = tail
                            progress.update(task, completed=chunk.core_end / SAMPLE_RATE)
                            if not console.is_terminal:
                                console.print(
                                    f"  Обработано {chunk.core_end / SAMPLE_RATE:.1f} с аудио."
                                )
                    if not state.position(job):
                        raise UserError("В файле нет декодируемых аудиосэмплов.")
                    if digest(source_file) != fingerprint:
                        raise UserError(
                            "Исходный файл изменился во время обработки. Запустите команду заново."
                        )
                    state.export(job, target, overwrite)
                    summary.completed += 1
                    logger.info("Completed %s -> %s", source_file, target)
                    console.print(f"  Сохранено: {target}")
                except KeyboardInterrupt:
                    if job and processing:
                        state.fail(job, "Прервано пользователем")
                    raise
                except Exception as exc:
                    summary.failed += 1
                    if job and processing:
                        state.fail(job, str(exc))
                    logger.exception("Failed %s", source_file)
                    console.print(f"  Ошибка: {exc}")
                    if backend:
                        backend.close()
                        backend = None
        finally:
            if backend:
                backend.close()
            state.close()
            logger.removeHandler(handler)
            handler.close()
    console.print(
        f"Готово: {summary.completed}; пропущено: {summary.skipped}; ошибок: {summary.failed}."
    )
    console.print(f"Время: {time.monotonic() - started:.1f} с. Результаты: {output}")
    return summary
