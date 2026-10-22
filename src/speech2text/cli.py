import argparse
import shutil
import sys
from pathlib import Path

from rich.console import Console

from . import __version__
from .config import DEFAULT_MODELS, LANGUAGES, MODELS, Settings, UserError


def model_arguments(parser):
    parser.add_argument(
        "--model", choices=MODELS, default="gigaam-v3", help="модель (по умолчанию gigaam-v3)"
    )
    parser.add_argument(
        "--models-dir", type=Path, default=DEFAULT_MODELS, help="папка локальных моделей"
    )


def parser():
    p = argparse.ArgumentParser(
        prog="speech2text",
        description="Аудиофайл или папка → .txt. Локально, только CPU.",
        epilog=(
            "Подготовка: speech2text models download --model gigaam-v3. "
            "Диагностика: speech2text doctor."
        ),
    )
    p.add_argument("input", type=Path, help="аудиофайл или папка с записями")
    model_arguments(p)
    p.add_argument(
        "--language", choices=LANGUAGES, default="ru", help="язык всех записей (по умолчанию ru)"
    )
    p.add_argument("--output", type=Path, help="папка для результатов")
    p.add_argument("--recursive", action="store_true", help="обрабатывать вложенные папки")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="распознать заново и заменить существующие результаты",
    )
    p.add_argument("--threads", type=int, default=4, help="потоки CPU (по умолчанию 4)")
    p.add_argument(
        "--no-vad", action="store_true", help="не отсеивать тишину (для очень тихой речи)"
    )
    p.add_argument("--version", action="version", version=__version__)
    return p


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    console = Console(stderr=True, markup=False, highlight=False)
    try:
        if argv[:1] == ["models"]:
            p = argparse.ArgumentParser(prog="speech2text models")
            sub = p.add_subparsers(dest="action", required=True)
            model_arguments(
                sub.add_parser("download", help="скачать модель и подготовить CPU-движок")
            )
            args = p.parse_args(argv[1:])
            from .models import prepare
            from .state import directory_lock

            root = args.models_dir.expanduser().resolve()
            with directory_lock(root):
                prepare(args.model, root)
            console.print(f"Модель {args.model} готова: {root / args.model}")
            return 0
        if argv[:1] == ["doctor"]:
            p = argparse.ArgumentParser(prog="speech2text doctor")
            p.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
            args = p.parse_args(argv[1:])
            from .models import verify

            console.print(f"speech2text {__version__} | Python {sys.version.split()[0]} | CPU only")
            missing = False
            for tool in ("ffmpeg", "ffprobe"):
                location = shutil.which(tool)
                console.print(f"{tool}: {location or 'не найден'}")
                missing |= location is None
            for model in MODELS:
                try:
                    verify(model, args.models_dir.expanduser().resolve())
                    console.print(f"{model}: готова, контрольные суммы совпали")
                except UserError:
                    console.print(f"{model}: не подготовлена")
            return 2 if missing else 0
        args = parser().parse_args(argv)
        settings = Settings(args.model, args.language, args.threads, not args.no_vad)
        settings.validate()
        from .pipeline import run

        return run(
            args.input,
            args.output,
            settings,
            args.models_dir.expanduser().resolve(),
            args.recursive,
            args.overwrite,
            console,
        ).exit_code
    except KeyboardInterrupt:
        console.print("\nОстановлено. Прогресс сохранён; повторите ту же команду для продолжения.")
        return 130
    except UserError as exc:
        console.print(f"Ошибка: {exc}")
        return 2
    except Exception as exc:
        console.print(f"Ошибка: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
