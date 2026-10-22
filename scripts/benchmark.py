"""Small reference-based end-to-end benchmark; intentionally separate from inference."""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from speech2text.audio import probe


def words(text):
    return re.findall(r"\w+", text.lower().replace("ё", "е"))


def wer(reference, hypothesis):
    expected, actual = words(reference), words(hypothesis)
    if not expected:
        raise ValueError("Эталонный текст пуст.")
    row = list(range(len(actual) + 1))
    for i, word in enumerate(expected, 1):
        current = [i]
        for j, candidate in enumerate(actual, 1):
            current.append(min(current[-1] + 1, row[j] + 1, row[j - 1] + (word != candidate)))
        row = current
    return row[-1] / len(expected)


def main():
    p = argparse.ArgumentParser(
        description="Сравнение моделей на аудио с проверенным эталонным текстом"
    )
    p.add_argument("audio", type=Path)
    p.add_argument("reference", type=Path)
    p.add_argument("--language", choices=["ru", "en"], default="ru")
    p.add_argument("--output", type=Path, default=Path("test-results/benchmark"))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    duration = probe(args.audio.resolve())
    reference = args.reference.read_text(encoding="utf-8")
    rows = []
    models = ["gigaam-v3", "whisper-large-v3"] if args.language == "ru" else ["whisper-large-v3"]
    for model in models:
        output = args.output.resolve() / model
        started = time.perf_counter()
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "speech2text",
                str(args.audio.resolve()),
                "--model",
                model,
                "--language",
                args.language,
                "--output",
                str(output),
                "--overwrite",
            ],
            capture_output=True,
            text=True,
        )
        elapsed = time.perf_counter() - started
        print(result.stderr, file=sys.stderr)
        row = {
            "model": model,
            "language": args.language,
            "seconds_including_startup": elapsed,
            "audio_seconds": duration,
            "exit_code": result.returncode,
        }
        if result.returncode == 0:
            text = (output / args.audio.with_suffix(".txt").name).read_text(encoding="utf-8")
            row.update(
                wer=wer(reference, text),
                transcript=text,
                real_time_factor=elapsed / duration if duration else None,
            )
        rows.append(row)
    report = args.output / "report.json"
    report.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(report)
    return 0 if all(row["exit_code"] == 0 for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
