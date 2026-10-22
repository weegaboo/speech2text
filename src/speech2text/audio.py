import json
import math
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import SAMPLE_RATE, UserError

CORE_SAMPLES = 22 * SAMPLE_RATE
CONTEXT_SAMPLES = int(0.6 * SAMPLE_RATE)


@dataclass
class Chunk:
    samples: np.ndarray
    audio_start: int
    core_start: int
    core_end: int
    has_speech: bool
    final: bool


def require_audio_tools():
    for name in ("ffmpeg", "ffprobe"):
        if not shutil.which(name):
            raise UserError(f"Не найден {name}. Установите: brew install ffmpeg")


def probe(path: Path) -> float | None:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=duration:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise UserError(f"Не удалось прочитать аудио: {result.stderr.strip()[-1000:]}")
    data = json.loads(result.stdout)
    if not data.get("streams"):
        raise UserError("В файле не обнаружена аудиодорожка.")
    for duration in (data["streams"][0].get("duration"), data.get("format", {}).get("duration")):
        try:
            value = float(duration)
            if math.isfinite(value) and value > 0:
                return value
        except (TypeError, ValueError):
            pass
    return None


@contextmanager
def pcm_reader(path: Path, start_sample: int):
    # Output seeking is accurate even for formats without a useful seek table.
    command = ["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-i", str(path)]
    if start_sample:
        command += ["-ss", f"{start_sample / SAMPLE_RATE:.8f}"]
    command += [
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "pipe:1",
    ]
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=errors, start_new_session=True
        )
        try:
            yield process.stdout
            code = process.wait()
            if code:
                errors.seek(0)
                raise UserError(
                    "Ошибка декодирования: " + errors.read().decode(errors="replace")[-1500:]
                )
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdout.close()


class SpeechDetector:
    def __init__(self):
        from silero_vad import load_silero_vad

        # This model is shipped inside the installed wheel; no torch.hub/network access.
        self.model = load_silero_vad(onnx=True)

    def __call__(self, samples: np.ndarray) -> list[tuple[int, int]]:
        import torch
        from silero_vad import get_speech_timestamps

        spans = get_speech_timestamps(
            torch.from_numpy(samples.copy()),
            self.model,
            sampling_rate=SAMPLE_RATE,
            threshold=0.35,
            min_speech_duration_ms=100,
            min_silence_duration_ms=350,
            speech_pad_ms=150,
        )
        return [(s["start"], s["end"]) for s in spans]


def chunks(
    path: Path,
    start: int = 0,
    detector: Callable[[np.ndarray], list[tuple[int, int]]] | None = None,
) -> Iterator[Chunk]:
    """At most 23.2 seconds in memory; overlapping audio, non-overlapping ownership."""
    buffer_start = max(0, start - CONTEXT_SAMPLES)
    core_start = start
    buffer = np.empty(0, dtype=np.float32)
    with pcm_reader(path, buffer_start) as reader:
        eof = False
        while True:
            needed = core_start + CORE_SAMPLES + CONTEXT_SAMPLES - buffer_start - len(buffer)
            if needed > 0 and not eof:
                block = reader.read(needed * 4)
                if len(block) % 4:
                    raise UserError("Декодер вернул неполный аудиосэмпл.")
                eof = len(block) < needed * 4
                buffer = np.concatenate((buffer, np.frombuffer(block, dtype="<f4")))
            audio_end = buffer_start + len(buffer)
            if audio_end <= core_start:
                break
            spans = detector(buffer) if detector else [(0, len(buffer))]
            core_end = min(core_start + CORE_SAMPLES, audio_end)
            final = eof and core_end == audio_end
            if not final and detector:
                # Prefer a pause in the last ten seconds of the core window.
                lower = core_start + 12 * SAMPLE_RATE
                gaps = list(
                    zip(
                        [0] + [e for _, e in spans],
                        [s for s, _ in spans] + [len(buffer)],
                        strict=True,
                    )
                )
                candidates = []
                for left, right in gaps:
                    if right - left < int(0.35 * SAMPLE_RATE):
                        continue
                    midpoint = buffer_start + (left + right) // 2
                    if lower <= midpoint <= core_end:
                        candidates.append(midpoint)
                if candidates:
                    core_end = max(candidates)
            has_speech = any(
                buffer_start + left < core_end and buffer_start + right > core_start
                for left, right in spans
            )
            used_end = min(audio_end, core_end + CONTEXT_SAMPLES)
            yield Chunk(
                buffer[: used_end - buffer_start].copy(),
                buffer_start,
                core_start,
                core_end,
                has_speech,
                final,
            )
            core_start = core_end
            new_start = max(0, core_start - CONTEXT_SAMPLES)
            buffer = buffer[new_start - buffer_start :].copy()
            buffer_start = new_start
