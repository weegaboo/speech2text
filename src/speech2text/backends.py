import json
import os
import re
import struct
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import Chunk
from .config import SAMPLE_RATE, Settings, UserError


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Transcript:
    text: str
    words: list[Word]


class GigaAM:
    def __init__(self, settings: Settings, root: Path, log_path: Path):
        import gigaam
        import torch

        torch.set_num_threads(settings.threads)
        self.model = gigaam.load_model(
            "v3_e2e_rnnt",
            device="cpu",
            fp16_encoder=False,
            use_flash=False,
            download_root=str(root / "gigaam-v3"),
        )
        if any(p.device.type != "cpu" for p in self.model.parameters()):
            raise UserError("GigaAM должна работать только на CPU.")

    def transcribe(self, samples: np.ndarray) -> Transcript:
        with tempfile.TemporaryDirectory(prefix="speech2text-") as tmp:
            wav = Path(tmp) / "chunk.wav"
            with wave.open(str(wav), "wb") as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(SAMPLE_RATE)
                out.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
            result = self.model.transcribe(str(wav), word_timestamps=True)
        return Transcript(result.text, [Word(w.text, w.start, w.end) for w in result.words or []])

    def close(self):
        self.model = None


class Whisper:
    def __init__(self, settings: Settings, root: Path, log_path: Path):
        folder = root / "whisper-large-v3"
        self.errors = log_path.open("ab")
        self.process = subprocess.Popen(
            [
                str(folder / "build/speech2text-whisper"),
                str(folder / "ggml-large-v3.bin"),
                settings.language,
                str(settings.threads),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.errors,
            start_new_session=True,
            env={**os.environ, "GGML_METAL_DEVICE": "", "CUDA_VISIBLE_DEVICES": ""},
        )
        try:
            if not self._response().get("ready"):
                raise UserError("Whisper не подтвердил готовность.")
        except BaseException:
            self.close()
            raise

    def _response(self) -> dict:
        line = self.process.stdout.readline()
        if not line:
            raise UserError("Процесс Whisper завершился. Подробности в журнале whisper.log.")
        try:
            response = json.loads(line)
        except ValueError as exc:
            raise UserError("Whisper вернул некорректный ответ.") from exc
        if "error" in response:
            raise UserError(response["error"])
        return response

    def transcribe(self, samples: np.ndarray) -> Transcript:
        try:
            self.process.stdin.write(struct.pack("<I", len(samples)))
            self.process.stdin.write(samples.astype("<f4").tobytes())
            self.process.stdin.flush()
            result = self._response()
        except BrokenPipeError as exc:
            raise UserError("Процесс Whisper прерван. Повторите команду для продолжения.") from exc
        return Transcript(result["text"], [Word(**word) for word in result["words"]])

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()
        self.errors.close()


def owned_text(result: Transcript, chunk: Chunk) -> tuple[str, float | None, float | None]:
    return assemble(result, chunk, [])[0:3]


def assemble(
    result: Transcript, chunk: Chunk, previous: list[dict]
) -> tuple[str, float | None, float | None, list[dict]]:
    """Reconcile the overlap by acoustic time AND word, preserving genuine repetitions.

    The prior tail also remembers uncommitted words. This avoids losing a word if its
    midpoint moves from one side of the cut to the other between model calls.
    """
    if not result.text.strip():
        return "", None, None, []
    if not result.words:
        raise UserError("Модель вернула текст без временных меток; безопасная сборка невозможна.")
    offset = chunk.audio_start / SAMPLE_RATE
    lower, upper = chunk.core_start / SAMPLE_RATE, chunk.core_end / SAMPLE_RATE

    def normalize(text):
        return re.sub(r"\W", "", text.casefold().replace("ё", "е"))

    absolute = [Word(w.text, w.start + offset, w.end + offset) for w in result.words]

    def nearby_match(current_index: int, previous_index: int) -> bool:
        if not (0 <= current_index < len(absolute) and 0 <= previous_index < len(previous)):
            return False
        current, prev = absolute[current_index], previous[previous_index]
        return (
            bool(normalize(current.text))
            and normalize(current.text) == normalize(prev["text"])
            and abs((current.start + current.end - prev["start"] - prev["end"]) / 2) <= 1.0
        )

    used_previous = set()
    selected = []
    tail = []
    for current_index, w in enumerate(absolute):
        midpoint = (w.start + w.end) / 2
        keep = lower <= midpoint < upper
        matches = []
        for index, prev in enumerate(previous):
            if index in used_previous or normalize(w.text) != normalize(prev["text"]):
                continue
            intersection = min(w.end, prev["end"]) - max(w.start, prev["start"])
            shorter = min(w.end - w.start, prev["end"] - prev["start"])
            acoustic_match = shorter > 0 and intersection >= shorter * 0.5
            # Whisper's heuristic timestamps can drift or give zero-length tokens.
            # Require a neighbouring matching word as evidence before tolerating drift.
            contextual_match = nearby_match(current_index, index) and (
                nearby_match(current_index - 1, index - 1)
                or nearby_match(current_index + 1, index + 1)
            )
            if acoustic_match or contextual_match:
                distance = abs(midpoint - (prev["start"] + prev["end"]) / 2)
                matches.append((distance, index))
        if matches:
            _, index = min(matches)
            used_previous.add(index)
            keep = not previous[index]["emitted"] and midpoint < upper
        if keep:
            selected.append(w)
        if w.end >= upper - 1.2:
            tail.append({"text": w.text, "start": w.start, "end": w.end, "emitted": keep})
    words = selected
    if not words:
        return "", None, None, tail
    text = (
        result.text if len(words) == len(result.words) else " ".join(w.text.strip() for w in words)
    )
    return " ".join(text.split()), words[0].start, words[-1].end, tail
