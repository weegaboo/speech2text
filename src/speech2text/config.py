from dataclasses import dataclass
from pathlib import Path

MODELS = ("gigaam-v3", "whisper-large-v3")
LANGUAGES = ("ru", "en")
GIGA_REVISION = "7447938d791c4f3e643386ee22c33777004293a5"
WHISPER_REVISION = "d1be6fde11ac6e0407606b4e42fe72d34add8037"
PIPELINE_VERSION = 3
SAMPLE_RATE = 16000
DEFAULT_MODELS = Path.home() / "Library/Caches/speech2text/models"


class UserError(Exception):
    """An actionable error that does not need a traceback in the terminal."""


@dataclass(frozen=True)
class Settings:
    model: str = "gigaam-v3"
    language: str = "ru"
    threads: int = 4
    vad: bool = True

    def validate(self):
        if self.model not in MODELS or self.language not in LANGUAGES:
            raise UserError("Неизвестная модель или язык.")
        if self.model == "gigaam-v3" and self.language != "ru":
            raise UserError(
                "GigaAM-v3 поддерживает только ru. Для en выберите --model whisper-large-v3."
            )
        if not 1 <= self.threads <= 128:
            raise UserError("--threads должен быть от 1 до 128.")
