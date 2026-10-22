#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
if ! command -v uv >/dev/null 2>&1; then
  echo "Установите uv: brew install uv" >&2
  exit 2
fi
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v cmake >/dev/null 2>&1; then
  echo "Установите инструменты: brew install ffmpeg cmake" >&2
  exit 2
fi
uv sync --locked
echo "Окружение готово. Загрузите нужные модели:"
echo "  uv run speech2text models download --model gigaam-v3"
echo "  uv run speech2text models download --model whisper-large-v3"
echo "Запуск: uv run speech2text /путь/аудио.m4a"
