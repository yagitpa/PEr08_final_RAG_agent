"""Общие фикстуры тестов препроцессинга."""

from __future__ import annotations

from pathlib import Path

import pytest

from zerocoder_assistant.config import get_settings
from zerocoder_assistant.config.settings import ChunkingConfig
from zerocoder_assistant.preprocessing import NotePreprocessor, ProcessedNote, iter_note_files


@pytest.fixture(scope="session")
def chunking() -> ChunkingConfig:
    """Настройки чанкинга по умолчанию."""
    return ChunkingConfig()


@pytest.fixture(scope="session")
def preprocessor(chunking: ChunkingConfig) -> NotePreprocessor:
    return NotePreprocessor(chunking)


@pytest.fixture(scope="session")
def notes_dir() -> Path:
    """Корень реального корпуса конспектов.

    Тесты, которым он нужен, пропускаются, если корпус недоступен: путь задаётся
    через NOTES_DIR и на чужой машине может отсутствовать.
    """
    directory = get_settings().notes_dir
    if not directory.is_dir() or not any(iter_note_files(directory)):
        pytest.skip(f"корпус конспектов недоступен: {directory}")
    return directory


@pytest.fixture(scope="session")
def corpus(preprocessor: NotePreprocessor, notes_dir: Path) -> list[ProcessedNote]:
    """Весь корпус, обработанный один раз на сессию тестов."""
    return [preprocessor.process_file(path, notes_dir) for path in iter_note_files(notes_dir)]
