"""Общие фикстуры: корпус конспектов, изолированные настройки, тестовое хранилище."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fakes import make_chunk

from zerocoder_assistant.config import get_settings
from zerocoder_assistant.config.settings import ChunkingConfig, Settings
from zerocoder_assistant.preprocessing import NotePreprocessor, ProcessedNote, iter_note_files
from zerocoder_assistant.vectorstore.chroma_store import ChromaVectorStore


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


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Настройки, изолированные во временной папке.

    Порог поднят до 0.5 против рабочих 0.33: заглушка эмбеддера даёт сходство
    ровно 1.0 или 0.0, и при таких значениях порог виден в тестах однозначно.
    """
    return Settings(
        openai_api_key="test-key",
        chroma_dir=tmp_path / "chroma",
        cache_db=tmp_path / "cache.db",
        prompts_dir=tmp_path / "prompts",
        top_k=2,
        overfetch_factor=3,
        relevance_threshold=0.5,
        dedup_threshold=0.8,
    )


@pytest.fixture
def store(settings: Settings) -> Iterator[ChromaVectorStore]:
    """Хранилище с тремя фрагментами и предсказуемыми векторами."""
    with ChromaVectorStore(settings.chroma_dir, "fake-embed-v1") as store:
        # Запрос [1,0] даёт similarity 1.0 для [1,0], 0.95 для [0.95,0.05]
        # и 0.0 для [0,1] — то есть третий фрагмент заведомо ниже порога.
        store.upsert(
            [
                make_chunk(0, "первый фрагмент про кэширование запросов в системе"),
                make_chunk(1, "второй фрагмент про кэширование ответов и векторов"),
                make_chunk(2, "совсем посторонний фрагмент про другое", lesson_id="PEr01"),
            ],
            [[1.0, 0.0], [0.95, 0.05], [0.0, 1.0]],
        )
        yield store
