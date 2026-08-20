"""
Сборка индекса: конспекты -> чанки -> векторы -> хранилище.

Ключевое свойство — идемпотентность. Сборщик сравнивает `content_hash` каждого
чанка с тем, что уже лежит в базе, и векторизует только изменившееся. Повторный
запуск без правок в конспектах не делает ни одного обращения к API.

Отсюда же следует, что сборка **возобновляема**: если она прервалась на середине,
уже записанные чанки остаются в базе, и следующий запуск продолжит с того места,
а не начнёт заново. Манифест при этом записывается только после успеха — иначе он
объявил бы полным индекс, собранный наполовину.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from zerocoder_assistant.config.settings import Settings, get_settings
from zerocoder_assistant.embeddings.base import EmbeddingProvider
from zerocoder_assistant.embeddings.factory import build_embedding_provider
from zerocoder_assistant.errors import IndexBuildError, IndexMismatchError
from zerocoder_assistant.preprocessing import (
    Chunk,
    NotePreprocessor,
    ProcessedNote,
    iter_note_files,
)
from zerocoder_assistant.vectorstore.chroma_store import ChromaVectorStore
from zerocoder_assistant.vectorstore.manifest import IndexManifest

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IndexPlan:
    """Что именно изменится при сборке."""

    added: list[Chunk] = field(default_factory=list)
    updated: list[Chunk] = field(default_factory=list)
    unchanged: list[Chunk] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    @property
    def to_embed(self) -> list[Chunk]:
        """Чанки, которые придётся векторизовать (то есть оплатить)."""
        return [*self.added, *self.updated]

    @property
    def is_noop(self) -> bool:
        return not self.added and not self.updated and not self.removed


@dataclass(frozen=True, slots=True)
class IndexReport:
    """Результат сборки."""

    notes: int
    chunks: int
    added: int
    updated: int
    unchanged: int
    removed: int
    embedded: int
    failed_files: list[tuple[str, str]]
    duration_seconds: float
    collection: str
    manifest: IndexManifest | None
    dry_run: bool = False
    staleness: list[str] = field(default_factory=list)


class IndexBuilder:
    """Приводит векторное хранилище в соответствие с конспектами."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        preprocessor: NotePreprocessor | None = None,
        embedder: EmbeddingProvider | None = None,
        store: ChromaVectorStore | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._preprocessor = preprocessor or NotePreprocessor(self._settings.chunking)
        self._embedder = embedder
        self._store = store

    # -- публичный интерфейс ----------------------------------------------

    def build(
        self,
        *,
        rebuild: bool = False,
        lessons: Sequence[str] | None = None,
        dry_run: bool = False,
    ) -> IndexReport:
        """Собирает или обновляет индекс.

        `rebuild` очищает коллекцию и векторизует всё заново — нужен при смене
        модели эмбеддингов или правил препроцессинга.

        `lessons` ограничивает сборку конкретными уроками. В этом режиме удаление
        отключается: иначе частичная сборка снесла бы из индекса всё остальное.

        Вместе с `rebuild` фильтр запрещён. Поодиночке каждый безопасен, а
        вместе они делают ровно то, от чего защищает предыдущий абзац: очистка
        сносит коллекцию целиком, и наполняется она потом только отобранными
        уроками. Отчёт при этом выглядит успешным и показывает `removed=0` —
        плановых удалений действительно не было, содержимое исчезло раньше.

        `dry_run` показывает план, ничего не меняя и не тратя ни одного запроса.
        """
        if rebuild and lessons:
            raise IndexBuildError(
                "Полная пересборка (--rebuild) несовместима с фильтром по урокам "
                "(--lesson): очистка снесёт коллекцию целиком, а наполнится она "
                "только отобранными уроками.\n"
                "Пересоберите весь индекс без --lesson либо обновите отдельные "
                "уроки без --rebuild."
            )

        started = time.monotonic()
        embedder = self._embedder or build_embedding_provider(self._settings)

        # Сверка манифеста идёт первой, до разбора конспектов и до открытия
        # хранилища. Иначе несовместимая настройка сначала тратит секунды на
        # препроцессинг, а открытие хранилища успевает создать пустую коллекцию
        # под новую модель — она остаётся сиротой после отказа.
        staleness = self._check_manifest(embedder, rebuild=rebuild)

        notes, failed = self._load_notes(lessons)
        chunks = [chunk for note in notes for chunk in note.chunks]

        store = self._store or ChromaVectorStore(self._settings.chroma_dir, embedder.model_id)
        owns_store = self._store is None
        try:
            if rebuild and not dry_run:
                store.reset()

            stored = {} if rebuild else store.stored_hashes()
            # Сбой чтения делает сборку такой же неполной, как и фильтр по
            # урокам: часть конспектов мы просто не видели. Разница только в
            # том, что фильтр — это выбор, а сбой — случайность, и именно
            # поэтому он опаснее. Без этой оговорки нечитаемый файл (битая
            # кодировка, несинхронизированный плейсхолдер OneDrive, блокировка
            # антивирусом) выглядел бы как удалённый, и все его чанки уходили
            # бы из базы знаний — при том что на диске файл цел.
            plan = self._plan(chunks, stored, partial=bool(lessons) or bool(failed))

            embedded = 0
            manifest: IndexManifest | None = None
            if not dry_run:
                embedded = self._apply(plan, store, embedder)
                manifest = self._write_manifest(store, embedder, len(chunks), len(notes))

            return IndexReport(
                notes=len(notes),
                chunks=len(chunks),
                added=len(plan.added),
                updated=len(plan.updated),
                unchanged=len(plan.unchanged),
                removed=len(plan.removed),
                embedded=embedded,
                failed_files=failed,
                duration_seconds=time.monotonic() - started,
                collection=store.collection_name,
                manifest=manifest,
                dry_run=dry_run,
                staleness=staleness,
            )
        finally:
            if owns_store:
                store.close()

    # -- шаги ---------------------------------------------------------------

    def _load_notes(
        self, lessons: Sequence[str] | None
    ) -> tuple[list[ProcessedNote], list[tuple[str, str]]]:
        """Препроцессинг всех конспектов. Сбой на файле не роняет сборку."""
        wanted = {lesson.casefold() for lesson in lessons} if lessons else None
        notes: list[ProcessedNote] = []
        failed: list[tuple[str, str]] = []

        for path in iter_note_files(self._settings.notes_dir):
            try:
                note = self._preprocessor.process_file(path, self._settings.notes_dir)
            except Exception as exc:
                logger.warning("Не удалось обработать %s: %s", path.name, exc)
                failed.append((path.name, str(exc)))
                continue

            if wanted is not None and (note.note.lesson_id or "").casefold() not in wanted:
                continue
            notes.append(note)

        return notes, failed

    def _check_manifest(self, embedder: EmbeddingProvider, *, rebuild: bool) -> list[str]:
        """Сверяет существующий индекс с текущими настройками.

        Несовместимость по модели — отказ, если только не запрошена пересборка:
        поиск в чужом векторном пространстве вернул бы правдоподобный мусор.
        """
        manifest = IndexManifest.load(self._settings.chroma_dir)
        if manifest is None:
            return []

        reason = manifest.incompatibility(self._settings.embed_provider, embedder.model_id)
        if reason and not rebuild:
            raise IndexMismatchError(reason)

        staleness = manifest.staleness(self._settings.chunking)
        for note in staleness:
            logger.warning("Индекс устарел: %s", note)
        return staleness

    @staticmethod
    def _plan(chunks: Sequence[Chunk], stored: dict[str, str], *, partial: bool) -> IndexPlan:
        """Разница между тем, что есть в базе, и тем, что получилось из конспектов."""
        plan = IndexPlan()
        current: set[str] = set()

        for chunk in chunks:
            current.add(chunk.chunk_id)
            previous = stored.get(chunk.chunk_id)
            if previous is None:
                plan.added.append(chunk)
            elif previous != chunk.hash:
                plan.updated.append(chunk)
            else:
                plan.unchanged.append(chunk)

        if not partial:
            # При сборке по фильтру всё остальное «отсутствует» лишь потому,
            # что мы его не читали, — удалять такое нельзя.
            plan.removed.extend(sorted(set(stored) - current))
        return plan

    def _apply(self, plan: IndexPlan, store: ChromaVectorStore, embedder: EmbeddingProvider) -> int:
        """Векторизует изменившееся и приводит хранилище в соответствие."""
        to_embed = plan.to_embed
        if to_embed:
            logger.info("Векторизация: %d чанков", len(to_embed))
            vectors = embedder.embed([chunk.text for chunk in to_embed])
            store.upsert(to_embed, vectors)

        if plan.removed:
            store.delete(plan.removed)

        return len(to_embed)

    def _write_manifest(
        self,
        store: ChromaVectorStore,
        embedder: EmbeddingProvider,
        chunks: int,
        notes: int,
    ) -> IndexManifest:
        manifest = IndexManifest.create(
            collection=store.collection_name,
            embed_provider=self._settings.embed_provider,
            embed_model=embedder.model_id,
            dimension=embedder.dimension,
            chunking=self._settings.chunking,
            chunks=chunks,
            notes=notes,
        )
        manifest.save(self._settings.chroma_dir)
        return manifest
