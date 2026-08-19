"""
Интеграционные проверки на реальном корпусе конспектов.

Это тот самый прогон, который в docs/architecture.md записан как верификация
этапа 1. Синтетические примеры не ловят перекос оформления, который есть в
живых файлах: разнобой в нумерации заголовков, таблицы, вложенные листинги.

Если корпус недоступен (NOTES_DIR не настроен), тесты пропускаются.
"""

from __future__ import annotations

from pathlib import Path

from zerocoder_assistant.config.settings import ChunkingConfig
from zerocoder_assistant.preprocessing import NotePreprocessor, ProcessedNote
from zerocoder_assistant.preprocessing.headings import is_excluded_section, normalize_for_match
from zerocoder_assistant.reporting import build_corpus_stats

#: Доля чанков вне коридора размеров, выше которой чанкинг считается разлаженным.
#: Не ноль: короткие секции корпуса и атомарные листинги выходят за границы
#: законно, и требовать идеала значило бы либо резать код, либо терять текст.
MAX_OUT_OF_RANGE_SHARE = 0.15


def all_chunks(corpus: list[ProcessedNote]) -> list:
    return [chunk for note in corpus for chunk in note.chunks]


class TestCorpusProcessing:
    def test_every_file_yields_chunks(self, corpus: list[ProcessedNote]) -> None:
        empty = [note.note.source_file for note in corpus if not note.chunks]

        assert not empty, f"файлы без единого чанка: {empty}"

    def test_chunk_ids_are_unique(self, corpus: list[ProcessedNote]) -> None:
        """Коллизия id при upsert молча затирает чужой чанк, поэтому проверяем."""
        chunks = all_chunks(corpus)
        ids = {chunk.chunk_id for chunk in chunks}

        assert len(ids) == len(chunks)

    def test_code_fences_are_balanced(self, corpus: list[ProcessedNote]) -> None:
        """Непарный ``` означает разорванный посередине листинг."""
        broken = [c.chunk_id for c in all_chunks(corpus) if c.text.count("```") % 2]

        assert not broken, f"чанки с разорванным листингом: {broken[:5]}"

    def test_no_empty_chunks(self, corpus: list[ProcessedNote]) -> None:
        assert all(chunk.text.strip() for chunk in all_chunks(corpus))

    def test_metadata_is_flat_scalars(self, corpus: list[ProcessedNote]) -> None:
        """Chroma принимает только скаляры — проверяем до того, как соберём индекс."""
        for chunk in all_chunks(corpus):
            for key, value in chunk.as_metadata().items():
                assert isinstance(value, str | int | float | bool), f"{key}={value!r}"


class TestSectionFiltering:
    def test_service_sections_are_dropped(self, corpus: list[ProcessedNote]) -> None:
        leaked = [
            chunk.chunk_id
            for chunk in all_chunks(corpus)
            if any(is_excluded_section(part) for part in chunk.heading_path)
        ]

        assert not leaked, f"служебные секции просочились: {leaked[:5]}"

    def test_boilerplate_disclaimer_removed(self, corpus: list[ProcessedNote]) -> None:
        """Шапка «Дорогой студент!» одинакова во всех файлах — классический дубль."""
        leaked = [
            chunk.chunk_id
            for chunk in all_chunks(corpus)
            if "дорогой студент" in normalize_for_match(chunk.text)
        ]

        assert not leaked

    def test_summary_sections_are_kept(self, corpus: list[ProcessedNote]) -> None:
        """«Результат дня» — лучший чанк для вопроса «о чём был урок»."""
        summaries = [c for c in all_chunks(corpus) if c.content_type == "summary"]

        assert summaries

    def test_admin_header_not_indexed(self, corpus: list[ProcessedNote]) -> None:
        """Шапка «Курс / Время чтения» почти одинакова во всех файлах.

        Оставить её — значит завести два десятка близнецов, конкурирующих за
        места в top_k с настоящим содержимым.
        """
        leaked = [
            chunk.chunk_id
            for chunk in all_chunks(corpus)
            if "время чтения и просмотра" in normalize_for_match(chunk.text)
        ]

        assert not leaked

    def test_short_definitions_survive(self, corpus: list[ProcessedNote]) -> None:
        """Короткое определение — плотное знание, а не шум: порог по размеру его не режет."""
        texts = [chunk.text for chunk in all_chunks(corpus)]

        assert any("автономная программа" in text for text in texts)
        assert any("Нода Merge" in text for text in texts)

    def test_module_prefixed_prose_survives(self, corpus: list[ProcessedNote]) -> None:
        """Разбор кода из PEr07 начинается как «Модуль N:», но это содержимое."""
        texts = [chunk.text for chunk in all_chunks(corpus)]

        assert any("rag.py" in text and "ядро нашей системы" in text for text in texts)


class TestChunkSizes:
    def test_most_chunks_within_corridor(
        self, corpus: list[ProcessedNote], chunking: ChunkingConfig
    ) -> None:
        stats = build_corpus_stats(corpus, chunking)

        assert stats.out_of_range / stats.chunks <= MAX_OUT_OF_RANGE_SHARE

    def test_oversized_chunks_are_only_code(
        self, corpus: list[ProcessedNote], chunking: ChunkingConfig
    ) -> None:
        """Превышать лимит вправе только неделимый листинг."""
        oversized = [c for c in all_chunks(corpus) if c.token_count > chunking.max_tokens]

        assert all(chunk.content_type == "code" for chunk in oversized)

    def test_median_near_target(
        self, corpus: list[ProcessedNote], chunking: ChunkingConfig
    ) -> None:
        stats = build_corpus_stats(corpus, chunking)
        lower_bound = chunking.target_tokens * 0.6

        assert stats.median_tokens >= lower_bound


class TestMetadataExtraction:
    def test_lesson_ids_recognised(self, corpus: list[ProcessedNote]) -> None:
        without = [note.note.source_file for note in corpus if not note.note.lesson_id]

        assert not without, f"не распознан lesson_id: {without}"

    def test_module_number_parsed_where_folder_has_one(self, corpus: list[ProcessedNote]) -> None:
        numbered = [note for note in corpus if note.note.module and "Модуль" in note.note.module]

        assert numbered
        assert all(note.note.module_num is not None for note in numbered)


class TestDeterminism:
    def test_repeated_run_is_identical(
        self, preprocessor: NotePreprocessor, notes_dir: Path, corpus: list[ProcessedNote]
    ) -> None:
        """Идемпотентная переиндексация опирается на стабильность хешей."""
        sample = corpus[0]
        again = preprocessor.process_file(notes_dir / sample.note.source_file, notes_dir)

        assert [c.hash for c in again.chunks] == [c.hash for c in sample.chunks]
        assert [c.chunk_id for c in again.chunks] == [c.chunk_id for c in sample.chunks]
