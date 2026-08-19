"""
Поиск: порядок операций, порог, дедупликация, фильтры, кэш.

Эмбеддер подменяется заглушкой с управляемым сходством, поэтому проверяется
именно логика отбора, а не качество конкретной модели.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from zerocoder_assistant.cache.sqlite_cache import SqliteCache
from zerocoder_assistant.config.settings import Settings
from zerocoder_assistant.preprocessing.models import Chunk, NoteMetadata, content_hash
from zerocoder_assistant.retrieval import Retriever, build_where, deduplicate
from zerocoder_assistant.retrieval.dedup import jaccard, shingles
from zerocoder_assistant.vectorstore.chroma_store import ChromaVectorStore


@dataclass
class FakeHit:
    """Минимум, который дедупликации нужен от результата поиска."""

    chunk_id: str
    text: str
    similarity: float


class TestFilters:
    def test_no_filters(self) -> None:
        assert build_where() is None

    def test_single_lesson(self) -> None:
        assert build_where(lessons=["PEr06"]) == {"lesson_id": "PEr06"}

    def test_several_lessons_use_in(self) -> None:
        assert build_where(lessons=["PEr06", "PEr07"]) == {"lesson_id": {"$in": ["PEr06", "PEr07"]}}

    def test_two_dimensions_use_and(self) -> None:
        """Chroma не принимает два ключа верхнего уровня — нужен явный $and."""
        where = build_where(lessons=["PEr06"], modules=[5])

        assert set(where) == {"$and"}
        assert len(where["$and"]) == 2

    def test_content_type_filter(self) -> None:
        assert build_where(content_types=["summary"]) == {"content_type": "summary"}


class TestShingles:
    def test_identical_texts(self) -> None:
        text = "векторный поиск находит фрагменты по смысловой близости а не по словам"

        assert jaccard(shingles(text), shingles(text)) == 1.0

    def test_unrelated_texts(self) -> None:
        left = shingles("векторный поиск находит фрагменты по смысловой близости")
        right = shingles("борщ варят из свёклы капусты и говядины на медленном огне")

        assert jaccard(left, right) == 0.0

    def test_short_text_does_not_crash(self) -> None:
        assert shingles("два слова")
        assert jaccard(shingles(""), shingles("текст")) == 0.0


class TestDeduplicate:
    def test_keeps_everything_when_different(self) -> None:
        hits = [
            FakeHit("a", "векторный поиск ищет по смыслу а не по ключевым словам", 0.9),
            FakeHit("b", "кэширование позволяет пропустить дорогие шаги конвейера", 0.8),
        ]
        kept, removed = deduplicate(hits, threshold=0.8)

        assert len(kept) == 2
        assert removed == 0

    def test_exact_duplicate_removed(self) -> None:
        text = "перекрытие чанков помогает не терять контекст на границе"
        kept, removed = deduplicate([FakeHit("a", text, 0.9), FakeHit("b", text, 0.8)], 0.8)

        assert [hit.chunk_id for hit in kept] == ["a"]
        assert removed == 1

    def test_overlapping_chunks_collapse(self) -> None:
        """Перекрытие в 15% даёт именно такие пары — дословно совпадающие хвосты."""
        shared = (
            "перекрытие набирается целыми предложениями с конца предыдущего чанка "
            "чтобы не терять смысловую связность между соседними фрагментами текста"
        )
        hits = [
            FakeHit("a", f"{shared} и далее идёт продолжение раздела", 0.9),
            FakeHit("b", f"{shared} и далее идёт продолжение раздела почти так же", 0.85),
        ]
        kept, removed = deduplicate(hits, threshold=0.8)

        assert len(kept) == 1
        assert removed == 1

    def test_more_relevant_survives(self) -> None:
        """Вход отсортирован по убыванию релевантности — остаётся первый."""
        text = "порог релевантности отсекает слабые фрагменты до передачи в модель"
        kept, _ = deduplicate([FakeHit("лучший", text, 0.9), FakeHit("хуже", text, 0.4)], 0.8)

        assert kept[0].chunk_id == "лучший"

    def test_empty_input(self) -> None:
        assert deduplicate([], 0.8) == ([], 0)


class FakeEmbedder:
    """Возвращает заранее заданный вектор для каждого запроса."""

    def __init__(self, vectors: dict[str, list[float]] | None = None) -> None:
        self.vectors = vectors or {}
        self.calls = 0

    @property
    def model_id(self) -> str:
        return "fake-embed-v1"

    @property
    def dimension(self) -> int:
        return 2

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += len(texts)
        return [self.vectors.get(text, [1.0, 0.0]) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


def make_chunk(index: int, text: str, lesson_id: str = "PEr08") -> Chunk:
    note = NoteMetadata(
        source_file=f"{lesson_id}.md",
        lesson_title=f"{lesson_id}. Урок",
        lesson_id=lesson_id,
        module_num=5,
    )
    return Chunk(
        chunk_id=f"{lesson_id}:s000:c{index:02d}",
        text=text,
        note=note,
        heading_path=(f"{lesson_id}. Урок", "Теория"),
        section_title="Теория",
        chunk_index=index,
        chunks_in_section=1,
        content_type="theory",
        token_count=len(text),
        hash=content_hash(text),
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        openai_api_key="test-key",
        chroma_dir=tmp_path / "chroma",
        cache_db=tmp_path / "cache.db",
        top_k=2,
        overfetch_factor=3,
        relevance_threshold=0.5,
        dedup_threshold=0.8,
    )


@pytest.fixture
def store(settings: Settings) -> Iterator[ChromaVectorStore]:
    with ChromaVectorStore(settings.chroma_dir, "fake-embed-v1") as store:
        # Векторы подобраны так, чтобы сходство было предсказуемым:
        # запрос [1,0] даёт similarity 1.0 для [1,0] и 0.0 для [0,1].
        store.upsert(
            [
                make_chunk(0, "первый фрагмент про кэширование запросов в системе"),
                make_chunk(1, "второй фрагмент про кэширование ответов и векторов"),
                make_chunk(2, "совсем посторонний фрагмент про другое", lesson_id="PEr01"),
            ],
            [[1.0, 0.0], [0.95, 0.05], [0.0, 1.0]],
        )
        yield store


class TestRetriever:
    def test_threshold_cuts_weak_hits(self, settings: Settings, store: ChromaVectorStore) -> None:
        retriever = Retriever(settings, embedder=FakeEmbedder(), store=store)
        result = retriever.retrieve("вопрос", use_cache=False)

        assert result.below_threshold >= 1
        assert all(chunk.similarity >= settings.relevance_threshold for chunk in result.chunks)

    def test_nothing_relevant_is_empty_not_error(
        self, settings: Settings, store: ChromaVectorStore
    ) -> None:
        """Пустая выдача — это ответ «в базе нет», а не сбой."""
        embedder = FakeEmbedder({"чужое": [0.0, -1.0]})
        result = Retriever(settings, embedder=embedder, store=store).retrieve(
            "чужое", use_cache=False
        )

        assert result.is_empty
        assert result.chunks == []

    def test_top_k_applied_after_filtering(
        self, settings: Settings, store: ChromaVectorStore
    ) -> None:
        result = Retriever(settings, embedder=FakeEmbedder(), store=store).retrieve(
            "вопрос", use_cache=False
        )

        assert len(result.chunks) <= settings.top_k

    def test_overfetch_examines_more_than_top_k(
        self, settings: Settings, store: ChromaVectorStore
    ) -> None:
        """Кандидатов берётся больше, иначе порог сокращал бы контекст."""
        result = Retriever(settings, embedder=FakeEmbedder(), store=store).retrieve(
            "вопрос", use_cache=False
        )

        assert result.candidates > settings.top_k

    def test_explicit_top_k_overrides_settings(
        self, settings: Settings, store: ChromaVectorStore
    ) -> None:
        result = Retriever(settings, embedder=FakeEmbedder(), store=store).retrieve(
            "вопрос", top_k=1, use_cache=False
        )

        assert len(result.chunks) == 1

    def test_filter_limits_lessons(self, settings: Settings, store: ChromaVectorStore) -> None:
        embedder = FakeEmbedder()
        retriever = Retriever(settings, embedder=embedder, store=store)
        result = retriever.retrieve("вопрос", where=build_where(lessons=["PEr01"]), use_cache=False)

        assert all(chunk.metadata["lesson_id"] == "PEr01" for chunk in result.chunks)

    def test_timings_reported(self, settings: Settings, store: ChromaVectorStore) -> None:
        result = Retriever(settings, embedder=FakeEmbedder(), store=store).retrieve(
            "вопрос", use_cache=False
        )

        assert {"embed", "search", "total"} <= set(result.timings_ms)

    def test_sources_are_unique_and_readable(
        self, settings: Settings, store: ChromaVectorStore
    ) -> None:
        result = Retriever(settings, embedder=FakeEmbedder(), store=store).retrieve(
            "вопрос", use_cache=False
        )

        assert result.sources
        assert len(result.sources) == len(set(result.sources))


class TestRetrieverCache:
    def test_second_call_comes_from_cache(
        self, settings: Settings, store: ChromaVectorStore
    ) -> None:
        cache = SqliteCache(settings.cache_db)
        embedder = FakeEmbedder()
        retriever = Retriever(settings, embedder=embedder, store=store, cache=cache)

        first = retriever.retrieve("вопрос")
        calls_after_first = embedder.calls
        second = retriever.retrieve("вопрос")

        assert not first.from_cache
        assert second.from_cache
        assert embedder.calls == calls_after_first  # эмбеддер больше не звали
        assert [c.chunk_id for c in second.chunks] == [c.chunk_id for c in first.chunks]

    def test_changed_top_k_misses_cache(self, settings: Settings, store: ChromaVectorStore) -> None:
        """Ровно тот сценарий, на котором наивный кэш начинает врать."""
        cache = SqliteCache(settings.cache_db)
        retriever = Retriever(settings, embedder=FakeEmbedder(), store=store, cache=cache)

        retriever.retrieve("вопрос")
        other = retriever.retrieve("вопрос", top_k=1)

        assert not other.from_cache

    def test_changed_filter_misses_cache(
        self, settings: Settings, store: ChromaVectorStore
    ) -> None:
        cache = SqliteCache(settings.cache_db)
        retriever = Retriever(settings, embedder=FakeEmbedder(), store=store, cache=cache)

        retriever.retrieve("вопрос")
        other = retriever.retrieve("вопрос", where=build_where(lessons=["PEr01"]))

        assert not other.from_cache

    def test_embedding_cache_reused_across_params(
        self, settings: Settings, store: ChromaVectorStore
    ) -> None:
        """Вектор запроса от top_k не зависит, поэтому L1 переживает смену параметров."""
        cache = SqliteCache(settings.cache_db)
        embedder = FakeEmbedder()
        retriever = Retriever(settings, embedder=embedder, store=store, cache=cache)

        retriever.retrieve("вопрос")
        calls = embedder.calls
        retriever.retrieve("вопрос", top_k=1)  # L2 промахнулся, L1 должен попасть

        assert embedder.calls == calls

    def test_use_cache_false_bypasses(self, settings: Settings, store: ChromaVectorStore) -> None:
        cache = SqliteCache(settings.cache_db)
        retriever = Retriever(settings, embedder=FakeEmbedder(), store=store, cache=cache)

        retriever.retrieve("вопрос")
        again = retriever.retrieve("вопрос", use_cache=False)

        assert not again.from_cache
