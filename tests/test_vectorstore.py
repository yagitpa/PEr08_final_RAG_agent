"""Векторное хранилище и манифест индекса.

Сеть не нужна: векторы передаются в хранилище напрямую, поэтому весь слой
проверяется на синтетических эмбеддингах.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from zerocoder_assistant.config.settings import ChunkingConfig
from zerocoder_assistant.preprocessing.models import Chunk, NoteMetadata, content_hash
from zerocoder_assistant.vectorstore.chroma_store import ChromaVectorStore, collection_name_for
from zerocoder_assistant.vectorstore.manifest import (
    MANIFEST_FILENAME,
    PREPROCESSING_VERSION,
    IndexManifest,
)

NOTE = NoteMetadata(
    source_file="Модуль 5/PEr08.md",
    lesson_title="PEr08. Практика",
    lesson_id="PEr08",
    module="Модуль 5",
    module_num=5,
)


def make_chunk(index: int, text: str, lesson_id: str = "PEr08") -> Chunk:
    note = (
        NOTE
        if lesson_id == "PEr08"
        else NoteMetadata(
            source_file=f"{lesson_id}.md", lesson_title=lesson_id, lesson_id=lesson_id
        )
    )
    return Chunk(
        chunk_id=f"{note.source_file}:s000:c{index:02d}",
        text=text,
        note=note,
        heading_path=("PEr08. Практика", "Теория"),
        section_title="Теория",
        chunk_index=index,
        chunks_in_section=1,
        content_type="theory",
        token_count=len(text),
        hash=content_hash(text),
    )


class TestCollectionNaming:
    def test_name_includes_model(self) -> None:
        """Коллекция именуется по модели: чужие векторы не подмешаются."""
        assert collection_name_for("text-embedding-3-small") == "notes__text-embedding-3-small"

    def test_unsafe_characters_replaced(self) -> None:
        assert collection_name_for("sber/GigaChat Embeddings") == "notes__sber-GigaChat-Embeddings"

    def test_different_models_give_different_collections(self) -> None:
        assert collection_name_for("text-embedding-3-small") != collection_name_for(
            "text-embedding-3-large"
        )


class TestChromaVectorStore:
    @pytest.fixture
    def store(self, tmp_path: Path) -> Iterator[ChromaVectorStore]:
        with ChromaVectorStore(tmp_path / "chroma", "text-embedding-3-small") as store:
            yield store

    def test_starts_empty(self, store: ChromaVectorStore) -> None:
        assert store.count() == 0
        assert store.query([1.0, 0.0], top_k=3) == []

    def test_upsert_then_query_finds_nearest(self, store: ChromaVectorStore) -> None:
        store.upsert(
            [make_chunk(0, "про векторы"), make_chunk(1, "про кэширование")],
            [[1.0, 0.0], [0.0, 1.0]],
        )
        hits = store.query([1.0, 0.0], top_k=2)

        assert hits[0].text == "про векторы"
        assert hits[0].similarity == pytest.approx(1.0)
        assert hits[1].similarity == pytest.approx(0.0)

    def test_similarity_is_inverse_of_distance(self, store: ChromaVectorStore) -> None:
        """Chroma отдаёт расстояние; порог релевантности задаётся в сходстве."""
        store.upsert([make_chunk(0, "текст")], [[1.0, 0.0]])
        hit = store.query([1.0, 0.0], top_k=1)[0]

        assert hit.similarity == pytest.approx(1.0 - hit.distance)

    def test_metadata_round_trip(self, store: ChromaVectorStore) -> None:
        chunk = make_chunk(0, "текст")
        store.upsert([chunk], [[1.0, 0.0]])
        metadata = store.query([1.0, 0.0], top_k=1)[0].metadata

        assert metadata["lesson_id"] == "PEr08"
        assert metadata["module_num"] == 5
        assert metadata["content_hash"] == chunk.hash

    def test_where_filter_limits_results(self, store: ChromaVectorStore) -> None:
        store.upsert(
            [make_chunk(0, "из PEr08"), make_chunk(1, "из PEr06", lesson_id="PEr06")],
            [[1.0, 0.0], [0.9, 0.1]],
        )
        hits = store.query([1.0, 0.0], top_k=5, where={"lesson_id": "PEr06"})

        assert [hit.metadata["lesson_id"] for hit in hits] == ["PEr06"]

    def test_upsert_replaces_by_id(self, store: ChromaVectorStore) -> None:
        store.upsert([make_chunk(0, "старый текст")], [[1.0, 0.0]])
        store.upsert([make_chunk(0, "новый текст")], [[1.0, 0.0]])

        assert store.count() == 1
        assert store.query([1.0, 0.0], top_k=1)[0].text == "новый текст"

    def test_stored_hashes_feed_idempotent_reindex(self, store: ChromaVectorStore) -> None:
        chunks = [make_chunk(i, f"текст {i}") for i in range(3)]
        store.upsert(chunks, [[1.0, 0.0]] * 3)
        stored = store.stored_hashes()

        assert stored == {chunk.chunk_id: chunk.hash for chunk in chunks}

    def test_delete_removes(self, store: ChromaVectorStore) -> None:
        chunks = [make_chunk(i, f"текст {i}") for i in range(3)]
        store.upsert(chunks, [[1.0, 0.0]] * 3)
        store.delete([chunks[0].chunk_id])

        assert store.count() == 2

    def test_reset_clears_collection(self, store: ChromaVectorStore) -> None:
        store.upsert([make_chunk(0, "текст")], [[1.0, 0.0]])
        store.reset()

        assert store.count() == 0

    def test_length_mismatch_refused(self, store: ChromaVectorStore) -> None:
        with pytest.raises(ValueError, match="количество должно совпадать"):
            store.upsert([make_chunk(0, "текст")], [[1.0, 0.0], [0.0, 1.0]])

    def test_persists_between_instances(self, tmp_path: Path) -> None:
        path = tmp_path / "chroma"
        with ChromaVectorStore(path, "text-embedding-3-small") as store:
            store.upsert([make_chunk(0, "текст")], [[1.0, 0.0]])

        with ChromaVectorStore(path, "text-embedding-3-small") as reopened:
            assert reopened.count() == 1

    def test_other_model_gets_separate_collection(self, tmp_path: Path) -> None:
        """Смена модели не должна подмешивать чужие векторы в существующий индекс."""
        path = tmp_path / "chroma"
        with ChromaVectorStore(path, "text-embedding-3-small") as store:
            store.upsert([make_chunk(0, "текст")], [[1.0, 0.0]])

        with ChromaVectorStore(path, "text-embedding-3-large") as other:
            assert other.count() == 0

    def test_directory_removable_after_close(self, tmp_path: Path) -> None:
        """На Windows незакрытый клиент держит файлы, и каталог не удалить.

        Это ломает не только тесты, но и пересборку индекса с нуля.
        """
        path = tmp_path / "chroma"
        with ChromaVectorStore(path, "text-embedding-3-small") as store:
            store.upsert([make_chunk(0, "текст")], [[1.0, 0.0]])

        shutil.rmtree(path)  # упадёт PermissionError, если close() не отпустил файлы
        assert not path.exists()


class TestManifest:
    @staticmethod
    def build(**overrides: object) -> IndexManifest:
        defaults: dict[str, object] = {
            "collection": "notes__text-embedding-3-small",
            "embed_provider": "openai",
            "embed_model": "text-embedding-3-small",
            "dimension": 1536,
            "chunking": ChunkingConfig(),
            "chunks": 920,
            "notes": 46,
        }
        return IndexManifest.create(**{**defaults, **overrides})  # type: ignore[arg-type]

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        original = self.build()
        original.save(tmp_path)

        assert IndexManifest.load(tmp_path) == original

    def test_absent_manifest_means_no_index(self, tmp_path: Path) -> None:
        assert IndexManifest.load(tmp_path) is None

    def test_broken_manifest_does_not_crash(self, tmp_path: Path) -> None:
        (tmp_path / MANIFEST_FILENAME).write_text("{не json", encoding="utf-8")

        assert IndexManifest.load(tmp_path) is None

    def test_manifest_from_older_schema_ignored(self, tmp_path: Path) -> None:
        (tmp_path / MANIFEST_FILENAME).write_text(
            json.dumps({"embed_model": "text-embedding-3-small"}), encoding="utf-8"
        )

        assert IndexManifest.load(tmp_path) is None

    def test_same_model_is_compatible(self) -> None:
        assert self.build().incompatibility("openai", "text-embedding-3-small") is None

    def test_other_model_is_incompatible(self) -> None:
        """Главная тихая поломка RAG: поиск не падает, а возвращает мусор."""
        reason = self.build().incompatibility("openai", "text-embedding-3-large")

        assert reason is not None
        assert "text-embedding-3-large" in reason

    def test_same_model_other_provider_is_compatible(self) -> None:
        """Одна модель через OpenAI и через ProxyAPI даёт сравнимые векторы."""
        assert self.build().incompatibility("proxyapi", "text-embedding-3-small") is None

    def test_fresh_manifest_is_not_stale(self) -> None:
        assert self.build().staleness(ChunkingConfig()) == []

    def test_changed_chunking_reported_as_stale(self) -> None:
        reasons = self.build().staleness(ChunkingConfig(target_tokens=300, max_tokens=600))

        assert len(reasons) == 1
        assert "target_tokens: 400 -> 300" in reasons[0]

    def test_preprocessing_version_recorded(self) -> None:
        assert self.build().preprocessing_version == PREPROCESSING_VERSION
