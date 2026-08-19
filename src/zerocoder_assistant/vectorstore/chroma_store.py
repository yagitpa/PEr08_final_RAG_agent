"""
Векторное хранилище на ChromaDB.

Два решения, заданные архитектурой:

* **Коллекция именуется по модели эмбеддингов.** Смена модели создаёт отдельную
  коллекцию, а не подмешивает чужие векторы в существующую. Вместе с манифестом
  это закрывает главную тихую поломку RAG.
* **Метрика — косинусная.** Chroma возвращает расстояние, а не сходство;
  для косинуса `similarity = 1 - distance` (проверено: одинаковые векторы дают
  0.0, ортогональные — 1.0). Пересчёт делает `SearchHit`, чтобы порог
  релевантности на этапе 4 задавался в понятных величинах.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from zerocoder_assistant.preprocessing.models import Chunk

logger = logging.getLogger(__name__)

#: Имена коллекций Chroma: буквы, цифры, дефис и подчёркивание.
_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9_-]+")

COLLECTION_PREFIX = "notes"
DISTANCE_METRIC = "cosine"

#: Chroma ограничивает размер одной операции добавления.
UPSERT_BATCH_SIZE = 256


def collection_name_for(embed_model: str, prefix: str = COLLECTION_PREFIX) -> str:
    """Имя коллекции для модели эмбеддингов: `notes__text-embedding-3-small`."""
    slug = _UNSAFE_NAME_CHARS.sub("-", embed_model).strip("-")
    return f"{prefix}__{slug}"


@dataclass(frozen=True, slots=True)
class SearchHit:
    """Найденный фрагмент."""

    chunk_id: str
    text: str
    metadata: dict[str, Any]
    distance: float

    @property
    def similarity(self) -> float:
        """Сходство в понятной шкале: 1.0 — совпадение, 0.0 — ничего общего."""
        return 1.0 - self.distance


class ChromaVectorStore:
    """Локальное хранилище векторов с сохранением на диск."""

    def __init__(self, path: Path, embed_model: str, prefix: str = COLLECTION_PREFIX) -> None:
        self.path = path
        self.collection_name = collection_name_for(embed_model, prefix)
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection: Collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": DISTANCE_METRIC},
        )
        logger.debug("Коллекция %s: %d векторов", self.collection_name, self.count())

    def count(self) -> int:
        return self._collection.count()

    def stored_hashes(self) -> dict[str, str]:
        """Соответствие `chunk_id -> content_hash` для всего, что уже в индексе.

        На этом строится идемпотентная переиндексация: заново векторизуются
        только изменившиеся чанки, а исчезнувшие удаляются.
        """
        stored: dict[str, str] = {}
        offset = 0
        while True:
            page = self._collection.get(
                include=["metadatas"], limit=UPSERT_BATCH_SIZE, offset=offset
            )
            ids = page.get("ids") or []
            if not ids:
                break
            metadatas = page.get("metadatas") or []
            for chunk_id, metadata in zip(ids, metadatas, strict=True):
                stored[chunk_id] = str((metadata or {}).get("content_hash", ""))
            offset += len(ids)
        return stored

    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        """Добавляет или заменяет чанки вместе с их векторами."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"чанков {len(chunks)}, векторов {len(embeddings)} — количество должно совпадать"
            )
        if not chunks:
            return

        for start in range(0, len(chunks), UPSERT_BATCH_SIZE):
            batch = chunks[start : start + UPSERT_BATCH_SIZE]
            self._collection.upsert(
                ids=[chunk.chunk_id for chunk in batch],
                documents=[chunk.text for chunk in batch],
                metadatas=[chunk.as_metadata() for chunk in batch],
                embeddings=[list(vector) for vector in embeddings[start : start + len(batch)]],
            )
        logger.info("В коллекцию %s записано %d чанков", self.collection_name, len(chunks))

    def delete(self, chunk_ids: Sequence[str]) -> None:
        if not chunk_ids:
            return
        for start in range(0, len(chunk_ids), UPSERT_BATCH_SIZE):
            self._collection.delete(ids=list(chunk_ids[start : start + UPSERT_BATCH_SIZE]))
        logger.info("Из коллекции %s удалено %d чанков", self.collection_name, len(chunk_ids))

    def query(
        self,
        embedding: Sequence[float],
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        """Ближайшие фрагменты к вектору запроса."""
        if self.count() == 0:
            return []

        result = self._collection.query(
            query_embeddings=[list(embedding)],
            n_results=min(top_k, self.count()),
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        return [
            SearchHit(
                chunk_id=chunk_id,
                text=document or "",
                metadata=dict(metadata or {}),
                distance=float(distance),
            )
            for chunk_id, document, metadata, distance in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=True,
            )
        ]

    def reset(self) -> None:
        """Полностью очищает коллекцию (для `index build --rebuild`)."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": DISTANCE_METRIC},
        )
        logger.info("Коллекция %s очищена", self.collection_name)

    def close(self) -> None:
        """Отпускает файлы базы.

        На Windows это обязательно: пока клиент жив, каталог с базой удалить
        нельзя — файлы заняты. Без явного закрытия ломается и удаление
        временных каталогов в тестах, и пересборка индекса с нуля.
        После закрытия хранилищем пользоваться нельзя.
        """
        self._client.close()

    def __enter__(self) -> ChromaVectorStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
