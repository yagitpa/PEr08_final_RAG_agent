"""
Поиск релевантных фрагментов.

Порядок операций принципиален:

    overfetch -> порог -> дедупликация -> срез top_k

Если взять сразу `top_k` и потом отсечь по порогу, порог будет уменьшать выдачу
вместо того, чтобы её улучшать: из пяти найденных останется три, и место двух
отсеянных ничем не займётся. При обратном порядке кандидатов берётся втрое
больше, слабые выбрасываются, дубли схлопываются, и только потом отрезается
ровно `top_k` — то есть порог повышает качество, а не сокращает контекст.

Отдельное правило: **если после порога не осталось ничего, это ответ, а не
ошибка.** Пустая выдача означает «в базе знаний такого нет», и генерацию в этом
случае запускать нельзя — именно так порог экономит деньги и снимает
галлюцинации на вопросах вне корпуса.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from zerocoder_assistant.cache.keys import RetrievalParams, embedding_key, retrieval_key
from zerocoder_assistant.cache.sqlite_cache import SqliteCache
from zerocoder_assistant.config.settings import Settings, get_settings
from zerocoder_assistant.embeddings.base import EmbeddingProvider
from zerocoder_assistant.embeddings.factory import build_embedding_provider
from zerocoder_assistant.errors import IndexMismatchError
from zerocoder_assistant.retrieval.dedup import deduplicate
from zerocoder_assistant.vectorstore.chroma_store import ChromaVectorStore
from zerocoder_assistant.vectorstore.manifest import IndexManifest

logger = logging.getLogger(__name__)

#: Отметка индекса, когда манифест недоступен. Попадает в ключ кэша, поэтому
#: результаты, посчитанные без манифеста, не смешиваются с остальными.
UNKNOWN_INDEX_VERSION = "unknown"


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """Найденный фрагмент в виде, пригодном и для контекста, и для кэша."""

    chunk_id: str
    text: str
    similarity: float
    metadata: dict[str, Any]

    @property
    def source(self) -> str:
        """Человекочитаемая ссылка на источник для показа пользователю."""
        lesson = self.metadata.get("lesson_id")
        section = self.metadata.get("section_title") or self.metadata.get("lesson_title")
        return f"{lesson} > {section}" if lesson else str(section or self.chunk_id)

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "similarity": self.similarity,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetrievedChunk:
        return cls(
            chunk_id=data["chunk_id"],
            text=data["text"],
            similarity=data["similarity"],
            metadata=data["metadata"],
        )


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Результат поиска вместе с тем, как он получен."""

    query: str
    chunks: list[RetrievedChunk]
    from_cache: bool = False
    candidates: int = 0
    below_threshold: int = 0
    duplicates: int = 0
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        """Ничего не найдено — значит в базе знаний ответа нет."""
        return not self.chunks

    @property
    def sources(self) -> list[str]:
        seen: dict[str, None] = {}
        for chunk in self.chunks:
            seen.setdefault(chunk.source, None)
        return list(seen)


class Retriever:
    """Находит фрагменты базы знаний, релевантные запросу."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        embedder: EmbeddingProvider | None = None,
        store: ChromaVectorStore | None = None,
        cache: SqliteCache | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._embedder = embedder or build_embedding_provider(self._settings)
        self._manifest = IndexManifest.load(self._settings.chroma_dir)
        self._guard_index()
        self._store = store or ChromaVectorStore(self._settings.chroma_dir, self._embedder.model_id)
        self._owns_store = store is None
        self._cache = cache

    # -- жизненный цикл ----------------------------------------------------

    def close(self) -> None:
        if self._owns_store:
            self._store.close()

    def __enter__(self) -> Retriever:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- поиск -------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> RetrievalResult:
        """Ищет фрагменты по запросу.

        Пустой результат — законный исход, а не сбой: он означает, что в базе
        знаний нет ничего достаточно близкого.
        """
        params = self._params(top_k, where)
        timings: dict[str, float] = {}

        if use_cache and self._cache is not None:
            cached = self._cache.get_retrieval(retrieval_key(query, params))
            if cached is not None:
                logger.debug("Поиск: попадание в кэш L2")
                return RetrievalResult(
                    query=query,
                    chunks=[RetrievedChunk.from_dict(item) for item in cached],
                    from_cache=True,
                    timings_ms={"total": 0.0},
                )

        started = time.perf_counter()
        vector = self._embed(query, use_cache=use_cache)
        timings["embed"] = _elapsed_ms(started)

        searched = time.perf_counter()
        raw = self._store.query(vector, params.top_k * params.overfetch_factor, where=where)
        timings["search"] = _elapsed_ms(searched)

        result = self._select(query, raw, params)
        timings["total"] = _elapsed_ms(started)

        if use_cache and self._cache is not None:
            self._cache.set_retrieval(
                retrieval_key(query, params),
                query,
                params.fingerprint(),
                [chunk.as_dict() for chunk in result.chunks],
            )

        return RetrievalResult(
            query=result.query,
            chunks=result.chunks,
            candidates=result.candidates,
            below_threshold=result.below_threshold,
            duplicates=result.duplicates,
            timings_ms=timings,
        )

    # -- шаги ---------------------------------------------------------------

    def _guard_index(self) -> None:
        """Отказывается работать с индексом, собранным другой моделью."""
        if self._manifest is None:
            return
        reason = self._manifest.incompatibility(
            self._settings.embed_provider, self._embedder.model_id
        )
        if reason:
            raise IndexMismatchError(reason)

    def _params(self, top_k: int | None, where: dict[str, Any] | None) -> RetrievalParams:
        return RetrievalParams(
            embed_model=self._embedder.model_id,
            top_k=top_k or self._settings.top_k,
            overfetch_factor=self._settings.overfetch_factor,
            relevance_threshold=self._settings.relevance_threshold,
            dedup_threshold=self._settings.dedup_threshold,
            filters=where,
            index_version=self._manifest.built_at if self._manifest else UNKNOWN_INDEX_VERSION,
        )

    def _embed(self, query: str, *, use_cache: bool) -> list[float]:
        """Вектор запроса, по возможности из кэша L1."""
        if not use_cache or self._cache is None:
            return self._embedder.embed_query(query)

        key = embedding_key(query, self._embedder.model_id)
        cached = self._cache.get_embedding(key)
        if cached is not None:
            logger.debug("Эмбеддинг: попадание в кэш L1")
            return cached

        vector = self._embedder.embed_query(query)
        self._cache.set_embedding(key, query, self._embedder.model_id, vector)
        return vector

    def _select(self, query: str, raw: Sequence[Any], params: RetrievalParams) -> RetrievalResult:
        """Порог, дедупликация и срез — в этом порядке."""
        candidates = [
            RetrievedChunk(
                chunk_id=hit.chunk_id,
                text=hit.text,
                similarity=hit.similarity,
                metadata=hit.metadata,
            )
            for hit in raw
        ]

        relevant = [chunk for chunk in candidates if chunk.similarity >= params.relevance_threshold]
        below = len(candidates) - len(relevant)

        unique, duplicates = deduplicate(relevant, params.dedup_threshold)
        selected = unique[: params.top_k]

        logger.debug(
            "Поиск: кандидатов %d, ниже порога %d, дублей %d, отобрано %d",
            len(candidates),
            below,
            duplicates,
            len(selected),
        )
        return RetrievalResult(
            query=query,
            chunks=selected,
            candidates=len(candidates),
            below_threshold=below,
            duplicates=duplicates,
        )


def _elapsed_ms(since: float) -> float:
    return (time.perf_counter() - since) * 1000
