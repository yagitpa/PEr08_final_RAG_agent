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
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from zerocoder_assistant.cache.keys import RetrievalParams, embedding_key, retrieval_key
from zerocoder_assistant.cache.sqlite_cache import SqliteCache
from zerocoder_assistant.config.settings import Settings, get_settings
from zerocoder_assistant.embeddings.base import EmbeddingProvider
from zerocoder_assistant.embeddings.factory import build_embedding_provider
from zerocoder_assistant.errors import IndexMismatchError
from zerocoder_assistant.observability.timing import Stopwatch
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
        """Человекочитаемая ссылка на источник для показа пользователю.

        Номер части обязателен, когда секция разбита на несколько чанков: в
        выдаче их запросто окажется три подряд, и без номера три строки списка
        источников выглядят как одна повторённая трижды.
        """
        lesson = self.metadata.get("lesson_id")
        section = self.metadata.get("section_title") or self.metadata.get("lesson_title")
        place = f"{lesson} > {section}" if lesson else str(section or self.chunk_id)

        parts = self.metadata.get("chunks_in_section") or 1
        if parts > 1:
            index = int(self.metadata.get("chunk_index", 0)) + 1
            return f"{place} (часть {index}/{parts})"
        return place

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
    #: Сходство лучшего кандидата ДО отсечения порогом. Отобранный фрагмент
    #: по определению не бывает ниже порога, поэтому судить по нему о том,
    #: насколько запрос вообще близок к базе, нельзя: подняв порог, всегда
    #: получишь «высокое сходство» — просто у меньшего числа вопросов.
    top_candidate_similarity: float | None = None
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
        params = self.params(top_k, where)
        watch = Stopwatch()

        if use_cache and self._cache is not None:
            cached = self._cache.get_retrieval(retrieval_key(query, params))
            if cached is not None:
                hits, top_candidate = cached
                logger.debug("Поиск: попадание в кэш L2")
                return RetrievalResult(
                    query=query,
                    chunks=[RetrievedChunk.from_dict(item) for item in hits],
                    from_cache=True,
                    top_candidate_similarity=top_candidate,
                    timings_ms=watch.finish(),
                )

        with watch.stage("embed"):
            vector = self._embed(query, use_cache=use_cache)
        with watch.stage("search"):
            raw = self._store.query(vector, params.top_k * params.overfetch_factor, where=where)

        result = self._select(query, raw, params)
        timings = watch.finish()

        if use_cache and self._cache is not None:
            self._cache.set_retrieval(
                retrieval_key(query, params),
                query,
                params.fingerprint(),
                [chunk.as_dict() for chunk in result.chunks],
                result.top_candidate_similarity,
            )

        return RetrievalResult(
            query=result.query,
            chunks=result.chunks,
            candidates=result.candidates,
            below_threshold=result.below_threshold,
            duplicates=result.duplicates,
            top_candidate_similarity=result.top_candidate_similarity,
            timings_ms=timings,
        )

    def candidates(
        self,
        query: str,
        *,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> RetrievalResult:
        """Всё, что нашлось: без порога и без среза, только дедупликация.

        Нужно оценке. Отчёт о качестве обязан отвечать на вопрос «а что было бы
        при другом пороге», и получить этот ответ, прогоняя корпус заново на
        каждое значение порога, слишком дорого: сходства от порога не зависят,
        зависит только отсечение. Кэш здесь только уровня L1 — результат зависит
        от порога и в L2 ему не место.
        """
        params = self.params(top_k, where)
        watch = Stopwatch()

        with watch.stage("embed"):
            vector = self._embed(query, use_cache=use_cache)
        with watch.stage("search"):
            raw = self._store.query(vector, params.top_k * params.overfetch_factor, where=where)

        found = [
            RetrievedChunk(
                chunk_id=hit.chunk_id,
                text=hit.text,
                similarity=hit.similarity,
                metadata=hit.metadata,
            )
            for hit in raw
        ]
        unique, duplicates = deduplicate(found, params.dedup_threshold)

        return RetrievalResult(
            query=query,
            chunks=unique,
            candidates=len(found),
            duplicates=duplicates,
            top_candidate_similarity=max((chunk.similarity for chunk in found), default=None),
            timings_ms=watch.finish(),
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

    def params(self, top_k: int | None, where: dict[str, Any] | None) -> RetrievalParams:
        """Отпечаток параметров поиска.

        Публичный: тот же отпечаток входит в ключ кэша ответов, и генератору
        нужно уметь заглянуть в него до поиска — иначе попадание в L3 не сможет
        пропустить всю цепочку целиком.
        """
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
            top_candidate_similarity=max((chunk.similarity for chunk in candidates), default=None),
        )
