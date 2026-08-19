"""Кэш и его ключи.

Ключевое свойство, которое здесь проверяется: параметры входят в ключ. Кэш,
который этого не делает, отдаёт старый результат после смены top_k — и это не
теоретическая придирка, а сценарий из домашних заданий модуля.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from zerocoder_assistant.cache.keys import (
    RetrievalParams,
    answer_key,
    embedding_key,
    normalize_query,
    retrieval_key,
)
from zerocoder_assistant.cache.sqlite_cache import SqliteCache

BASE = RetrievalParams(
    embed_model="text-embedding-3-small",
    top_k=5,
    overfetch_factor=3,
    relevance_threshold=0.33,
    dedup_threshold=0.8,
    filters=None,
    index_version="2026-08-19T09:56:26+00:00",
)


class TestNormalizeQuery:
    def test_collapses_whitespace(self) -> None:
        assert normalize_query("  что   такое\nRAG?  ") == "что такое RAG?"

    def test_case_preserved(self) -> None:
        """Регистр не трогаем: ключ L1 адресует вектор конкретного текста.

        Если нормализовать ключ сильнее, чем текст перед векторизацией, два
        разных запроса разделят один вектор.
        """
        assert normalize_query("Что такое RAG") != normalize_query("что такое rag")


class TestKeys:
    def test_same_query_same_key(self) -> None:
        assert retrieval_key("вопрос", BASE) == retrieval_key("вопрос", BASE)

    def test_whitespace_does_not_change_key(self) -> None:
        assert retrieval_key("вопрос  тут", BASE) == retrieval_key(" вопрос тут ", BASE)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("top_k", 3),
            ("overfetch_factor", 5),
            ("relevance_threshold", 0.5),
            ("dedup_threshold", 0.9),
            ("index_version", "другая-сборка"),
            ("embed_model", "text-embedding-3-large"),
        ],
    )
    def test_every_parameter_changes_the_key(self, field: str, value: object) -> None:
        """Иначе смена настройки вернула бы результат, посчитанный по старым."""
        other = replace(BASE, **{field: value})

        assert retrieval_key("вопрос", other) != retrieval_key("вопрос", BASE)

    def test_filters_change_the_key(self) -> None:
        filtered = replace(BASE, filters={"lesson_id": "PEr06"})

        assert retrieval_key("вопрос", filtered) != retrieval_key("вопрос", BASE)

    def test_filter_key_order_does_not_matter(self) -> None:
        """Словарь фильтров, собранный в разном порядке, — это один и тот же фильтр."""
        first = replace(BASE, filters={"a": 1, "b": 2})
        second = replace(BASE, filters={"b": 2, "a": 1})

        assert retrieval_key("вопрос", first) == retrieval_key("вопрос", second)

    def test_embedding_key_ignores_search_params(self) -> None:
        """Вектор запроса от top_k не зависит — значит и ключ L1 не должен."""
        assert embedding_key("вопрос", "модель") == embedding_key("вопрос", "модель")

    def test_embedding_key_depends_on_model(self) -> None:
        assert embedding_key("вопрос", "модель-а") != embedding_key("вопрос", "модель-б")

    def test_levels_do_not_collide(self) -> None:
        """Три уровня живут в разных таблицах, но ключи всё равно не должны совпадать."""
        keys = {
            embedding_key("вопрос", BASE.embed_model),
            retrieval_key("вопрос", BASE),
            answer_key("вопрос", BASE, {"model": "gpt-4o-mini"}),
        }

        assert len(keys) == 3

    def test_answer_key_depends_on_generation_settings(self) -> None:
        first = answer_key("вопрос", BASE, {"model": "gpt-4o-mini", "prompt": "v1"})
        second = answer_key("вопрос", BASE, {"model": "gpt-4o-mini", "prompt": "v2"})

        assert first != second


class TestSqliteCache:
    @pytest.fixture
    def cache(self, tmp_path: Path) -> SqliteCache:
        return SqliteCache(tmp_path / "cache.db")

    def test_embedding_round_trip(self, cache: SqliteCache) -> None:
        vector = [0.125, -0.5, 0.75]
        cache.set_embedding("k", "вопрос", "модель", vector)

        assert cache.get_embedding("k") == pytest.approx(vector)

    def test_large_vector_round_trip(self, cache: SqliteCache) -> None:
        """Реальный вектор — 1536 чисел; проверяем, что упаковка их не портит."""
        vector = [i / 1000 for i in range(1536)]
        cache.set_embedding("k", "вопрос", "модель", vector)

        assert cache.get_embedding("k") == pytest.approx(vector, abs=1e-6)

    def test_missing_keys_return_none(self, cache: SqliteCache) -> None:
        assert cache.get_embedding("нет") is None
        assert cache.get_retrieval("нет") is None
        assert cache.get_answer("нет") is None

    def test_retrieval_round_trip(self, cache: SqliteCache) -> None:
        hits = [{"chunk_id": "a", "text": "текст", "similarity": 0.5, "metadata": {"x": 1}}]
        cache.set_retrieval("k", "вопрос", "отпечаток", hits)

        assert cache.get_retrieval("k") == hits

    def test_answer_round_trip(self, cache: SqliteCache) -> None:
        cache.set_answer("k", "вопрос", "отпечаток", "ответ", ["PEr07 > Кеширование"])

        assert cache.get_answer("k") == ("ответ", ["PEr07 > Кеширование"])

    def test_cyrillic_survives(self, cache: SqliteCache) -> None:
        cache.set_answer("k", "вопрос", "п", "Кэш хранит ответы", ["источник"])
        answer, sources = cache.get_answer("k")

        assert answer == "Кэш хранит ответы"
        assert sources == ["источник"]

    def test_overwrite_by_key(self, cache: SqliteCache) -> None:
        cache.set_answer("k", "вопрос", "п", "первый", [])
        cache.set_answer("k", "вопрос", "п", "второй", [])

        assert cache.get_answer("k")[0] == "второй"

    def test_stats_counts_levels(self, cache: SqliteCache) -> None:
        cache.set_embedding("a", "q", "m", [1.0])
        cache.set_retrieval("b", "q", "p", [])
        cache.set_answer("c", "q", "p", "a", [])
        stats = cache.stats()

        assert (stats.embeddings, stats.retrievals, stats.answers) == (1, 1, 1)
        assert stats.total == 3

    def test_clear_one_level(self, cache: SqliteCache) -> None:
        cache.set_embedding("a", "q", "m", [1.0])
        cache.set_answer("c", "q", "p", "a", [])
        cache.clear(level="answers")
        stats = cache.stats()

        assert stats.embeddings == 1
        assert stats.answers == 0

    def test_clear_everything(self, cache: SqliteCache) -> None:
        cache.set_embedding("a", "q", "m", [1.0])
        cache.set_retrieval("b", "q", "p", [])
        cache.clear()

        assert cache.stats().total == 0

    def test_unknown_level_is_explicit(self, cache: SqliteCache) -> None:
        with pytest.raises(ValueError, match="Неизвестный уровень"):
            cache.clear(level="выдуманный")

    def test_survives_reopen(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.db"
        SqliteCache(path).set_answer("k", "q", "p", "ответ", [])

        assert SqliteCache(path).get_answer("k")[0] == "ответ"
