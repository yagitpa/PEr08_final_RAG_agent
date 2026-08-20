"""Кэш и его ключи.

Ключевое свойство, которое здесь проверяется: параметры входят в ключ. Кэш,
который этого не делает, отдаёт старый результат после смены top_k — и это не
теоретическая придирка, а сценарий из домашних заданий модуля.
"""

from __future__ import annotations

import sqlite3
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
from zerocoder_assistant.cache.sqlite_cache import SqliteCache, _add_missing_columns
from zerocoder_assistant.observability.counters import LevelUsage

BASE = RetrievalParams(
    embed_model="text-embedding-3-small",
    top_k=5,
    overfetch_factor=3,
    relevance_threshold=0.33,
    dedup_threshold=0.8,
    filters=None,
    index_version="2026-08-19T09:56:26+00:00",
)


def level_usage(cache: SqliteCache, name: str) -> LevelUsage:
    """Счётчики одного уровня кэша."""
    return next(level for level in cache.snapshot_usage().levels if level.level == name)


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
    def test_key_is_stable_across_runs(self) -> None:
        """Ключ закреплён значением, а не сравнением вызова с самим собой.

        Сравнить два вызова подряд — значит проверить, что функция
        детерминирована, чего никто и не подозревал. Смысл здесь в другом:
        формула ключа не должна меняться незаметно. Любая её правка обесценивает
        весь накопленный кэш разом, и узнать об этом надо от упавшего теста, а
        не от счёта за повторную векторизацию корпуса.

        Если правка формулы сделана осознанно — впишите новое значение и
        очистите кэш (`zassist cache clear`).
        """
        assert retrieval_key("вопрос", BASE) == "6a8454ae12679a58db0a7057a0f9b945"

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
        """Вектор запроса от top_k не зависит — значит и ключ L1 не должен.

        Прежняя редакция сравнивала один и тот же вызов с самим собой и
        утверждение из докстринга не проверяла вовсе. Проверяется оно так:
        параметры поиска меняются, ключ L2 вслед за ними уезжает, ключ L1
        остаётся на месте.
        """
        other = replace(BASE, top_k=BASE.top_k + 3, relevance_threshold=0.7)

        assert retrieval_key("вопрос", other) != retrieval_key("вопрос", BASE)
        assert embedding_key("вопрос", BASE.embed_model) == embedding_key(
            "вопрос", other.embed_model
        )

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
        cache.set_retrieval("k", "вопрос", "отпечаток", hits, 0.71)

        assert cache.get_retrieval("k") == (hits, 0.71)

    def test_retrieval_keeps_best_candidate_similarity(self, cache: SqliteCache) -> None:
        """Сходство лучшего кандидата до порога переживает кэш.

        Иначе прогон оценки по горячему кэшу считал бы разделимость классов по
        одним вопросам и не считал бы по другим — в зависимости от того, что
        успело закэшироваться.
        """
        cache.set_retrieval("k", "вопрос", "отпечаток", [], 0.88)

        assert cache.get_retrieval("k") == ([], 0.88)

    def test_retrieval_written_by_older_version_still_reads(self, cache: SqliteCache) -> None:
        """У старых записей колонки не было — это None, а не падение."""
        cache.set_retrieval("k", "вопрос", "отпечаток", [])

        assert cache.get_retrieval("k") == ([], None)

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


class TestUsageCounters:
    """Проводка счётчиков в сам кэш, а не логика счёта.

    `UsageCounters` проверен отдельно, но его подключение к трём геттерам — нет,
    и мутации, снимавшие любую из трёх проводок, проходили весь набор тестов.
    Ни одна цифра `cache usage` при этом не была ничем подкреплена.
    """

    @pytest.fixture
    def cache(self, tmp_path: Path) -> SqliteCache:
        return SqliteCache(tmp_path / "cache.db")

    def test_embedding_lookup_is_counted(self, cache: SqliteCache) -> None:
        cache.get_embedding("нет")
        cache.set_embedding("есть", "вопрос", "модель", [0.1, 0.2])
        cache.get_embedding("есть")

        level = level_usage(cache, "embeddings")

        assert (level.hits, level.misses) == (1, 1)

    def test_retrieval_lookup_is_counted(self, cache: SqliteCache) -> None:
        cache.get_retrieval("нет")
        cache.set_retrieval("есть", "вопрос", "отпечаток", [])
        cache.get_retrieval("есть")

        level = level_usage(cache, "retrieval")

        assert (level.hits, level.misses) == (1, 1)

    def test_answer_lookup_is_counted(self, cache: SqliteCache) -> None:
        cache.get_answer("нет")
        cache.set_answer("есть", "вопрос", "отпечаток", "ответ", [])
        cache.get_answer("есть")

        level = level_usage(cache, "answers")

        assert (level.hits, level.misses) == (1, 1)

    def test_levels_are_counted_separately(self, cache: SqliteCache) -> None:
        """Промах на одном уровне не портит долю попаданий на другом."""
        cache.set_embedding("k", "вопрос", "модель", [1.0])
        cache.get_embedding("k")
        cache.get_answer("k")

        usage = cache.snapshot_usage()
        by_level = {level.level: level for level in usage.levels}

        assert by_level["embeddings"].hit_rate == 1.0
        assert by_level["answers"].hit_rate == 0.0
        assert by_level["retrieval"].lookups == 0
        assert usage.hit_rate == 0.5

    def test_writes_are_not_lookups(self, cache: SqliteCache) -> None:
        """Запись — не обращение: иначе доля попаданий падала бы от наполнения."""
        cache.set_embedding("k", "вопрос", "модель", [1.0])
        cache.set_answer("k", "вопрос", "отпечаток", "ответ", [])

        assert cache.snapshot_usage().lookups == 0


class TestSchemaMigrationRace:
    """Две команды разом видят нехватку колонки — вторая не должна падать.

    Между `PRAGMA table_info` и `ALTER TABLE` есть окно. Попасть в него легко:
    прогон оценки в одном терминале и `ask --repl` в другом. Файл кэша после
    гонки в порядке, и объявлять это ошибкой команды не за что.
    """

    def test_duplicate_column_is_not_an_error(self) -> None:
        connection = _RacingConnection("duplicate column name: top_similarity")

        _add_missing_columns(connection)  # не должно поднять исключение

        assert connection.altered, "ALTER всё-таки был выполнен"

    def test_other_failures_still_surface(self) -> None:
        """Испорченный файл кэша молчать не должен."""
        connection = _RacingConnection("database disk image is malformed")

        with pytest.raises(sqlite3.OperationalError, match="malformed"):
            _add_missing_columns(connection)


class _RacingConnection:
    """Соединение, где колонки нет по PRAGMA, но ALTER её уже не добавляет."""

    def __init__(self, message: str) -> None:
        self.message = message
        self.altered = False

    def execute(self, statement: str) -> list[tuple[object, ...]]:
        if statement.startswith("PRAGMA"):
            return []
        self.altered = True
        raise sqlite3.OperationalError(self.message)
