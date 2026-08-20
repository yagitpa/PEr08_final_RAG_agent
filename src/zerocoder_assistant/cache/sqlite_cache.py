"""
Кэш на SQLite: три уровня из PEr07 в одном файле, тремя таблицами.

| Уровень | Что хранит | Что пропускает |
|---|---|---|
| L1 | вектор запроса | обращение к API эмбеддингов |
| L2 | результаты поиска | векторизацию и поиск по базе |
| L3 | готовый ответ | всю цепочку целиком |

Уровни независимы: попадание в L2 избавляет и от L1, но промах в L3 не мешает
попасть в L2. Поэтому даже на новом вопросе, похожем по параметрам на прежний,
часть работы всё равно пропускается.

**L3 сознательно не хранит ответы, посчитанные с историей диалога.** Ответ на
«а подробнее про это?» зависит от предыдущих реплик, а ключ кэша про них ничего
не знает — сохранить такой ответ значит начать врать при следующем совпадении
формулировки. Уровни L1 и L2 от истории не зависят и кешируются всегда.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from array import array
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from zerocoder_assistant.observability.counters import (
    LEVEL_ANSWERS,
    LEVEL_EMBEDDINGS,
    LEVEL_RETRIEVAL,
    CacheUsage,
    UsageCounters,
)

logger = logging.getLogger(__name__)

#: Векторы хранятся как упакованные float32, а не как JSON: 1536 чисел занимают
#: 6 КБ вместо ~30 КБ. Точность float32 совпадает с той, что и так использует
#: векторное хранилище, так что потерь нет.
_VECTOR_TYPE = "f"

SCHEMA = """
CREATE TABLE IF NOT EXISTS query_embeddings (
    key         TEXT PRIMARY KEY,
    query       TEXT NOT NULL,
    embed_model TEXT NOT NULL,
    vector      BLOB NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS retrievals (
    key           TEXT PRIMARY KEY,
    query         TEXT NOT NULL,
    params        TEXT NOT NULL,
    hits          TEXT NOT NULL,
    top_candidate REAL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS answers (
    key        TEXT PRIMARY KEY,
    query      TEXT NOT NULL,
    params     TEXT NOT NULL,
    answer     TEXT NOT NULL,
    sources    TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

_TABLES = ("query_embeddings", "retrievals", "answers")

#: Колонки, добавленные после первой версии схемы. Кэш переживает обновление
#: кода, и запись без сходства лучшего кандидата — не повод его сбрасывать:
#: недостающее значение просто вернётся как None.
_ADDED_COLUMNS = (("retrievals", "top_candidate", "REAL"),)


@dataclass(frozen=True, slots=True)
class CacheStats:
    """Сколько записей лежит на каждом уровне."""

    embeddings: int
    retrievals: int
    answers: int
    size_bytes: int

    @property
    def total(self) -> int:
        return self.embeddings + self.retrievals + self.answers


class SqliteCache:
    """Кэш запросов, поиска и ответов."""

    def __init__(self, path: Path) -> None:
        self.path = path
        # Счётчики прогона: в SQLite следа обращений не остаётся, а «сколько
        # записей накоплено» — это про размер, а не про пользу.
        self.usage = UsageCounters()
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            connection.executescript(SCHEMA)
            _add_missing_columns(connection)
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    # -- L1: векторы запросов ---------------------------------------------

    def get_embedding(self, key: str) -> list[float] | None:
        row = self._fetch("SELECT vector FROM query_embeddings WHERE key = ?", key)
        self.usage.record(LEVEL_EMBEDDINGS, hit=row is not None)
        if row is None:
            return None
        vector = array(_VECTOR_TYPE)
        vector.frombytes(row[0])
        return list(vector)

    def set_embedding(
        self, key: str, query: str, embed_model: str, vector: Sequence[float]
    ) -> None:
        self._write(
            "INSERT OR REPLACE INTO query_embeddings (key, query, embed_model, vector, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (key, query, embed_model, array(_VECTOR_TYPE, vector).tobytes(), _now()),
        )

    # -- L2: результаты поиска --------------------------------------------

    def get_retrieval(self, key: str) -> tuple[list[dict[str, Any]], float | None] | None:
        """Фрагменты и сходство лучшего кандидата до отсечения порогом.

        Второе значение нужно оценке: судить о близости запроса к базе по
        отобранным фрагментам нельзя, они по определению не бывают ниже порога.
        У записей, сделанных прежней версией, его нет — тогда None.
        """
        row = self._fetch("SELECT hits, top_candidate FROM retrievals WHERE key = ?", key)
        self.usage.record(LEVEL_RETRIEVAL, hit=row is not None)
        return (json.loads(row[0]), row[1]) if row else None

    def set_retrieval(
        self,
        key: str,
        query: str,
        params: str,
        hits: Sequence[dict[str, Any]],
        top_candidate: float | None = None,
    ) -> None:
        self._write(
            "INSERT OR REPLACE INTO retrievals"
            " (key, query, params, hits, top_candidate, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                key,
                query,
                params,
                json.dumps(list(hits), ensure_ascii=False),
                top_candidate,
                _now(),
            ),
        )

    # -- L3: готовые ответы ------------------------------------------------

    def get_answer(self, key: str) -> tuple[str, list[str]] | None:
        row = self._fetch("SELECT answer, sources FROM answers WHERE key = ?", key)
        self.usage.record(LEVEL_ANSWERS, hit=row is not None)
        return (row[0], json.loads(row[1])) if row else None

    def set_answer(
        self, key: str, query: str, params: str, answer: str, sources: Sequence[str]
    ) -> None:
        self._write(
            "INSERT OR REPLACE INTO answers (key, query, params, answer, sources, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (key, query, params, answer, json.dumps(list(sources), ensure_ascii=False), _now()),
        )

    # -- обслуживание ------------------------------------------------------

    def stats(self) -> CacheStats:
        with closing(self._connect()) as connection:
            counts = {
                table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in _TABLES
            }
        return CacheStats(
            embeddings=counts["query_embeddings"],
            retrievals=counts["retrievals"],
            answers=counts["answers"],
            size_bytes=self.path.stat().st_size if self.path.exists() else 0,
        )

    def snapshot_usage(self) -> CacheUsage:
        """Попадания и промахи с начала прогона."""
        return self.usage.snapshot()

    def clear(self, level: str | None = None) -> int:
        """Очищает кэш целиком или один уровень. Возвращает число удалённых записей."""
        tables = _TABLES if level is None else (_resolve_table(level),)
        removed = 0
        with closing(self._connect()) as connection:
            for table in tables:
                removed += connection.execute(f"DELETE FROM {table}").rowcount
            connection.commit()
            connection.execute("VACUUM")
        logger.info("Из кэша удалено записей: %d", removed)
        return removed

    # -- низкий уровень ----------------------------------------------------

    def _fetch(self, sql: str, key: str) -> tuple[Any, ...] | None:
        with closing(self._connect()) as connection:
            return connection.execute(sql, (key,)).fetchone()

    def _write(self, sql: str, values: tuple[Any, ...]) -> None:
        with closing(self._connect()) as connection:
            connection.execute(sql, values)
            connection.commit()


#: Понятные имена уровней для команды `cache clear --level`. Имена те же, что
#: в счётчиках попаданий: уровень должен называться одинаково и в очистке, и в
#: отчёте, иначе их не сопоставить глазами.
LEVEL_TABLES = {
    LEVEL_EMBEDDINGS: "query_embeddings",
    LEVEL_RETRIEVAL: "retrievals",
    LEVEL_ANSWERS: "answers",
}


def _resolve_table(level: str) -> str:
    table = LEVEL_TABLES.get(level)
    if table is None:
        raise ValueError(f"Неизвестный уровень кэша {level!r}. Доступны: {', '.join(LEVEL_TABLES)}")
    return table


def _add_missing_columns(connection: sqlite3.Connection) -> None:
    """Дописать колонки, появившиеся после создания файла кэша."""
    for table, column, kind in _ADDED_COLUMNS:
        existing = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            logger.info("Кэш: добавляю колонку %s.%s", table, column)
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
