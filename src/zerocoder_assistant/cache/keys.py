"""
Ключи кэша.

Главное правило: **параметры входят в ключ, а не лежат рядом с ним.** Если
закешировать ответ по одному тексту запроса, то смена `top_k` или порога
релевантности вернёт старый результат, посчитанный по другим настройкам. Именно
этот сценарий домашние задания модуля просят продемонстрировать, и именно на нём
наивный кэш врёт.

Нормализация запроса намеренно скромная — только схлопывание пробелов, без
приведения регистра. Ключ первого уровня адресует эмбеддинг конкретного текста,
и если ключ нормализовать сильнее, чем сам текст перед векторизацией, то два
разных запроса разделят один вектор. Терять точность ради лишних попаданий в кэш
здесь не стоит.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

#: Длина усечённого sha256 в ключах: 32 hex-символа = 128 бит.
KEY_LENGTH = 32

_WHITESPACE = re.compile(r"\s+")


def normalize_query(text: str) -> str:
    """Каноническая форма запроса: без краевых и повторных пробелов.

    Именно эта форма и векторизуется, и адресует кэш — иначе ключ и содержимое
    разойдутся.
    """
    return _WHITESPACE.sub(" ", text).strip()


def _digest(*parts: Any) -> str:
    """Устойчивый хеш от набора значений.

    Значения сериализуются в JSON с сортировкой ключей: словарь фильтров должен
    давать один и тот же ключ независимо от порядка, в котором его собрали.
    """
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:KEY_LENGTH]


@dataclass(frozen=True, slots=True)
class RetrievalParams:
    """Всё, что влияет на результат поиска.

    `index_version` — отметка сборки индекса из манифеста. Без неё кэш пережил бы
    пересборку и продолжал отдавать фрагменты, которых в индексе уже нет.
    """

    embed_model: str
    top_k: int
    overfetch_factor: int
    relevance_threshold: float
    dedup_threshold: float
    filters: dict[str, Any] | None
    index_version: str

    def fingerprint(self) -> str:
        return _digest(
            self.embed_model,
            self.top_k,
            self.overfetch_factor,
            self.relevance_threshold,
            self.dedup_threshold,
            self.filters,
            self.index_version,
        )


def embedding_key(query: str, embed_model: str) -> str:
    """Ключ L1: вектор запроса.

    Зависит только от текста и модели — ни от top_k, ни от фильтров, ни от
    истории диалога. Поэтому кешируется всегда.
    """
    return _digest("embedding", normalize_query(query), embed_model)


def retrieval_key(query: str, params: RetrievalParams) -> str:
    """Ключ L2: результаты поиска.

    Зависит от текста и от всех параметров поиска, но не от истории диалога:
    что нашлось в базе, от предыдущих реплик не меняется.
    """
    return _digest("retrieval", normalize_query(query), params.fingerprint())


def answer_key(query: str, params: RetrievalParams, generation: dict[str, Any]) -> str:
    """Ключ L3: готовый ответ.

    Помимо параметров поиска включает всё, что влияет на генерацию: модель,
    температуру, версию промпта. Историю диалога сюда не подмешиваем — вместо
    этого ответ кешируется только при пустой истории (см. `SqliteCache`).
    """
    return _digest("answer", normalize_query(query), params.fingerprint(), generation)
