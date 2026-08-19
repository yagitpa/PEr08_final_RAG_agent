"""
Дедупликация результатов поиска.

Зачем это нужно именно здесь. Чанкинг использует перекрытие в 15%: конец одного
фрагмента повторяется в начале следующего. Значит соседние чанки одной секции
гарантированно похожи, и по запросу, попавшему в зону перекрытия, поиск вернёт
их оба. Без дедупликации половина контекста, отданного модели, — повторы, за
которые платится токенами и вниманием модели.

Сравнение идёт по шинглам, а не по множеству слов. Два фрагмента об одном и том
же предмете имеют почти одинаковый словарь, но разный порядок слов; перекрытие
же копирует текст дословно. Шинглы ловят именно дословность.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol, TypeVar

#: Длина шингла в словах. Пять — компромисс: короче начинает срабатывать на
#: устойчивых оборотах вроде «в этом случае мы можем», длиннее перестаёт видеть
#: перекрытие в одно предложение.
SHINGLE_SIZE = 5

_WORD = re.compile(r"\w+", flags=re.UNICODE)


class _Comparable(Protocol):
    """Минимум, который нужен дедупликации от результата поиска."""

    chunk_id: str
    text: str
    similarity: float


#: Синтаксис дженериков PEP 695 (`def f[T](...)`) появился в 3.12, а проект
#: объявлен совместимым с 3.11 — поэтому обычный TypeVar.
_Hit = TypeVar("_Hit", bound=_Comparable)


def shingles(text: str, size: int = SHINGLE_SIZE) -> set[tuple[str, ...]]:
    """Множество последовательностей из `size` слов подряд."""
    words = [word.casefold() for word in _WORD.findall(text)]
    if len(words) < size:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + size]) for i in range(len(words) - size + 1)}


def jaccard(left: set[tuple[str, ...]], right: set[tuple[str, ...]]) -> float:
    """Доля общих шинглов: 1.0 — тексты дословно совпадают, 0.0 — ничего общего."""
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    return intersection / (len(left) + len(right) - intersection)


def deduplicate(hits: Sequence[_Hit], threshold: float) -> tuple[list[_Hit], int]:
    """Оставляет по одному фрагменту из каждой группы похожих.

    Порядок входа считается порядком убывания релевантности, поэтому из пары
    похожих остаётся тот, что найден раньше — то есть более релевантный.

    Возвращает отобранные фрагменты и число отброшенных.
    """
    kept: list[_Hit] = []
    kept_shingles: list[set[tuple[str, ...]]] = []
    seen_texts: set[str] = set()
    removed = 0

    for hit in hits:
        # Точный дубль отсекается дёшево, без разбора на шинглы.
        normalized = " ".join(_WORD.findall(hit.text.casefold()))
        if normalized in seen_texts:
            removed += 1
            continue

        current = shingles(hit.text)
        if any(jaccard(current, previous) >= threshold for previous in kept_shingles):
            removed += 1
            continue

        kept.append(hit)
        kept_shingles.append(current)
        seen_texts.add(normalized)

    return kept, removed
