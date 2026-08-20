"""
Учёт попаданий в кэш.

`cache stats` показывает, сколько записей НАКОПЛЕНО, — это про размер, а не про
пользу. Кэш из тысячи записей, в который никто не попадает, выглядит по этой
команде отлично. Полезность измеряется долей попаданий за прогон, и считать её
надо в процессе, потому что в SQLite такого следа не остаётся.

Счётчики живут в объекте кэша и умирают вместе с ним: одна команда CLI — один
прогон. Для `eval run` этого достаточно, там весь набор идёт через один объект.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

#: Имена уровней. Те же, что у `cache clear --level`, чтобы в отчёте и в команде
#: уровень назывался одинаково.
LEVEL_EMBEDDINGS = "embeddings"
LEVEL_RETRIEVAL = "retrieval"
LEVEL_ANSWERS = "answers"

LEVELS = (LEVEL_EMBEDDINGS, LEVEL_RETRIEVAL, LEVEL_ANSWERS)


@dataclass(frozen=True, slots=True)
class LevelUsage:
    """Обращения к одному уровню кэша за прогон."""

    level: str
    hits: int
    misses: int

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Доля попаданий. Ноль обращений — ноль, а не деление на ноль."""
        return self.hits / self.lookups if self.lookups else 0.0


@dataclass(frozen=True, slots=True)
class CacheUsage:
    """Попадания по всем уровням."""

    levels: tuple[LevelUsage, ...]

    @property
    def hits(self) -> int:
        return sum(level.hits for level in self.levels)

    @property
    def lookups(self) -> int:
        return sum(level.lookups for level in self.levels)

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0


class UsageCounters:
    """Считает попадания и промахи по уровням кэша."""

    __slots__ = ("_hits", "_misses")

    def __init__(self) -> None:
        self._hits: Counter[str] = Counter()
        self._misses: Counter[str] = Counter()

    def record(self, level: str, *, hit: bool) -> None:
        (self._hits if hit else self._misses)[level] += 1

    def reset(self) -> None:
        self._hits.clear()
        self._misses.clear()

    def snapshot(self) -> CacheUsage:
        return CacheUsage(
            levels=tuple(
                LevelUsage(level=level, hits=self._hits[level], misses=self._misses[level])
                for level in LEVELS
            )
        )
