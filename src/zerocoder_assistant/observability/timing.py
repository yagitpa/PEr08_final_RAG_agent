"""
Замер длительности этапов.

Тайминги — не украшение вывода. Выигрыш кэша либо измерен, либо о нём нельзя
ничего утверждать: «стало быстрее» на глаз не отличается от «показалось».
Поэтому разбивка по этапам возвращается вместе с результатом, а не пишется
только в лог.

Секундомер общий для поиска и генерации: обе стадии считали время вручную, и
две копии `(perf_counter() - since) * 1000` рано или поздно разъезжаются в том,
что считать за «всего».
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

#: Ключ суммарной длительности. Заполняется на `finish()`, а не суммированием
#: этапов: между этапами есть работа, которую никто не размечал, и молча терять
#: её в отчёте нельзя.
TOTAL = "total"

#: Порядок этапов в отчёте — от раннего к позднему, а не по алфавиту.
STAGE_ORDER = ("embed", "search", "context", "llm", TOTAL)


class Stopwatch:
    """Накапливает длительности размеченных этапов, в миллисекундах."""

    __slots__ = ("_marks", "_started")

    def __init__(self) -> None:
        self._started = time.perf_counter()
        self._marks: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Замерить один этап.

        Повторный вход с тем же именем складывается с прежним: этап, вызванный
        в цикле, должен показать суммарную стоимость, а не последнюю итерацию.
        """
        started = time.perf_counter()
        try:
            yield
        finally:
            self._marks[name] = self._marks.get(name, 0.0) + _ms_since(started)

    def finish(self) -> dict[str, float]:
        """Разбивка вместе с суммарным временем от создания секундомера."""
        return {**self._marks, TOTAL: _ms_since(self._started)}


def _ms_since(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def percentile(values: list[float], share: float) -> float:
    """Значение, ниже которого лежит `share` измерений (0.5 — медиана).

    Без интерполяции: на наборе из полусотни вопросов она создаёт видимость
    точности, которой в таком объёме данных нет. Берётся ближайшее реальное
    измерение — число, которое действительно наблюдалось.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(share * (len(ordered) - 1)))
    return ordered[index]
