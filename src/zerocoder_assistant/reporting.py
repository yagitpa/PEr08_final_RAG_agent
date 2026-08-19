"""
Отчётность по препроцессингу: агрегаты и их текстовое представление.

Этап 1 проверяется глазами — «гистограмма токенов в коридоре, десять случайных
чанков осмысленны» (см. docs/architecture.md, «Верификация»). Чтобы эта проверка
была воспроизводимой, счёт и рендер вынесены сюда: здесь нет ни click, ни чтения
файлов, ни печати — только чистые функции над готовыми `ProcessedNote`. Их можно
позвать из теста, из другой команды CLI и позже из `index stats`, не поднимая
пайплайн целиком.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from zerocoder_assistant.config.constants import (
    CONTENT_TYPE_ACTUALIZATION,
    CONTENT_TYPE_CODE,
    CONTENT_TYPE_PRACTICE,
    CONTENT_TYPE_SUMMARY,
    CONTENT_TYPE_THEORY,
)

if TYPE_CHECKING:  # pragma: no cover - только для аннотаций
    from collections.abc import Iterable, Sequence

    from zerocoder_assistant.config.settings import ChunkingConfig
    from zerocoder_assistant.preprocessing.models import Chunk, ProcessedNote

# ---------------------------------------------------------------------------
# Параметры представления
# ---------------------------------------------------------------------------

#: Число интервалов гистограммы по умолчанию: при коридоре 80-500 токенов даёт
#: шаг порядка 35 токенов — достаточно мелко, чтобы увидеть провал у границ.
DEFAULT_HISTOGRAM_BINS: Final[int] = 12

#: Ширина полосы в символах. Вместе с подписью интервала укладывается в 80 колонок.
DEFAULT_HISTOGRAM_WIDTH: Final[int] = 44

BAR_CHAR: Final[str] = "#"

#: Сколько символов текста чанка показывать в примере.
DEFAULT_SAMPLE_CHARS: Final[int] = 400

#: Сколько самых частых пропущенных секций перечислять в сводке.
TOP_SKIPPED_SECTIONS: Final[int] = 10

#: Сколько примеров повторяющихся chunk_id показывать: список нужен только чтобы
#: было с чего начать поиск причины, а не чтобы перечислить все коллизии.
TOP_DUPLICATE_IDS: Final[int] = 5

#: Ширина колонки подписей в сводке — выравнивает числа в столбик.
LABEL_WIDTH: Final[int] = 22

#: Отступ для многострочных фрагментов (текст чанка, вложенные списки).
INDENT: Final[str] = "  "

ELLIPSIS: Final[str] = "..."

EMPTY_HISTOGRAM_LINE: Final[str] = "(чанков нет - гистограмма не строится)"

#: Порядок типов содержимого в отчёте: от самого массового к самому редкому по
#: корпусу. Фиксирован намеренно — сортировка по частоте меняла бы порядок строк
#: от прогона к прогону и мешала бы сравнивать два отчёта глазами.
CONTENT_TYPE_ORDER: Final[tuple[str, ...]] = (
    CONTENT_TYPE_THEORY,
    CONTENT_TYPE_PRACTICE,
    CONTENT_TYPE_CODE,
    CONTENT_TYPE_SUMMARY,
    CONTENT_TYPE_ACTUALIZATION,
)


# ---------------------------------------------------------------------------
# Агрегат
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CorpusStats:
    """Сводка по корпусу обработанных конспектов.

    Хранит и посчитанные метрики, и сами размеры чанков: гистограмму строит
    отдельная функция, и таскать ради неё список токенов параллельно со сводкой
    неудобно.
    """

    files: int
    chunks: int
    total_tokens: int
    min_tokens: int
    median_tokens: float
    mean_tokens: float
    max_tokens: int
    token_counts: tuple[int, ...]
    by_content_type: dict[str, int]
    undersized: int
    oversized: int
    skipped_sections: list[tuple[str, int]]
    dropped_boilerplate: int
    duplicate_chunk_ids: list[tuple[str, int]]

    @property
    def out_of_range(self) -> int:
        """Сколько чанков не попало в коридор [min_tokens, max_tokens]."""
        return self.undersized + self.oversized

    @property
    def out_of_range_pct(self) -> float:
        return _pct(self.out_of_range, self.chunks)


def build_corpus_stats(notes: Sequence[ProcessedNote], config: ChunkingConfig) -> CorpusStats:
    """Собрать сводку по результатам препроцессинга.

    Коридор берётся из конфигурации, а не из констант отчёта: сводка должна
    оценивать чанкинг по тем правилам, по которым он в этом прогоне и работал.
    """
    token_counts = tuple(chunk.token_count for chunk in _iter_chunks(notes))
    content_types = Counter(chunk.content_type for chunk in _iter_chunks(notes))
    skipped = Counter(title for note in notes for title in note.skipped_sections)
    chunk_ids = Counter(chunk.chunk_id for chunk in _iter_chunks(notes))

    return CorpusStats(
        files=len(notes),
        chunks=len(token_counts),
        total_tokens=sum(token_counts),
        min_tokens=min(token_counts, default=0),
        median_tokens=statistics.median(token_counts) if token_counts else 0.0,
        mean_tokens=statistics.fmean(token_counts) if token_counts else 0.0,
        max_tokens=max(token_counts, default=0),
        token_counts=token_counts,
        by_content_type=_ordered_content_types(content_types),
        undersized=sum(1 for value in token_counts if value < config.min_tokens),
        oversized=sum(1 for value in token_counts if value > config.max_tokens),
        skipped_sections=skipped.most_common(TOP_SKIPPED_SECTIONS),
        dropped_boilerplate=sum(note.dropped_boilerplate for note in notes),
        duplicate_chunk_ids=[(key, count) for key, count in chunk_ids.most_common() if count > 1],
    )


# ---------------------------------------------------------------------------
# Рендеринг
# ---------------------------------------------------------------------------


def render_histogram(
    token_counts: Sequence[int],
    *,
    bins: int = DEFAULT_HISTOGRAM_BINS,
    width: int = DEFAULT_HISTOGRAM_WIDTH,
) -> str:
    """ASCII-гистограмма распределения размеров чанков.

    Полоса масштабируется к самому населённому интервалу, а не к общему числу
    чанков: иначе на реальном распределении все столбцы схлопываются в точку.
    """
    values = list(token_counts)
    if not values:
        return EMPTY_HISTOGRAM_LINE

    bins = max(1, bins)
    width = max(1, width)
    low, high = min(values), max(values)

    if low == high:
        # Вырожденный случай: коридор нулевой ширины делить на интервалы нечем.
        edges = [(float(low), float(high))]
        buckets = [len(values)]
    else:
        step = (high - low) / bins
        buckets = [0] * bins
        for value in values:
            # Правая граница включается в последний интервал, а не создаёт bins+1-й.
            index = min(int((value - low) / step), bins - 1)
            buckets[index] += 1
        edges = [(low + step * i, low + step * (i + 1)) for i in range(bins)]

    labels = [f"{round(lo)}-{round(hi)}" for lo, hi in edges]
    label_width = max(len(label) for label in labels)
    count_width = len(str(max(buckets)))
    peak = max(buckets)

    lines = []
    for label, count in zip(labels, buckets, strict=True):
        bar = BAR_CHAR * _bar_length(count, peak, width)
        lines.append(f"{label:>{label_width}} | {bar:<{width}} {count:>{count_width}}")
    return "\n".join(lines)


def render_stats(stats: CorpusStats, config: ChunkingConfig) -> str:
    """Человекочитаемая сводка по корпусу."""
    lines = [
        _field("Файлов обработано", stats.files),
        _field("Чанков получено", stats.chunks),
        _field("Токенов всего", _thousands(stats.total_tokens)),
    ]

    if stats.chunks:
        lines.append(
            _field(
                "Токенов на чанк",
                f"min {stats.min_tokens} | медиана {stats.median_tokens:.0f} | "
                f"среднее {stats.mean_tokens:.1f} | max {stats.max_tokens}",
            )
        )
        lines.append(
            _field(
                f"Коридор {config.min_tokens}-{config.max_tokens}",
                f"вне коридора {stats.out_of_range} ({stats.out_of_range_pct:.1f}%): "
                f"мельче {stats.undersized}, крупнее {stats.oversized}",
            )
        )

    if stats.by_content_type:
        lines.append("Типы содержимого:")
        lines.extend(
            _rows(
                (name, f"{count:>5}  ({_pct(count, stats.chunks):.1f}%)")
                for name, count in stats.by_content_type.items()
            )
        )

    lines.append(_field("Удалено шаблонов", stats.dropped_boilerplate))

    if stats.skipped_sections:
        lines.append(f"Пропущенные секции (топ-{TOP_SKIPPED_SECTIONS}):")
        lines.extend(_rows((title, f"{count:>5}") for title, count in stats.skipped_sections))
    else:
        lines.append(_field("Пропущенные секции", "нет"))

    # Коллизия chunk_id при upsert не падает, а молча затирает чужой чанк, так
    # что заметить её можно только здесь — до того, как индекс собран.
    if stats.duplicate_chunk_ids:
        lines.append(f"ВНИМАНИЕ: повторяющихся chunk_id: {len(stats.duplicate_chunk_ids)}")
        lines.extend(
            _rows(
                (chunk_id, f"x{count}")
                for chunk_id, count in stats.duplicate_chunk_ids[:TOP_DUPLICATE_IDS]
            )
        )

    return "\n".join(lines)


def render_chunk_sample(chunk: Chunk, *, max_chars: int = DEFAULT_SAMPLE_CHARS) -> str:
    """Показать один чанк: паспорт плюс усечённый текст.

    Путь заголовков отдельной строкой не печатается: текст чанка и так начинается
    с contextual header, и дублировать его — только зашумлять вывод. Показывается
    ровно то, что уйдёт в эмбеддер.
    """
    header = (
        f"[{chunk.chunk_id}] {chunk.content_type} | {chunk.token_count} ток. | "
        f"чанк {chunk.chunk_index + 1}/{chunk.chunks_in_section} в секции"
    )
    body = _indent(_truncate(chunk.text, max_chars))
    return f"{header}\n{body}"


# ---------------------------------------------------------------------------
# Вспомогательное
# ---------------------------------------------------------------------------


def _iter_chunks(notes: Sequence[ProcessedNote]) -> Iterable[Chunk]:
    for note in notes:
        yield from note.chunks


def _ordered_content_types(counter: Counter[str]) -> dict[str, int]:
    """Известные типы в каноническом порядке, неизвестные — следом по алфавиту."""
    known = [name for name in CONTENT_TYPE_ORDER if counter.get(name)]
    unknown = sorted(set(counter) - set(CONTENT_TYPE_ORDER))
    return {name: counter[name] for name in (*known, *unknown)}


def _bar_length(count: int, peak: int, width: int) -> int:
    """Длина полосы. Непустой интервал никогда не рисуется нулём символов."""
    if count == 0:
        return 0
    return max(1, round(count / peak * width))


def _pct(part: int, total: int) -> float:
    return 100.0 * part / total if total else 0.0


def _thousands(value: int) -> str:
    """Разряды разделяются пробелом: 1234567 -> 1 234 567."""
    return f"{value:,}".replace(",", " ")


def _field(label: str, value: object) -> str:
    return f"{label + ':':<{LABEL_WIDTH}} {value}"


def _rows(pairs: Iterable[tuple[str, str]]) -> list[str]:
    """Вложенный столбик «подпись — значение».

    Ширина колонки считается по самой длинной подписи, а не берётся из константы:
    названия секций конспектов длиннее любого разумного значения по умолчанию, и
    при фиксированной ширине столбец с числами разъезжается.
    """
    rows = list(pairs)
    if not rows:
        return []
    width = max(LABEL_WIDTH - len(INDENT), *(len(name) for name, _ in rows))
    return [f"{INDENT}{name:<{width}} {value}" for name, value in rows]


def _truncate(text: str, max_chars: int) -> str:
    stripped = text.strip()
    if max_chars <= 0 or len(stripped) <= max_chars:
        return stripped
    return stripped[:max_chars].rstrip() + ELLIPSIS


def _indent(text: str) -> str:
    return "\n".join(f"{INDENT}{line}" if line else "" for line in text.splitlines())
