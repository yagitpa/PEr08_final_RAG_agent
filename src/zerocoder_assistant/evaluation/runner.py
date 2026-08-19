"""
Прогон golden set.

Два режима, и по умолчанию работает дешёвый.

**Только поиск.** Вопросы векторизуются (дальше — из кэша L1) и ищутся по
индексу. Языковая модель не вызывается вовсе. Здесь считается почти всё, что
имеет смысл считать: recall@k, MRR, корректность отказа, латентность. Прогон
стоит копейки и повторяется хоть на каждый коммит.

**С ответами (`--answers`).** Дополнительно вызывается модель — ради того, что
из поиска не видно: ссылается ли она на несуществующие фрагменты и сколько
времени занимает генерация. Стоит денег, поэтому включается флагом.

Подбор порога вынесен отдельно и устроен иначе: сходства от порога не зависят,
зависит только отсечение. Поэтому корпус проходится **один раз**, а полсотни
значений порога проверяются потом, на уже собранных числах.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from zerocoder_assistant.cache.sqlite_cache import SqliteCache
from zerocoder_assistant.config.settings import Settings, get_settings
from zerocoder_assistant.evaluation.golden_set import GoldenQuestion, GoldenSet
from zerocoder_assistant.evaluation.metrics import (
    EvaluationReport,
    QuestionOutcome,
    build_report,
    lessons_of,
)
from zerocoder_assistant.observability.timing import Stopwatch
from zerocoder_assistant.retrieval.retriever import Retriever

if TYPE_CHECKING:  # pragma: no cover - только для аннотаций
    from zerocoder_assistant.generation.answerer import Answerer

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Candidate:
    """Один найденный фрагмент в виде, достаточном для подбора порога."""

    similarity: float
    lesson: str


@dataclass(frozen=True, slots=True)
class ThresholdPoint:
    """Каким был бы прогон при этом значении порога."""

    threshold: float
    report: EvaluationReport

    @property
    def recall(self) -> float:
        return self.report.recall[max(self.report.recall)]

    @property
    def score(self) -> float:
        """Доля правильных исходов на всём наборе.

        Отвечаемые и неотвечаемые вопросы вносятся в неё как есть, без весов:
        как только появляется вес, «лучший порог» начинает зависеть от того,
        какой вес выбрали, а не от данных.
        """
        wrong = len(self.report.failures)
        return (self.report.total - wrong) / self.report.total if self.report.total else 0.0


class Evaluator:
    """Прогоняет набор вопросов через настоящий конвейер."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        retriever: Retriever | None = None,
        cache: SqliteCache | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._cache = cache
        self._retriever = retriever or Retriever(self._settings, cache=cache)
        self._owns_retriever = retriever is None

    def close(self) -> None:
        if self._owns_retriever:
            self._retriever.close()

    def __enter__(self) -> Evaluator:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- прогон -------------------------------------------------------------

    def run(
        self,
        golden: GoldenSet,
        *,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
        use_cache: bool = True,
        answerer: Answerer | None = None,
    ) -> EvaluationReport:
        """Прогнать набор и собрать сводку.

        `answerer` передаётся снаружи, а не создаётся здесь: дешёвый режим не
        должен даже уметь позвать модель, иначе однажды позовёт.
        """
        outcomes = [
            self._evaluate(question, top_k, where, use_cache, answerer)
            for question in _progress(golden.questions)
        ]
        usage = self._cache.snapshot_usage() if self._cache is not None else None
        return build_report(
            outcomes,
            cache_hits=usage.hits if usage else 0,
            cache_lookups=usage.lookups if usage else 0,
        )

    def _evaluate(
        self,
        question: GoldenQuestion,
        top_k: int | None,
        where: dict[str, Any] | None,
        use_cache: bool,
        answerer: Answerer | None,
    ) -> QuestionOutcome:
        watch = Stopwatch()

        if answerer is None:
            result = self._retriever.retrieve(
                question.question, top_k=top_k, where=where, use_cache=use_cache
            )
            chunks, answer, invented = result.chunks, None, ()
            from_cache = result.from_cache
        else:
            reply = answerer.answer(
                question.question, top_k=top_k, where=where, use_cache=use_cache
            )
            chunks = reply.retrieval.chunks if reply.retrieval else []
            answer = reply.text
            invented = tuple(reply.unknown_citations)
            from_cache = reply.from_cache

        return QuestionOutcome(
            question=question,
            lessons=lessons_of(chunks),
            similarities=tuple(chunk.similarity for chunk in chunks),
            latency_ms=watch.finish()["total"],
            from_cache=from_cache,
            answer=answer,
            unknown_citations=invented,
        )

    # -- подбор порога -------------------------------------------------------

    def collect_candidates(
        self,
        golden: GoldenSet,
        *,
        top_k: int | None = None,
        use_cache: bool = True,
    ) -> dict[str, list[Candidate]]:
        """Сходства всех найденных фрагментов по каждому вопросу, без порога."""
        collected: dict[str, list[Candidate]] = {}
        for question in _progress(golden.questions):
            result = self._retriever.candidates(question.question, top_k=top_k, use_cache=use_cache)
            collected[question.id] = [
                Candidate(similarity=chunk.similarity, lesson=lesson)
                for chunk, lesson in zip(result.chunks, lessons_of(result.chunks), strict=True)
            ]
        return collected

    def sweep(
        self,
        golden: GoldenSet,
        thresholds: Sequence[float],
        *,
        top_k: int | None = None,
        use_cache: bool = True,
    ) -> list[ThresholdPoint]:
        """Каким был бы прогон при каждом из значений порога.

        Дедупликация здесь считается один раз, на полном списке кандидатов, а в
        рабочем конвейере — уже после отсечения. На высоких порогах это может
        разойтись на единичный фрагмент, поэтому подобранное значение стоит
        подтвердить обычным прогоном.
        """
        limit = top_k or self._settings.top_k
        collected = self.collect_candidates(golden, top_k=top_k, use_cache=use_cache)

        points = []
        for threshold in thresholds:
            outcomes = [
                _simulate(question, collected[question.id], threshold, limit)
                for question in golden.questions
            ]
            points.append(ThresholdPoint(threshold=threshold, report=build_report(outcomes)))
        return points


def _simulate(
    question: GoldenQuestion, candidates: Sequence[Candidate], threshold: float, top_k: int
) -> QuestionOutcome:
    """Исход вопроса при заданном пороге, посчитанный по уже собранным сходствам."""
    selected = [candidate for candidate in candidates if candidate.similarity >= threshold][:top_k]
    return QuestionOutcome(
        question=question,
        lessons=tuple(candidate.lesson for candidate in selected),
        similarities=tuple(candidate.similarity for candidate in selected),
        latency_ms=0.0,
    )


def thresholds_range(start: float, stop: float, step: float) -> list[float]:
    """Значения порога от `start` до `stop` включительно.

    Считается умножением, а не накоплением: 0.2 + 0.01 * 40 раз даёт 0.6000000001
    и лишнюю точку в отчёте.
    """
    if step <= 0:
        raise ValueError("Шаг перебора должен быть положительным")
    count = round((stop - start) / step)
    return [round(start + step * index, 4) for index in range(count + 1)]


def _progress(questions: Sequence[GoldenQuestion]) -> Iterator[GoldenQuestion]:
    """Сообщать о ходе прогона в лог: полсотни запросов идут заметное время."""
    total = len(questions)
    for position, question in enumerate(questions, 1):
        logger.debug("Оценка %d/%d: %s", position, total, question.id)
        yield question
