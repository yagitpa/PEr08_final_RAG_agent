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
        self._given_retriever = retriever
        self._own_retriever: Retriever | None = None

    @property
    def retriever(self) -> Retriever:
        """Поиск создаётся при первом обращении.

        В режиме `--answers` работает поиск внутри генератора, а собственный не
        нужен вовсе: создавать провайдер эмбеддингов и второй клиент Chroma на
        тот же каталог ради неиспользуемого объекта незачем.
        """
        if self._given_retriever is not None:
            return self._given_retriever
        if self._own_retriever is None:
            self._own_retriever = Retriever(self._settings, cache=self._cache)
        return self._own_retriever

    def close(self) -> None:
        if self._own_retriever is not None:
            self._own_retriever.close()
            self._own_retriever = None

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
        use_answer_cache: bool = True,
        answerer: Answerer | None = None,
    ) -> EvaluationReport:
        """Прогнать набор и собрать сводку.

        `answerer` передаётся снаружи, а не создаётся здесь: дешёвый режим не
        должен даже уметь позвать модель, иначе однажды позовёт.
        """
        outcomes = [
            self._evaluate(question, top_k, where, use_cache, use_answer_cache, answerer)
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
        use_answer_cache: bool,
        answerer: Answerer | None,
    ) -> QuestionOutcome:
        watch = Stopwatch()

        if answerer is None:
            result = self.retriever.retrieve(
                question.question, top_k=top_k, where=where, use_cache=use_cache
            )
            chunks, answer, invented, generated = result.chunks, None, (), False
            from_cache = result.from_cache
            best = result.top_candidate_similarity
            contexts: tuple[str, ...] = ()
        else:
            reply = answerer.answer(
                question.question,
                top_k=top_k,
                where=where,
                use_cache=use_cache,
                use_answer_cache=use_answer_cache,
            )
            retrieval = reply.retrieval
            chunks = retrieval.chunks if retrieval else []
            answer = reply.text
            invented = tuple(reply.unknown_citations)
            from_cache = reply.from_cache
            # `grounded` ложно ровно тогда, когда выдача пуста и модель не
            # вызывалась: подставленный текст отказа не должен засчитываться
            # второму рубежу как его работа.
            generated = reply.grounded
            best = retrieval.top_candidate_similarity if retrieval else None
            contexts = tuple(chunk.text for chunk in chunks)

        return QuestionOutcome(
            question=question,
            lessons=lessons_of(chunks),
            similarities=tuple(chunk.similarity for chunk in chunks),
            latency_ms=watch.finish()["total"],
            from_cache=from_cache,
            answer=answer,
            generated=generated,
            unknown_citations=invented,
            best_similarity=best,
            contexts=contexts,
        )

    # -- подбор порога -------------------------------------------------------

    def collect_candidates(
        self,
        golden: GoldenSet,
        *,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> dict[str, list[Candidate]]:
        """Сходства всех найденных фрагментов по каждому вопросу, без порога."""
        collected: dict[str, list[Candidate]] = {}
        for question in _progress(golden.questions):
            result = self.retriever.candidates(
                question.question, top_k=top_k, where=where, use_cache=use_cache
            )
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
        where: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> list[ThresholdPoint]:
        """Каким был бы прогон при каждом из значений порога.

        Порядок операций сохраняется точно, а не приблизительно. Хранилище
        отдаёт кандидатов по убыванию сходства, дедупликация — жадный проход
        слева направо, оставляющий более похожего представителя группы. Значит
        отсечение по порогу — это взятие префикса, а дедупликация префикса
        совпадает с префиксом дедупликации. Настоящая предпосылка здесь —
        отсортированность выдачи; на неотсортированном входе тождество ломается.
        """
        limit = top_k or self._settings.top_k
        collected = self.collect_candidates(golden, top_k=top_k, where=where, use_cache=use_cache)

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
        # Лучший кандидат берётся ДО отсечения — иначе разделимость классов
        # менялась бы вместе с порогом, который она же и оценивает.
        best_similarity=max((candidate.similarity for candidate in candidates), default=None),
    )


def thresholds_range(start: float, stop: float, step: float) -> list[float]:
    """Значения порога от `start` до `stop` включительно.

    Считается умножением, а не накоплением: 0.2 + 0.01 * 40 раз даёт 0.6000000001
    и лишнюю точку в отчёте.
    """
    if step <= 0:
        raise ValueError("Шаг перебора должен быть положительным")
    if stop < start:
        raise ValueError(f"Верхняя граница перебора ({stop}) ниже нижней ({start})")
    # Округление вниз, а не к ближайшему: при round() остаток больше полушага
    # давал точку ВЫШЕ stop, хотя докстринг обещает диапазон включительно.
    count = int((stop - start) / step + 1e-9)
    return [round(start + step * index, 4) for index in range(count + 1)]


def _progress(questions: Sequence[GoldenQuestion]) -> Iterator[GoldenQuestion]:
    """Сообщать о ходе прогона в лог: полсотни запросов идут заметное время."""
    total = len(questions)
    for position, question in enumerate(questions, 1):
        logger.debug("Оценка %d/%d: %s", position, total, question.id)
        yield question
