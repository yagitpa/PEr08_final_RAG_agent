"""
Встроенные метрики качества.

Работают без RAGAS и без обращения к языковой модели: всё, что здесь считается,
выводится из результатов поиска. Это сознательный выбор — оценка, которую нельзя
запустить, потому что окружение сломалось (а RAGAS его ломает, о чём дважды
сказано в конспектах), не оценка.

Метрики разделены по классам вопросов, и это важнее, чем кажется. Одно число
«точность 82%» скрывает противоположные болезни: систему, которая находит всё
подряд и никогда не отказывает, и систему, которая молчит на половину вопросов.
Лечатся они сдвигом порога в разные стороны, поэтому и мерить их надо порознь.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from zerocoder_assistant.config.constants import REFUSAL_MARKERS
from zerocoder_assistant.evaluation.golden_set import (
    KIND_IN_DOMAIN_ABSENT,
    KIND_OUT_OF_DOMAIN,
    GoldenQuestion,
)
from zerocoder_assistant.observability.timing import percentile

if TYPE_CHECKING:  # pragma: no cover - только для аннотаций
    from zerocoder_assistant.retrieval.retriever import RetrievedChunk

#: Значения k, по которым считается recall. Больше `TOP_K` смысла не имеет:
#: в контекст всё равно уедет не больше.
RECALL_LEVELS = (1, 3, 5)

#: Метка урока, когда в метаданных фрагмента его нет.
UNKNOWN_LESSON = "?"


@dataclass(frozen=True, slots=True)
class QuestionOutcome:
    """Что случилось с одним вопросом набора."""

    question: GoldenQuestion
    lessons: tuple[str, ...]
    similarities: tuple[float, ...]
    latency_ms: float
    from_cache: bool = False
    answer: str | None = None
    unknown_citations: tuple[int, ...] = ()

    @property
    def refused(self) -> bool:
        """Поиск не отдал ничего — ассистент отвечает «в конспектах нет»."""
        return not self.lessons

    @property
    def top_similarity(self) -> float | None:
        return self.similarities[0] if self.similarities else None

    @property
    def answer_refused(self) -> bool:
        """Сказала ли модель в ответе, что этого в конспектах нет.

        Второй рубеж обороны. Первый — порог релевантности — держит только
        вопросы, далёкие от корпуса целиком; на смежной теме фрагменты находятся
        с высоким сходством, и отказать может лишь сама модель.
        """
        if self.answer is None:
            return False
        lowered = self.answer.lower()
        return any(marker in lowered for marker in REFUSAL_MARKERS)

    @property
    def hallucinated(self) -> bool:
        """Ответ на вопрос, ответа на который в конспектах нет.

        Поиск что-то отдал, и модель не оговорилась, что этого в базе не было.
        """
        return not self.question.answerable and self.answer is not None and not self.answer_refused

    @property
    def correct(self) -> bool:
        """Правильный исход с точки зрения набора.

        Для отвечаемого вопроса — найден нужный урок. Для неотвечаемого —
        отказ, причём на любом из двух рубежей: порог мог не пропустить
        фрагменты вовсе либо их пропустил, а модель сама сказала, что ответа в
        конспектах нет. Второй путь засчитывается наравне с первым, потому что
        для студента исход один и тот же.
        """
        if not self.question.answerable:
            return self.refused or self.answer_refused
        return self.hit_at(len(self.lessons))

    def hit_at(self, k: int) -> bool:
        """Попал ли ожидаемый урок в первые k отобранных фрагментов."""
        return bool(set(self.lessons[:k]) & set(self.question.lessons))

    def rank(self) -> int | None:
        """Позиция первого фрагмента из ожидаемого урока, считая с единицы."""
        for position, lesson in enumerate(self.lessons, 1):
            if lesson in self.question.lessons:
                return position
        return None


@dataclass(frozen=True, slots=True)
class RefusalStats:
    """Поведение на неотвечаемых вопросах, раздельно по их природе."""

    kind: str
    total: int
    refused: int

    @property
    def accuracy(self) -> float:
        return self.refused / self.total if self.total else 0.0

    @property
    def invented(self) -> int:
        """Сколько раз система нашла «ответ» там, где его нет."""
        return self.total - self.refused


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Сводка прогона по набору."""

    total: int
    answerable: int
    unanswerable: int
    recall: dict[int, float]
    mrr: float
    false_refusals: int
    refusals: tuple[RefusalStats, ...]
    latency_p50: float
    latency_p95: float
    cache_hits: int
    cache_lookups: int
    min_similarity_answerable: float | None
    max_similarity_unanswerable: float | None
    answered: int = 0
    with_unknown_citations: int = 0
    answer_refusals: int = 0
    hallucinations: int = 0
    unanswerable_answered: int = 0
    outcomes: tuple[QuestionOutcome, ...] = field(default_factory=tuple)

    @property
    def failures(self) -> tuple[QuestionOutcome, ...]:
        """Вопросы, на которых система повела себя не так, как ожидает набор."""
        return tuple(outcome for outcome in self.outcomes if not outcome.correct)

    @property
    def separation(self) -> float | None:
        """Зазор между «ответ есть» и «ответа нет» по сходству лучшего фрагмента.

        Отрицательный зазор означает, что классы перекрываются и **никакой**
        порог не разделит их без ошибок: правильные и посторонние вопросы дают
        одинаковое сходство. Это не повод крутить порог, это повод править
        корпус или чанкинг.
        """
        if self.min_similarity_answerable is None or self.max_similarity_unanswerable is None:
            return None
        return self.min_similarity_answerable - self.max_similarity_unanswerable

    @property
    def answer_refusal_accuracy(self) -> float:
        """Доля неотвечаемых вопросов, на которых отказала уже модель."""
        if not self.unanswerable_answered:
            return 0.0
        return self.answer_refusals / self.unanswerable_answered

    @property
    def refusal_accuracy(self) -> float:
        total = sum(stats.total for stats in self.refusals)
        refused = sum(stats.refused for stats in self.refusals)
        return refused / total if total else 0.0

    @property
    def cache_hit_rate(self) -> float:
        return self.cache_hits / self.cache_lookups if self.cache_lookups else 0.0


def lessons_of(chunks: Sequence[RetrievedChunk]) -> tuple[str, ...]:
    """Уроки отобранных фрагментов в порядке выдачи."""
    return tuple(str(chunk.metadata.get("lesson_id", UNKNOWN_LESSON)) for chunk in chunks)


def recall_at(outcomes: Sequence[QuestionOutcome], k: int) -> float:
    """Доля отвечаемых вопросов, у которых нужный урок попал в первые k."""
    relevant = [outcome for outcome in outcomes if outcome.question.answerable]
    if not relevant:
        return 0.0
    return sum(outcome.hit_at(k) for outcome in relevant) / len(relevant)


def mean_reciprocal_rank(outcomes: Sequence[QuestionOutcome]) -> float:
    """MRR: насколько высоко в выдаче стоит нужный урок.

    Дополняет recall@k там, где тот слеп: recall@5 не отличает «нужное первым»
    от «нужное пятым», а разница существенна — до пятого фрагмента внимание
    модели доходит хуже.
    """
    relevant = [outcome for outcome in outcomes if outcome.question.answerable]
    if not relevant:
        return 0.0
    total = 0.0
    for outcome in relevant:
        rank = outcome.rank()
        if rank is not None:
            total += 1 / rank
    return total / len(relevant)


def build_report(
    outcomes: Sequence[QuestionOutcome],
    *,
    cache_hits: int = 0,
    cache_lookups: int = 0,
) -> EvaluationReport:
    """Собрать сводку по исходам прогона."""
    answerable = [outcome for outcome in outcomes if outcome.question.answerable]
    unanswerable = [outcome for outcome in outcomes if not outcome.question.answerable]
    latencies = [outcome.latency_ms for outcome in outcomes]

    found = [outcome.top_similarity for outcome in answerable if outcome.top_similarity is not None]
    noise = [
        outcome.top_similarity for outcome in unanswerable if outcome.top_similarity is not None
    ]

    return EvaluationReport(
        total=len(outcomes),
        answerable=len(answerable),
        unanswerable=len(unanswerable),
        recall={level: recall_at(outcomes, level) for level in RECALL_LEVELS},
        mrr=mean_reciprocal_rank(outcomes),
        false_refusals=sum(outcome.refused for outcome in answerable),
        refusals=_refusal_stats(unanswerable),
        latency_p50=percentile(latencies, 0.5),
        latency_p95=percentile(latencies, 0.95),
        cache_hits=cache_hits,
        cache_lookups=cache_lookups,
        min_similarity_answerable=min(found) if found else None,
        max_similarity_unanswerable=max(noise) if noise else None,
        answered=sum(outcome.answer is not None for outcome in outcomes),
        with_unknown_citations=sum(bool(outcome.unknown_citations) for outcome in outcomes),
        answer_refusals=sum(
            outcome.answer_refused for outcome in unanswerable if outcome.answer is not None
        ),
        hallucinations=sum(outcome.hallucinated for outcome in outcomes),
        unanswerable_answered=sum(outcome.answer is not None for outcome in unanswerable),
        outcomes=tuple(outcomes),
    )


def _refusal_stats(unanswerable: Sequence[QuestionOutcome]) -> tuple[RefusalStats, ...]:
    stats = []
    for kind in (KIND_IN_DOMAIN_ABSENT, KIND_OUT_OF_DOMAIN):
        group = [outcome for outcome in unanswerable if outcome.question.kind == kind]
        if group:
            stats.append(
                RefusalStats(
                    kind=kind,
                    total=len(group),
                    refused=sum(outcome.refused for outcome in group),
                )
            )
    return tuple(stats)
