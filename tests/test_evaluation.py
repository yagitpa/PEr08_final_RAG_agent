"""
Оценка: разбор golden set, метрики, подбор порога.

Метрики проверяются на сконструированных исходах, а не на живом индексе:
recall должен считаться одинаково независимо от того, что сегодня в базе.
Отдельно прогоняется настоящий набор из docs/ — он часть репозитория и обязан
оставаться валидным.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from zerocoder_assistant.evaluation import (
    DEFAULT_GOLDEN_SET,
    GoldenQuestion,
    GoldenSet,
    GoldenSetError,
    QuestionOutcome,
    build_report,
    mean_reciprocal_rank,
    recall_at,
    thresholds_range,
)
from zerocoder_assistant.evaluation.runner import Candidate, _simulate
from zerocoder_assistant.observability import Stopwatch, UsageCounters, percentile

PROJECT_GOLDEN_SET = Path(__file__).resolve().parents[1] / DEFAULT_GOLDEN_SET


def question(
    identifier: str = "q001",
    *,
    lessons: tuple[str, ...] = ("PEr03",),
    answerable: bool = True,
    kind: str | None = None,
) -> GoldenQuestion:
    return GoldenQuestion(
        id=identifier,
        question="вопрос",
        lessons=lessons,
        answerable=answerable,
        kind=kind,
    )


def outcome(
    found: tuple[str, ...],
    *,
    expected: tuple[str, ...] = ("PEr03",),
    answerable: bool = True,
    kind: str | None = None,
    similarities: tuple[float, ...] | None = None,
    answer: str | None = None,
) -> QuestionOutcome:
    return QuestionOutcome(
        question=question(lessons=expected, answerable=answerable, kind=kind),
        lessons=found,
        similarities=similarities or tuple(0.5 for _ in found),
        latency_ms=10.0,
        answer=answer,
    )


def write_golden(path: Path, questions: list[dict[str, object]]) -> Path:
    path.write_text(
        yaml.safe_dump({"version": 1, "questions": questions}, allow_unicode=True),
        encoding="utf-8",
    )
    return path


class TestGoldenSetValidation:
    def test_answerable_without_lessons_is_rejected(self, tmp_path: Path) -> None:
        """Такой вопрос никогда не будет засчитан и занизит recall молча."""
        path = write_golden(
            tmp_path / "g.yaml", [{"id": "q1", "question": "вопрос", "answerable": True}]
        )

        with pytest.raises(GoldenSetError, match="список уроков пуст"):
            GoldenSet.load(path)

    def test_unanswerable_with_lessons_is_rejected(self, tmp_path: Path) -> None:
        path = write_golden(
            tmp_path / "g.yaml",
            [
                {
                    "id": "q1",
                    "question": "вопрос",
                    "answerable": False,
                    "kind": "out_of_domain",
                    "lessons": ["PEr01"],
                }
            ],
        )

        with pytest.raises(GoldenSetError, match="перечисляет уроки"):
            GoldenSet.load(path)

    def test_unanswerable_needs_kind(self, tmp_path: Path) -> None:
        path = write_golden(
            tmp_path / "g.yaml", [{"id": "q1", "question": "вопрос", "answerable": False}]
        )

        with pytest.raises(GoldenSetError, match="kind"):
            GoldenSet.load(path)

    def test_duplicate_ids_are_rejected(self, tmp_path: Path) -> None:
        path = write_golden(
            tmp_path / "g.yaml",
            [
                {"id": "q1", "question": "первый", "lessons": ["PEr01"]},
                {"id": "q1", "question": "второй", "lessons": ["PEr02"]},
            ],
        )

        with pytest.raises(GoldenSetError, match="повторяющиеся id"):
            GoldenSet.load(path)

    def test_missing_file_is_predictable_failure(self, tmp_path: Path) -> None:
        with pytest.raises(GoldenSetError, match="не найден"):
            GoldenSet.load(tmp_path / "нет.yaml")


class TestProjectGoldenSet:
    """Набор из docs/ — часть репозитория, а не черновик."""

    def test_loads(self) -> None:
        golden = GoldenSet.load(PROJECT_GOLDEN_SET)

        assert len(golden) >= 40
        assert golden.answerable
        assert golden.unanswerable

    def test_has_both_kinds_of_unanswerable(self) -> None:
        """Смежные темы важнее посторонних: на них система и склонна выдумывать."""
        golden = GoldenSet.load(PROJECT_GOLDEN_SET)
        kinds = {question.kind for question in golden.unanswerable}

        assert kinds == {"in_domain_absent", "out_of_domain"}

    def test_covers_more_than_one_module(self) -> None:
        golden = GoldenSet.load(PROJECT_GOLDEN_SET)

        assert len(golden.lessons()) >= 20


class TestMetrics:
    def test_recall_counts_only_the_first_k(self) -> None:
        outcomes = [outcome(("PEr01", "PEr02", "PEr03"))]

        assert recall_at(outcomes, 1) == 0.0
        assert recall_at(outcomes, 3) == 1.0

    def test_recall_ignores_unanswerable_questions(self) -> None:
        outcomes = [
            outcome(("PEr03",)),
            outcome((), expected=(), answerable=False, kind="out_of_domain"),
        ]

        assert recall_at(outcomes, 1) == 1.0

    def test_mrr_rewards_higher_position(self) -> None:
        first = mean_reciprocal_rank([outcome(("PEr03", "PEr01"))])
        second = mean_reciprocal_rank([outcome(("PEr01", "PEr03"))])

        assert first == 1.0
        assert second == 0.5

    def test_mrr_distinguishes_what_recall_cannot(self) -> None:
        """recall@3 одинаков, а качество выдачи — нет."""
        top = [outcome(("PEr03", "PEr01", "PEr02"))]
        bottom = [outcome(("PEr01", "PEr02", "PEr03"))]

        assert recall_at(top, 3) == recall_at(bottom, 3)
        assert mean_reciprocal_rank(top) > mean_reciprocal_rank(bottom)

    def test_refusal_stats_split_by_kind(self) -> None:
        outcomes = [
            outcome((), expected=(), answerable=False, kind="out_of_domain"),
            outcome(("PEr01",), expected=(), answerable=False, kind="in_domain_absent"),
        ]

        report = build_report(outcomes)
        by_kind = {stats.kind: stats for stats in report.refusals}

        assert by_kind["out_of_domain"].accuracy == 1.0
        assert by_kind["in_domain_absent"].accuracy == 0.0
        assert by_kind["in_domain_absent"].invented == 1

    def test_negative_separation_is_reported(self) -> None:
        """Перекрытие классов — не повод крутить порог, а вывод о корпусе."""
        outcomes = [
            outcome(("PEr03",), similarities=(0.34,)),
            outcome(
                ("PEr01",),
                expected=(),
                answerable=False,
                kind="in_domain_absent",
                similarities=(0.59,),
            ),
        ]

        report = build_report(outcomes)

        assert report.separation is not None
        assert report.separation < 0

    def test_false_refusal_counted_only_for_answerable(self) -> None:
        outcomes = [
            outcome(()),
            outcome((), expected=(), answerable=False, kind="out_of_domain"),
        ]

        assert build_report(outcomes).false_refusals == 1


class TestAnswerLevelRefusal:
    def test_model_refusal_counts_as_correct(self) -> None:
        """Второй рубеж: порог пропустил, а модель сказала «в конспектах нет»."""
        refused = outcome(
            ("PEr07",),
            expected=(),
            answerable=False,
            kind="in_domain_absent",
            answer="В конспектах этого нет. Рядом лежит кэширование [1].",
        )

        assert refused.answer_refused
        assert refused.correct
        assert not refused.hallucinated

    def test_confident_answer_to_absent_topic_is_a_hallucination(self) -> None:
        invented = outcome(
            ("PEr07",),
            expected=(),
            answerable=False,
            kind="in_domain_absent",
            answer="Redis поднимается командой docker run redis [1].",
        )

        assert not invented.answer_refused
        assert invented.hallucinated
        assert not invented.correct

    def test_retrieval_only_run_does_not_claim_refusal(self) -> None:
        """Без ответа судить о поведении модели нельзя — и метрика молчит."""
        assert not outcome(
            ("PEr07",), expected=(), answerable=False, kind="out_of_domain"
        ).answer_refused


class TestThresholdSweep:
    def test_range_is_exact(self) -> None:
        """Накопление даёт 0.6000000001 и лишнюю точку в отчёте."""
        assert thresholds_range(0.2, 0.3, 0.05) == [0.2, 0.25, 0.3]

    def test_range_rejects_zero_step(self) -> None:
        with pytest.raises(ValueError, match="положительным"):
            thresholds_range(0.2, 0.6, 0.0)

    def test_higher_threshold_cuts_candidates(self) -> None:
        candidates = [Candidate(0.6, "PEr03"), Candidate(0.4, "PEr01"), Candidate(0.2, "PEr02")]

        low = _simulate(question(), candidates, 0.3, top_k=5)
        high = _simulate(question(), candidates, 0.5, top_k=5)

        assert low.lessons == ("PEr03", "PEr01")
        assert high.lessons == ("PEr03",)

    def test_threshold_above_everything_is_a_refusal(self) -> None:
        simulated = _simulate(question(), [Candidate(0.4, "PEr03")], 0.9, top_k=5)

        assert simulated.refused


class TestObservability:
    def test_stopwatch_reports_stages_and_total(self) -> None:
        watch = Stopwatch()
        with watch.stage("embed"):
            pass
        with watch.stage("search"):
            pass

        timings = watch.finish()

        assert set(timings) == {"embed", "search", "total"}
        assert timings["total"] >= timings["embed"]

    def test_repeated_stage_accumulates(self) -> None:
        """Этап в цикле должен показать суммарную стоимость, а не последний виток."""
        watch = Stopwatch()
        for _ in range(3):
            with watch.stage("search"):
                pass

        assert "search" in watch.finish()

    def test_percentile_picks_a_real_measurement(self) -> None:
        values = [10.0, 20.0, 30.0, 40.0]

        assert percentile(values, 0.5) in values
        assert percentile(values, 0.95) == 40.0
        assert percentile([], 0.5) == 0.0

    def test_cache_counters_split_by_level(self) -> None:
        counters = UsageCounters()
        counters.record("embeddings", hit=True)
        counters.record("embeddings", hit=False)
        counters.record("retrieval", hit=True)

        usage = counters.snapshot()
        by_level = {level.level: level for level in usage.levels}

        assert by_level["embeddings"].hit_rate == 0.5
        assert by_level["retrieval"].hit_rate == 1.0
        assert by_level["answers"].lookups == 0
        assert usage.hit_rate == pytest.approx(2 / 3)
