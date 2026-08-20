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
from zerocoder_assistant.evaluation.golden_set import (
    KIND_IN_DOMAIN_ABSENT,
    KIND_OUT_OF_DOMAIN,
)
from zerocoder_assistant.evaluation.refusal import looks_like_refusal
from zerocoder_assistant.evaluation.runner import Candidate, ThresholdPoint, _simulate
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
    best: float | None = None,
) -> QuestionOutcome:
    similarities = similarities or tuple(0.5 for _ in found)
    return QuestionOutcome(
        question=question(lessons=expected, answerable=answerable, kind=kind),
        lessons=found,
        similarities=similarities,
        latency_ms=10.0,
        answer=answer,
        # Модель вызывалась ровно тогда, когда есть её ответ: подставленный
        # текст отказа при пустой выдаче — не работа модели.
        generated=answer is not None,
        best_similarity=best if best is not None else (similarities[0] if similarities else None),
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

    def test_separation_uses_candidates_not_selected_fragments(self) -> None:
        """Иначе зазор улучшался бы ровно тогда, когда система ломается.

        Отобранный фрагмент физически не бывает ниже порога, поэтому «минимум
        по отвечаемым», посчитанный по нему, растёт вместе с порогом. Вопрос с
        пустой выдачей — худший исход из возможных — выпадал из минимума совсем
        и зазор от него только выигрывал.
        """
        outcomes = [
            # Порог всё срезал: отвечаемый вопрос остался без выдачи, но
            # лучший кандидат был совсем плох.
            outcome((), similarities=(), best=0.21),
            outcome(
                ("PEr01",),
                expected=(),
                answerable=False,
                kind="in_domain_absent",
                similarities=(0.59,),
            ),
        ]

        report = build_report(outcomes)

        assert report.min_similarity_answerable == 0.21
        assert report.separation == pytest.approx(0.21 - 0.59)

    def test_recall_levels_follow_actual_output(self) -> None:
        """С `--top-k 8` печатать recall@5 и считать исход по восьми нельзя."""
        report = build_report([outcome(tuple(f"PEr{n:02d}" for n in range(1, 9)))])

        assert max(report.recall) == 8

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

    def test_threshold_refusal_is_not_credited_to_the_model(self) -> None:
        """Тот же отказ нельзя засчитать дважды.

        При пустой выдаче модель не вызывается, а текст отказа подставляется
        готовой строкой. Если засчитать её второму рубежу, два рубежа перестают
        быть независимыми, а метрика «модель сама отказала» — измеримой.
        """
        blocked = QuestionOutcome(
            question=question(lessons=(), answerable=False, kind="in_domain_absent"),
            lessons=(),
            similarities=(),
            latency_ms=1.0,
            answer="В конспектах ничего подходящего не нашлось.",
            generated=False,
        )

        assert not blocked.answer_refused
        assert blocked.correct  # засчитан порогом, а не моделью

    def test_refusal_needs_two_signals_in_one_sentence(self) -> None:
        """Уверенный ответ, упомянувший конспекты, — не отказ.

        Промпт сам велит модели писать «в конспектах есть [1]», поэтому одного
        слова про источник мало: нужна ещё и отрицательная связка, и обе — в
        одном предложении.
        """
        confident = outcome(
            ("PEr07",),
            expected=(),
            answerable=False,
            kind="in_domain_absent",
            answer="В конспектах есть [1]: Redis поднимается командой docker run redis.",
        )

        assert not confident.answer_refused
        assert confident.hallucinated


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

    def test_score_ignores_questions_the_model_answers_for(self) -> None:
        """Смежная тема, которую порог пропустил, — не ошибка порога.

        Отказ на ней даёт ветка C системного промпта. Пока эти вопросы входили
        в оценку, перебор требовал от порога чужой работы: единственный способ
        отказать на смежной теме — подняться выше сходства отвечаемых вопросов.
        """
        report = build_report(
            [
                outcome(("PEr03",)),
                outcome((), answerable=False, kind=KIND_OUT_OF_DOMAIN),
                outcome(("PEr01",), answerable=False, kind=KIND_IN_DOMAIN_ABSENT),
            ]
        )
        point = ThresholdPoint(threshold=0.3, report=report)

        # Прежняя формула считала бы 2 из 3 — смежная тема шла в минус порогу.
        assert point.score == 1.0
        assert [item.question.kind for item in point.graded] == [None, KIND_OUT_OF_DOMAIN]

    def test_score_still_punishes_a_threshold_that_cuts_the_answer(self) -> None:
        report = build_report(
            [
                outcome(()),
                outcome((), answerable=False, kind=KIND_OUT_OF_DOMAIN),
                outcome(("PEr01",), answerable=False, kind=KIND_IN_DOMAIN_ABSENT),
            ]
        )

        assert ThresholdPoint(threshold=0.9, report=report).score == 0.5

    def test_refusal_columns_are_split_by_kind(self) -> None:
        report = build_report(
            [
                outcome((), answerable=False, kind=KIND_OUT_OF_DOMAIN),
                outcome(("PEr01",), answerable=False, kind=KIND_IN_DOMAIN_ABSENT),
            ]
        )
        point = ThresholdPoint(threshold=0.3, report=report)

        assert point.foreign_refused == (1, 1)
        assert point.adjacent_refused == (0, 1)


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
        """Значение проверяется точно, а не через `in values`.

        Проверка на вхождение проходила и у реализации, которая всегда
        возвращает максимум: любой процентиль — тоже одно из измерений.
        Медиана и p95 обязаны различаться.
        """
        values = [10.0, 20.0, 30.0, 40.0]

        assert percentile(values, 0.5) == 30.0
        assert percentile(values, 0.95) == 40.0
        assert percentile(values, 0.0) == 10.0
        assert percentile([], 0.5) == 0.0

    def test_percentile_ignores_input_order(self) -> None:
        assert percentile([40.0, 10.0, 30.0, 20.0], 0.5) == percentile(
            [10.0, 20.0, 30.0, 40.0], 0.5
        )

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


class TestRecallDefinition:
    """Границы recall@k. Здесь легко ошибиться на единицу и не заметить.

    Все проверки ниже написаны после прогона мутаций: `[:k]` → `[: k + 1]`,
    деление на число всех вопросов вместо отвечаемых и подобные подмены
    проходили прежний набор тестов целиком.
    """

    def test_k_is_inclusive_at_the_boundary(self) -> None:
        """Нужный урок ровно на k-й позиции засчитывается, на k+1-й — нет."""
        outcomes = [outcome(("PEr01", "PEr02", "PEr03"))]

        assert recall_at(outcomes, 2) == 0.0
        assert recall_at(outcomes, 3) == 1.0

    def test_denominator_is_answerable_only(self) -> None:
        """Неотвечаемые вопросы не разбавляют recall.

        Иначе набор можно было бы «улучшить», дописав в него посторонних
        вопросов, — метрика выросла бы, а система осталась прежней.
        """
        outcomes = [
            outcome(("PEr03",)),
            outcome((), expected=(), answerable=False, kind="out_of_domain"),
            outcome((), expected=(), answerable=False, kind="out_of_domain"),
        ]

        assert recall_at(outcomes, 1) == 1.0

    def test_mrr_denominator_is_answerable_only(self) -> None:
        outcomes = [
            outcome(("PEr03",)),
            outcome((), expected=(), answerable=False, kind="out_of_domain"),
        ]

        assert mean_reciprocal_rank(outcomes) == 1.0

    def test_mrr_ignores_questions_without_a_hit(self) -> None:
        """Промах вносит ноль, но остаётся в знаменателе."""
        outcomes = [outcome(("PEr03",)), outcome(("PEr01",))]

        assert mean_reciprocal_rank(outcomes) == 0.5


class TestCorrectness:
    """Что именно считается правильным исходом."""

    def test_answerable_is_correct_when_lesson_is_anywhere_in_output(self) -> None:
        assert outcome(("PEr01", "PEr02", "PEr03")).correct

    def test_answerable_is_wrong_when_lesson_is_missing(self) -> None:
        assert not outcome(("PEr01", "PEr02")).correct

    def test_answerable_with_empty_output_is_wrong(self) -> None:
        """Пустая выдача на отвечаемый вопрос — ложный отказ, а не «нет данных»."""
        empty = outcome(())

        assert empty.refused
        assert not empty.correct

    def test_unanswerable_is_correct_when_threshold_blocked_everything(self) -> None:
        assert outcome((), expected=(), answerable=False, kind="out_of_domain").correct

    def test_failures_list_only_wrong_outcomes(self) -> None:
        good = outcome(("PEr03",))
        bad = outcome(("PEr01",))

        assert build_report([good, bad]).failures == (bad,)


class TestSimulatedThreshold:
    """`_simulate` обязан повторять рабочий конвейер, а не напоминать его.

    Расхождение здесь тише всех прочих: подбор порога печатает правдоподобные
    числа, по ним выбирается значение, и только рабочий прогон показывает, что
    оно другое.
    """

    def test_candidate_exactly_at_threshold_is_kept(self) -> None:
        """Порог — включающая граница, как и в `Retriever`."""
        simulated = _simulate(question(), [Candidate(0.30, "PEr03")], 0.30, top_k=5)

        assert simulated.lessons == ("PEr03",)

    def test_candidate_just_below_threshold_is_dropped(self) -> None:
        simulated = _simulate(question(), [Candidate(0.2999, "PEr03")], 0.30, top_k=5)

        assert simulated.lessons == ()

    def test_output_is_cut_to_top_k(self) -> None:
        candidates = [Candidate(0.9 - index / 100, f"PEr{index:02d}") for index in range(6)]

        simulated = _simulate(question(), candidates, 0.1, top_k=3)

        assert len(simulated.lessons) == 3

    def test_threshold_applies_before_the_slice(self) -> None:
        """Порядок операций, а не только результат на удобных данных.

        На отсортированной выдаче «срез, потом порог» даёт то же самое, поэтому
        подмена порядка не ловится обычными примерами. Здесь список намеренно
        не отсортирован: правильный порядок оставит оба сильных кандидата,
        перевёрнутый — потеряет тот, что стоит дальше среза.
        """
        candidates = [
            Candidate(0.9, "PEr03"),
            Candidate(0.1, "PEr01"),
            Candidate(0.1, "PEr02"),
            Candidate(0.8, "PEr04"),
        ]

        simulated = _simulate(question(), candidates, 0.5, top_k=3)

        assert simulated.lessons == ("PEr03", "PEr04")

    def test_best_candidate_survives_the_threshold(self) -> None:
        """Разделимость считается по кандидатам, поэтому порог её не двигает."""
        candidates = [Candidate(0.4, "PEr03"), Candidate(0.2, "PEr01")]

        low = _simulate(question(), candidates, 0.1, top_k=5)
        high = _simulate(question(), candidates, 0.9, top_k=5)

        assert low.best_similarity == 0.4
        assert high.best_similarity == 0.4
        assert high.refused

    def test_range_rejects_inverted_bounds(self) -> None:
        """`--from 0.6 --to 0.2` раньше молча давал пустой перебор."""
        with pytest.raises(ValueError, match="ниже нижней"):
            thresholds_range(0.6, 0.2, 0.02)

    def test_range_never_exceeds_the_upper_bound(self) -> None:
        """Округление к ближайшему давало точку ВЫШЕ `stop`."""
        assert max(thresholds_range(0.2, 0.25, 0.02)) <= 0.25


class TestRefusalDetection:
    """Опознание отказа. Метрика уже дважды врала — здесь закреплены оба случая."""

    def test_substring_inside_a_word_is_not_a_negation(self) -> None:
        """«кабиНЕТ» и «инТЕРНЕТ» содержат «нет», но отрицанием не являются.

        Поиск подстрокой без границ слова превращал уверенную выдумку в
        засчитанный отказ: хватало «конспект» и любого слова с «нет» внутри.
        """
        assert not looks_like_refusal("В конспектах описан личный кабинет n8n [1].")
        assert not looks_like_refusal("Материал урока объясняет интернет-магазин [2].")

    def test_negation_is_a_whole_word_on_both_sides(self) -> None:
        """«НЕТерпение» начинается с «нет», но отрицанием не является.

        Граница нужна с обеих сторон: начальная отсекает «кабинет», конечная —
        слова, которые с «нет» начинаются.
        """
        assert not looks_like_refusal("В конспектах описано нетерпение пользователя [1].")

    def test_scope_word_inside_another_word_does_not_count(self) -> None:
        """«дезИНФОРМАЦИя» — не упоминание границ базы знаний.

        Границы нужны обоим спискам, а не только отрицаниям: слово о базе,
        найденное внутри чужого слова, даёт ту же пару признаков и то же
        ложное «модель отказалась».
        """
        assert not looks_like_refusal(
            "В уроке разбирается борьба с дезинформацией, а готовых рецептов нет."
        )

    def test_gap_disclaimer_after_a_cited_claim_is_not_a_refusal(self) -> None:
        """Ветка B отвечает по существу и обязана оговорить пробел.

        Считать такой ответ отказом значит снимать флаг галлюцинации с
        уверенно выдуманного текста — то есть ровно то, ради чего метрика и
        писалась.
        """
        branch_b = (
            "Размер чанка задаётся параметром `CHUNK_TARGET_TOKENS` [1]. "
            "Чем обосновано именно значение 400 токенов — во фрагментах не сказано."
        )

        assert not looks_like_refusal(branch_b)

    def test_disclaimer_appended_to_a_cited_claim_is_not_a_refusal(self) -> None:
        """Утверждение и оговорка в ОДНОМ предложении — всё ещё ветка B.

        Найдено на живом прогоне: «...могут использоваться для хранения
        векторов [1], хотя в конспектах это не указано». Проверка по
        предложению целиком считала такое отказом, потому что оба признака в
        нём есть. Решает порядок: отказ засчитывается, только если сложился
        до первой ссылки.
        """
        compound = (
            "Стоит взять локальное хранилище, такое как ChromaDB. "
            "Оба варианта работают на своей машине [1], хотя в конспектах это прямо не указано."
        )

        assert not looks_like_refusal(compound)

    def test_signals_split_by_a_citation_are_not_a_refusal(self) -> None:
        """«В конспектах есть [1], но X не описан» — утверждение с оговоркой.

        Признаки стоят по разные стороны от ссылки: упоминание базы — до,
        отрицание — после. Отказ складывается по ПОСЛЕДНЕМУ из двух, поэтому
        он оказывается после ссылки и не засчитывается. Если считать по
        первому, такой ответ снова станет отказом.
        """
        assert not looks_like_refusal(
            "В конспектах есть про размер чанка [1], но обоснование не описано."
        )

    def test_refusal_before_a_citation_in_the_same_sentence_counts(self) -> None:
        """А если отказ стоит ПЕРЕД ссылкой, это по-прежнему отказ."""
        assert looks_like_refusal("В конспектах этого нет, рядом лежит кэширование [1].")

    def test_refusal_before_any_citation_counts(self) -> None:
        """Ветка C отказывается первой, а ссылки приводит уже потом."""
        assert looks_like_refusal("В конспектах этого нет. Рядом лежит кэширование [1], [2].")

    def test_refusal_may_come_in_the_second_sentence(self) -> None:
        """Пока первое предложение ничего не утверждает, отказ ещё впереди."""
        assert looks_like_refusal("Эта тема в курсе не разбиралась. В материалах её нет.")

    def test_wordings_absent_from_the_prompt_are_recognised(self) -> None:
        """Модель отказывается и своими словами, не только словами промпта."""
        assert looks_like_refusal("Такой информации у меня нет.")
        assert looks_like_refusal("В предоставленном контексте ответа не содержится.")


class TestRefusalNextToCode:
    """Листинг в ответе не должен работать границей «пошли утверждения».

    Индексация `chunks[0]` раньше опознавалась как ссылка на фрагмент, и
    разбор обрывался на первом же куске кода — отказ, стоящий после листинга,
    не засчитывался.
    """

    def test_indexing_in_a_fence_does_not_end_the_scan(self) -> None:
        answer = (
            "Пример обращения к выдаче:\n\n"
            "```python\n"
            "first = chunks[0]\n"
            "```\n\n"
            "В конспектах о самом параметре ничего не сказано."
        )
        assert looks_like_refusal(answer)

    def test_real_citation_still_ends_the_scan(self) -> None:
        answer = (
            "Размер чанка задаётся параметром CHUNK_TARGET_TOKENS [1]. "
            "Чем обосновано именно 400, в конспектах не сказано."
        )
        assert not looks_like_refusal(answer)


class TestFalseRefusal:
    def test_model_refusal_on_answerable_question_is_a_failure(self) -> None:
        """Поиск нашёл нужное, а модель всё равно отказалась — это провал.

        Раньше такой исход считался правильным: `correct` смотрел только на
        попадание урока в выдачу. Ассистент, отказывающий на каждом втором
        отвечаемом вопросе при исправном поиске, показывал бы recall 100% и
        «ложных отказов 0».
        """
        refused = outcome(("PEr03",), answer="В конспектах этого нет.")

        assert refused.hit_at(1)
        assert refused.answer_refused
        assert not refused.correct
        assert refused.false_refusal

    def test_false_refusals_count_both_lines_of_defence(self) -> None:
        outcomes = [
            outcome(()),  # порог не пропустил ничего
            outcome(("PEr03",), answer="В конспектах этого нет."),  # отказала модель
            outcome(("PEr03",), answer="Перекрытие составляет 15% [1]."),  # ответила
        ]

        assert build_report(outcomes).false_refusals == 2

    def test_answered_question_stays_correct(self) -> None:
        answered = outcome(("PEr03",), answer="Перекрытие составляет 15% [1].")

        assert answered.correct
        assert not answered.false_refusal
