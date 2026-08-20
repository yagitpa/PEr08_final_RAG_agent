"""
Оценка через RAGAS: отбор данных, границы окружений, разбор результата.

Сам RAGAS здесь не запускается и не устанавливается — в этом окружении его нет
и быть не должно. Проверяется всё, что вокруг него: какие вопросы вообще
попадают в оценку, что происходит при ненастроенном окружении и правильно ли
разбирается то, что чужой интерпретатор напечатал.

Один тест всё-таки запускает подпроцесс по-настоящему — тем самым python, на
котором идут сами тесты. RAGAS в нём отсутствует, и это ровно тот случай,
который скрипт обязан сообщить кодом возврата, а не трассировкой.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from zerocoder_assistant.config.settings import Settings
from zerocoder_assistant.evaluation.golden_set import GoldenQuestion
from zerocoder_assistant.evaluation.metrics import QuestionOutcome
from zerocoder_assistant.evaluation.ragas_bridge import (
    MetricScore,
    RagasFailed,
    RagasNotConfigured,
    RagasReport,
    build_dataset,
    evaluate_with_ragas,
    resolve_interpreter,
    write_dataset,
)
from zerocoder_assistant.evaluation.ragas_runner import (
    _as_number,
    _averages,
    load_dataset,
)
from zerocoder_assistant.reporting import render_ragas


def outcome(
    identifier: str = "q1",
    *,
    answerable: bool = True,
    generated: bool = True,
    answer: str | None = "ответ по конспектам [1]",
    contexts: tuple[str, ...] = ("фрагмент про кэширование",),
) -> QuestionOutcome:
    return QuestionOutcome(
        question=GoldenQuestion(
            id=identifier,
            question="что такое overlap",
            lessons=("PEr03",) if answerable else (),
            answerable=answerable,
            kind=None if answerable else "in_domain_absent",
        ),
        lessons=("PEr03",),
        similarities=(0.6,),
        latency_ms=10.0,
        answer=answer,
        generated=generated,
        contexts=contexts,
    )


class TestDatasetSelection:
    """Что попадает в оценку, а что нет.

    Отбор здесь важнее самих метрик: RAGAS честно посчитает любую строку,
    которую ему дали, и молча испортит среднее.
    """

    def test_answerable_question_with_answer_is_included(self) -> None:
        dataset = build_dataset([outcome()])

        assert len(dataset) == 1
        assert dataset[0]["id"] == "q1"
        assert dataset[0]["contexts"] == ["фрагмент про кэширование"]

    def test_refusals_are_excluded(self) -> None:
        """faithfulness спрашивает, следует ли ответ из фрагментов.

        Отказ «в конспектах этого нет» из фрагментов не следует ПО ЗАМЫСЛУ.
        Посчитанный на отказах, faithfulness был бы низким там, где система
        ведёт себя правильно, — метрика наказывала бы за нужное поведение.
        """
        refusal = outcome("q2", answerable=False, answer="В конспектах этого нет. Рядом [1].")

        assert build_dataset([refusal]) == []

    def test_questions_without_a_model_call_are_excluded(self) -> None:
        """Дешёвый прогон без модели не даёт материала для RAGAS."""
        assert build_dataset([outcome(generated=False, answer=None)]) == []

    def test_empty_context_is_excluded(self) -> None:
        """Без фрагментов ни одна из метрик не определена."""
        assert build_dataset([outcome(contexts=())]) == []

    def test_mixed_set_keeps_only_the_usable(self) -> None:
        outcomes = [
            outcome("q1"),
            outcome("q2", answerable=False, answer="В конспектах этого нет."),
            outcome("q3", generated=False, answer=None),
            outcome("q4"),
        ]

        assert [row["id"] for row in build_dataset(outcomes)] == ["q1", "q4"]


class TestDatasetFile:
    def test_round_trip_keeps_cyrillic_readable(self, tmp_path: Path) -> None:
        path = write_dataset(build_dataset([outcome()]), tmp_path / "d.jsonl")
        written = path.read_text(encoding="utf-8")

        assert "конспектам" in written  # не \u043о...
        assert json.loads(written.strip())["id"] == "q1"

    def test_written_dataset_passes_the_runner_validation(self, tmp_path: Path) -> None:
        """Две стороны границы должны сойтись: что пишем, то и читается."""
        path = write_dataset(build_dataset([outcome()]), tmp_path / "d.jsonl")

        assert len(load_dataset(path)) == 1


class TestRunnerValidation:
    """Строгость на входе оценщика — по тем же причинам, что и у golden set."""

    def test_missing_file_is_a_message_not_a_traceback(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit, match="не найден"):
            load_dataset(tmp_path / "нет.jsonl")

    def test_empty_field_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "d.jsonl"
        path.write_text(
            json.dumps({"id": "q1", "question": "в", "answer": "", "contexts": ["ф"]}) + "\n",
            encoding="utf-8",
        )

        with pytest.raises(SystemExit, match="answer"):
            load_dataset(path)

    def test_contexts_must_be_a_list(self, tmp_path: Path) -> None:
        path = tmp_path / "d.jsonl"
        path.write_text(
            json.dumps({"id": "q1", "question": "в", "answer": "о", "contexts": "ф"}) + "\n",
            encoding="utf-8",
        )

        with pytest.raises(SystemExit, match="списком"):
            load_dataset(path)

    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "d.jsonl"
        row = json.dumps({"id": "q1", "question": "в", "answer": "о", "contexts": ["ф"]})
        path.write_text(f"\n{row}\n\n", encoding="utf-8")

        assert len(load_dataset(path)) == 1

    def test_empty_dataset_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "d.jsonl"
        path.write_text("\n\n", encoding="utf-8")

        with pytest.raises(SystemExit, match="пуст"):
            load_dataset(path)


class TestScoreArithmetic:
    """NaN от судьи не должен превращаться в ноль."""

    def test_nan_becomes_none(self) -> None:
        assert _as_number(float("nan")) is None
        assert _as_number(None) is None
        assert _as_number("не число") is None
        assert _as_number(0.75) == 0.75

    def test_average_skips_uncounted_questions(self) -> None:
        """Иначе один неразобранный ответ утягивал бы среднее к нулю."""
        per_question = [
            {"id": "q1", "scores": {"faithfulness": 1.0}},
            {"id": "q2", "scores": {"faithfulness": None}},
            {"id": "q3", "scores": {"faithfulness": 0.5}},
        ]

        averages = _averages(per_question, ["faithfulness"])

        assert averages["faithfulness"]["mean"] == 0.75
        assert averages["faithfulness"]["counted"] == 2

    def test_metric_counted_nowhere_is_none_not_zero(self) -> None:
        """Ноль означал бы «система полностью провалилась», а не «не посчитано»."""
        averages = _averages([{"id": "q1", "scores": {"faithfulness": None}}], ["faithfulness"])

        assert averages["faithfulness"]["mean"] is None
        assert averages["faithfulness"]["counted"] == 0


class TestEnvironmentBoundary:
    """Граница между окружениями: настроена или честно объявлена ненастроенной."""

    def test_unset_interpreter_explains_why_there_are_two_environments(
        self, tmp_path: Path
    ) -> None:
        settings = Settings(openai_api_key="k", ragas_python=None)

        with pytest.raises(RagasNotConfigured, match="RAGAS_PYTHON"):
            resolve_interpreter(settings)

    def test_typo_in_path_is_called_a_typo(self, tmp_path: Path) -> None:
        """А не FileNotFoundError из недр subprocess."""
        settings = Settings(openai_api_key="k", ragas_python=tmp_path / "нет.exe")

        with pytest.raises(RagasNotConfigured, match="несуществующий файл"):
            resolve_interpreter(settings)

    def test_empty_dataset_never_reaches_the_subprocess(self, tmp_path: Path) -> None:
        settings = Settings(openai_api_key="k", ragas_python=Path(sys.executable))

        with pytest.raises(RagasFailed, match="Нечего оценивать"):
            evaluate_with_ragas(settings, [])

    def test_interpreter_without_ragas_reports_it_by_exit_code(self, tmp_path: Path) -> None:
        """Настоящий запуск подпроцесса — тем python, на котором идут тесты.

        RAGAS в нём нет и не должно быть: ядро и оценка живут в разных
        окружениях именно потому, что RAGAS откатывает openai до 2.x. Скрипт
        обязан сообщить об этом кодом возврата, который мост превращает в
        «не настроено», а не в падение.
        """
        settings = Settings(
            openai_api_key="ключ-для-теста",
            ragas_python=Path(sys.executable),
            cache_db=tmp_path / "cache.db",
        )
        dataset = build_dataset([outcome()])

        with pytest.raises(RagasNotConfigured, match="нет RAGAS"):
            evaluate_with_ragas(
                settings,
                dataset,
                dataset_path=tmp_path / "d.jsonl",
                scores_path=tmp_path / "s.json",
            )


class TestReportParsing:
    def test_payload_becomes_typed_report(self) -> None:
        report = RagasReport.from_payload(
            {
                "questions": 12,
                "model": "gpt-4o-mini",
                "metrics": {"faithfulness": {"mean": 0.87, "counted": 12}},
                "per_question": [{"id": "q1", "scores": {"faithfulness": 1.0}}],
            }
        )

        assert report.questions == 12
        assert report.metrics == (MetricScore("faithfulness", 0.87, 12),)

    def test_missing_fields_do_not_crash(self) -> None:
        """Чужой процесс мог напечатать меньше, чем мы ждём."""
        report = RagasReport.from_payload({})

        assert report.questions == 0
        assert report.metrics == ()


class TestRagasRendering:
    def test_partial_coverage_is_visible(self) -> None:
        """Среднее по трём вопросам не должно выглядеть как среднее по сорока."""
        report = RagasReport(
            questions=40,
            model="gpt-4o-mini",
            metrics=(MetricScore("faithfulness", 0.9, 3),),
        )

        assert "посчитано на 3" in render_ragas(report)

    def test_full_coverage_is_not_annotated(self) -> None:
        report = RagasReport(
            questions=40,
            model="gpt-4o-mini",
            metrics=(MetricScore("faithfulness", 0.9, 40),),
        )

        assert "посчитано на" not in render_ragas(report)

    def test_uncounted_metric_is_not_shown_as_zero(self) -> None:
        report = RagasReport(
            questions=40, model="gpt-4o-mini", metrics=(MetricScore("faithfulness", None, 0),)
        )
        rendered = render_ragas(report)

        assert "не посчитана" in rendered
        assert "0.000" not in rendered

    def test_low_score_is_flagged(self) -> None:
        low = render_ragas(
            RagasReport(questions=1, model="m", metrics=(MetricScore("faithfulness", 0.4, 1),))
        )
        high = render_ragas(
            RagasReport(questions=1, model="m", metrics=(MetricScore("faithfulness", 0.9, 1),))
        )

        assert "ниже 0.7" in low
        assert "ниже 0.7" not in high
