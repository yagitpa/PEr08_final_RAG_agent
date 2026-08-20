"""
CLI: разбор команд диалога и отказ на бессмысленный вызов.

Проверяется то, что стоит денег при ошибке. Опечатка в команде («/exti») не
должна уезжать в модель как вопрос: это платный запрос и мусорная пара в
истории диалога.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from zerocoder_assistant.cli import evaluate
from zerocoder_assistant.cli.ask import ask
from zerocoder_assistant.cli.main import main
from zerocoder_assistant.cli.repl_commands import handle_command
from zerocoder_assistant.config.settings import Settings
from zerocoder_assistant.errors import ProviderRequestError
from zerocoder_assistant.generation.answerer import Answer
from zerocoder_assistant.memory import SessionHistory


@pytest.fixture
def history() -> SessionHistory:
    history = SessionHistory(5)
    history.add("что такое overlap", "перекрытие соседних чанков")
    return history


class TestReplCommands:
    @pytest.mark.parametrize("line", ["/exit", "/quit", "/EXIT", "/Quit "])
    def test_exit_variants(self, line: str, history: SessionHistory) -> None:
        assert handle_command(line, history).exit_requested

    def test_clear_forgets_the_dialogue(self, history: SessionHistory) -> None:
        outcome = handle_command("/clear", history)

        assert not outcome.exit_requested
        assert history.is_empty
        assert "1" in outcome.message

    def test_help_lists_commands(self, history: SessionHistory) -> None:
        assert "/clear" in handle_command("/help", history).message

    @pytest.mark.parametrize("line", ["/exti", "/сlear", "/помощь", "/"])
    def test_unknown_command_does_not_reach_the_model(
        self, line: str, history: SessionHistory
    ) -> None:
        """Опечатка — это опечатка, а не вопрос по конспектам."""
        outcome = handle_command(line, history)

        assert not outcome.exit_requested
        assert "Неизвестная команда" in outcome.message
        assert not history.is_empty  # историю опечатка не трогает

    def test_trailing_arguments_are_ignored(self, history: SessionHistory) -> None:
        assert "/clear" in handle_command("/help всё подряд", history).message


class TestAskInvocation:
    def test_question_or_repl_required(self) -> None:
        result = CliRunner().invoke(ask, [])

        assert result.exit_code != 0
        assert "--repl" in result.output

    def test_commands_are_registered(self) -> None:
        result = CliRunner().invoke(main, ["--help"])

        assert result.exit_code == 0
        for command in ("index", "search", "ask", "cache"):
            assert command in result.output


class TestEvalInvocation:
    """`eval` — команда, которая может стоить денег и портить измерения.

    Проверяется ровно то, что происходит ДО обращения к индексу и к модели:
    разбор набора и границ перебора. Ошибка на этом участке раньше вылетала
    голым исключением мимо обработчика CLI либо, что хуже, молча давала пустой
    результат.
    """

    @pytest.fixture(autouse=True)
    def isolated_settings(self, monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
        """Настройки во временной папке: команда не трогает рабочий кэш."""
        monkeypatch.setattr(evaluate, "get_settings", lambda: settings)

    def test_subcommands_are_registered(self) -> None:
        result = CliRunner().invoke(main, ["eval", "--help"])

        assert result.exit_code == 0
        for command in ("run", "threshold"):
            assert command in result.output

    def test_missing_golden_set_is_a_clean_failure(self, tmp_path: Path) -> None:
        result = CliRunner().invoke(
            main, ["eval", "run", "--golden", str(tmp_path / "нет.yaml"), "--no-cache"]
        )

        assert result.exit_code != 0
        assert "не найден" in result.output
        assert result.exception is None or isinstance(result.exception, SystemExit)

    def test_broken_golden_set_reports_the_question(self, tmp_path: Path) -> None:
        """Опечатка в наборе должна называть вопрос, а не падать трассировкой."""
        path = tmp_path / "g.yaml"
        path.write_text(
            "version: 1\nquestions:\n  - id: q1\n    question: вопрос\n    answerable: true\n",
            encoding="utf-8",
        )

        result = CliRunner().invoke(main, ["eval", "run", "--golden", str(path), "--no-cache"])

        assert result.exit_code != 0
        assert "q1" in result.output

    def test_non_dict_question_does_not_escape_as_attribute_error(self, tmp_path: Path) -> None:
        path = tmp_path / "g.yaml"
        path.write_text("version: 1\nquestions:\n  - просто строка\n", encoding="utf-8")

        result = CliRunner().invoke(main, ["eval", "run", "--golden", str(path), "--no-cache"])

        assert result.exit_code != 0
        assert not isinstance(result.exception, AttributeError)
        assert "ожидается словарь" in result.output

    def test_inverted_sweep_bounds_are_rejected(self, tmp_path: Path) -> None:
        """`--from 0.6 --to 0.2` раньше давал пустой перебор и пустой отчёт."""
        path = _valid_golden(tmp_path)

        result = CliRunner().invoke(
            main, ["eval", "threshold", "--golden", str(path), "--from", "0.6", "--to", "0.2"]
        )

        assert result.exit_code != 0
        assert "ниже нижней" in result.output

    def test_zero_step_is_rejected(self, tmp_path: Path) -> None:
        path = _valid_golden(tmp_path)

        result = CliRunner().invoke(
            main, ["eval", "threshold", "--golden", str(path), "--step", "0"]
        )

        assert result.exit_code != 0
        assert "положительным" in result.output


def _valid_golden(tmp_path: Path) -> Path:
    path = tmp_path / "golden.yaml"
    path.write_text(
        "version: 1\nquestions:\n"
        "  - id: q1\n    question: что такое overlap\n    lessons: [PEr03]\n",
        encoding="utf-8",
    )
    return path


class _FlakyAnswerer:
    """Первый вопрос обрывается сетью, второй отвечает."""

    def __init__(self, settings: object, cache: object = None) -> None:
        self.calls = 0

    def __enter__(self) -> _FlakyAnswerer:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def close(self) -> None:
        return None

    def answer(self, question: str, **_: object) -> Answer:
        self.calls += 1
        if self.calls == 1:
            raise ProviderRequestError("Модель gpt-4o-mini не ответила", "обрыв соединения")
        return Answer(
            question=question,
            text="Ответ по фрагменту [1].",
            sources=["PEr08 > Секция"],
            used_fragments=1,
        )


class TestDialogueSurvivesProviderFailure:
    """Обрыв связи не должен уносить память сессии.

    Диалог набирает контекст постепенно, и потерять его из-за одного
    неудачного запроса дороже, чем сам запрос: студент начинает разговор
    заново. Настоящая ошибка в коде при этом обязана прерывать работу.
    """

    def test_repl_continues_after_a_failed_question(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("zerocoder_assistant.generation.Answerer", _FlakyAnswerer)

        result = CliRunner().invoke(
            ask, ["--repl", "--no-cache"], input="первый вопрос\nвторой вопрос\n/exit\n"
        )

        assert result.exit_code == 0, result.output
        assert "Не получилось ответить" in result.output
        assert "обрыв соединения" in result.output
        # Главное: разговор продолжился и второй вопрос получил ответ.
        assert "Ответ по фрагменту [1]." in result.output

    def test_single_question_still_fails_the_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Разовый вопрос — это команда: сбой обязан дать ненулевой код возврата."""
        monkeypatch.setattr("zerocoder_assistant.generation.Answerer", _FlakyAnswerer)

        result = CliRunner().invoke(ask, ["вопрос", "--no-cache"])

        assert result.exit_code != 0
        assert "не ответила" in result.output


class TestContentTypeIsChecked:
    """Опечатка в типе давала пустую выдачу, неотличимую от «такого нет»."""

    def test_typo_is_rejected_with_the_list(self) -> None:
        result = CliRunner().invoke(ask, ["вопрос", "--content-type", "theroy"])

        assert result.exit_code == 2
        assert "theory" in result.output

    def test_known_type_is_accepted(self) -> None:
        """Проверка не должна отвергать настоящий тип."""
        result = CliRunner().invoke(ask, ["--content-type", "theory", "--help"])

        assert result.exit_code == 0
