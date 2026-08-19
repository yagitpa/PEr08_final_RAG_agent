"""
CLI: разбор команд диалога и отказ на бессмысленный вызов.

Проверяется то, что стоит денег при ошибке. Опечатка в команде («/exti») не
должна уезжать в модель как вопрос: это платный запрос и мусорная пара в
истории диалога.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from zerocoder_assistant.cli.ask import ask
from zerocoder_assistant.cli.main import main
from zerocoder_assistant.cli.repl_commands import handle_command
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
