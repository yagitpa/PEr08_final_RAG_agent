"""
Команды диалогового режима: `/clear`, `/help`, `/exit`.

Вынесены из цикла REPL и ничего не печатают сами — возвращают текст и решение
о выходе. Так их можно проверить тестом без подмены ввода-вывода, а цикл
остаётся коротким и занимается только вводом.

Ключевое правило: **всё, что начинается со слэша, командой и остаётся**, даже
если такой команды нет. Иначе опечатка «/exti» уезжает в модель как вопрос —
платный запрос и мусорная пара в истории диалога.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:  # pragma: no cover - только для аннотаций
    from zerocoder_assistant.memory import SessionHistory

COMMAND_PREFIX: Final[str] = "/"
COMMAND_CLEAR: Final[str] = "/clear"
COMMAND_HELP: Final[str] = "/help"
COMMANDS_EXIT: Final[tuple[str, ...]] = ("/exit", "/quit")

HELP_TEXT: Final[str] = (
    f"{COMMAND_CLEAR}  забыть историю диалога\n"
    f"{COMMANDS_EXIT[0]}   выйти ({COMMANDS_EXIT[1]} — то же самое)\n"
    f"{COMMAND_HELP}   эта справка"
)

UNKNOWN_TEMPLATE: Final[str] = (
    "Неизвестная команда {command}. Список команд — " + COMMAND_HELP + "."
)
CLEARED_TEMPLATE: Final[str] = "История очищена (забыто пар реплик: {forgotten})."


@dataclass(frozen=True, slots=True)
class CommandOutcome:
    """Что показать пользователю и надо ли выходить из диалога."""

    message: str = ""
    exit_requested: bool = False


def is_command(line: str) -> bool:
    return line.startswith(COMMAND_PREFIX)


def handle_command(line: str, history: SessionHistory) -> CommandOutcome:
    """Выполнить команду диалога.

    Регистр и хвост после первого слова игнорируются: «/EXIT» и «/help всё
    подряд» — это те же команды, а не поводы обратиться к модели.
    """
    command = line.split()[0].lower() if line.split() else COMMAND_PREFIX

    if command in COMMANDS_EXIT:
        return CommandOutcome(exit_requested=True)
    if command == COMMAND_HELP:
        return CommandOutcome(message=HELP_TEXT)
    if command == COMMAND_CLEAR:
        return CommandOutcome(message=CLEARED_TEMPLATE.format(forgotten=history.clear()))
    return CommandOutcome(message=UNKNOWN_TEMPLATE.format(command=command))
