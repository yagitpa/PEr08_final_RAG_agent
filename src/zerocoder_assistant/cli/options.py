"""
Опции, общие для команд поиска и ответа.

`search` и `ask` ограничивают выдачу одинаково — иначе отладить ответ поиском
было бы нельзя: сравнивать имеет смысл только то, что отобрано теми же
фильтрами. Один декоратор вместо двух копий гарантирует это буквально.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import click

_Command = TypeVar("_Command", bound=Callable[..., object])


def retrieval_options(command: _Command) -> _Command:
    """Фильтры и параметры отбора фрагментов."""
    decorators = (
        click.option(
            "--lesson",
            "lessons",
            multiple=True,
            metavar="LESSON_ID",
            help="Искать только в указанных уроках (можно повторять): --lesson PEr07.",
        ),
        click.option(
            "--module",
            "modules",
            multiple=True,
            type=int,
            metavar="N",
            help="Искать только в указанных модулях: --module 5.",
        ),
        click.option(
            "--content-type",
            "content_types",
            multiple=True,
            metavar="TYPE",
            help="Ограничить тип содержимого: theory, summary, practice, code, actualization.",
        ),
        click.option(
            "--top-k",
            type=click.IntRange(min=1),
            default=None,
            help="Сколько фрагментов отобрать.",
        ),
        click.option("--no-cache", is_flag=True, help="Не использовать и не обновлять кэш."),
    )
    for decorator in reversed(decorators):
        command = decorator(command)  # type: ignore[assignment]
    return command
