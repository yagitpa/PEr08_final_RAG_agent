"""
Корневая группа CLI: `zassist`.

Единственная точка входа проекта. Сама она ничего не делает — только собирает
подгруппы (`index`, `search`, `cache`, далее `notes`, `ask`, `eval`) и настраивает логи до
того, как отработает любая команда.
"""

from __future__ import annotations

import logging
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Final

import click

from zerocoder_assistant.cli.cache import cache_group
from zerocoder_assistant.cli.index import index_group
from zerocoder_assistant.cli.search import search
from zerocoder_assistant.config.settings import get_settings

#: Имя дистрибутива из pyproject.toml — источник номера версии для `--version`.
DISTRIBUTION_NAME: Final[str] = "zerocoder-assistant"

#: Подставляется, когда пакет запущен из исходников без установки (PYTHONPATH=src):
#: метаданных дистрибутива в этом случае нет, но команда обязана работать.
FALLBACK_VERSION: Final[str] = "0.0.0+src"

CLI_PROG_NAME: Final[str] = "zassist"

LOG_FORMAT: Final[str] = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
LOG_TIME_FORMAT: Final[str] = "%H:%M:%S"
DEFAULT_LOG_LEVEL: Final[str] = "INFO"

#: Ширина справки. Опции команд подписаны по-русски и в 80 колонок не помещаются.
HELP_WIDTH: Final[int] = 100

CONTEXT_SETTINGS: Final[dict[str, object]] = {
    "help_option_names": ["-h", "--help"],
    "max_content_width": HELP_WIDTH,
}


def _package_version() -> str:
    try:
        return version(DISTRIBUTION_NAME)
    except PackageNotFoundError:
        return FALLBACK_VERSION


def _configure_logging(level_name: str) -> None:
    """Поднять логирование на заданном уровне.

    Логи уходят в stderr, а отчёты команд — в stdout: только так вывод
    `index preview` можно перенаправить в файл, не смешав его с диагностикой.
    """
    level = logging.getLevelNamesMapping().get(level_name.strip().upper())
    logging.basicConfig(
        level=level if level is not None else DEFAULT_LOG_LEVEL,
        format=LOG_FORMAT,
        datefmt=LOG_TIME_FORMAT,
        stream=sys.stderr,
    )
    if level is None:
        logging.getLogger(__name__).warning(
            "Неизвестный уровень логирования %r, используется %s", level_name, DEFAULT_LOG_LEVEL
        )


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(_package_version(), prog_name=CLI_PROG_NAME)
@click.option(
    "--log-level",
    default=None,
    metavar="LEVEL",
    help="Уровень логирования (DEBUG, INFO, WARNING, ERROR); по умолчанию LOG_LEVEL из настроек.",
)
def main(log_level: str | None) -> None:
    """RAG-ассистент по конспектам курса Зерокодера."""
    _configure_logging(log_level or get_settings().log_level)


main.add_command(index_group)
main.add_command(search)
main.add_command(cache_group)
