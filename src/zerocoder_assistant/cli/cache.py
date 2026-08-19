"""Команды группы `cache` — состояние и очистка кэша."""

from __future__ import annotations

import click

from zerocoder_assistant.cache.sqlite_cache import LEVEL_TABLES
from zerocoder_assistant.config.settings import get_settings
from zerocoder_assistant.reporting import render_cache_stats


@click.group(name="cache")
def cache_group() -> None:
    """Кэш запросов, поиска и ответов."""


@cache_group.command(name="stats")
def stats() -> None:
    """Сколько записей лежит на каждом уровне кэша."""
    from zerocoder_assistant.cache import SqliteCache

    settings = get_settings()
    click.echo(render_cache_stats(SqliteCache(settings.cache_db).stats(), settings.cache_db))


@cache_group.command(name="clear")
@click.option(
    "--level",
    type=click.Choice(sorted(LEVEL_TABLES)),
    default=None,
    help="Очистить только один уровень; по умолчанию — все.",
)
@click.option("--yes", is_flag=True, help="Не спрашивать подтверждения.")
def clear(level: str | None, yes: bool) -> None:
    """Очистить кэш.

    Нужна после пересборки индекса или правки промптов: кэш переживает и то,
    и другое только частично — отпечаток конфигурации в ключе спасает от
    большинства случаев, но не от смены самих правил очистки текста.
    """
    from zerocoder_assistant.cache import SqliteCache

    settings = get_settings()
    target = level or "все уровни"
    if not yes:
        click.confirm(f"Очистить кэш ({target})?", abort=True)

    removed = SqliteCache(settings.cache_db).clear(level)
    click.echo(f"Удалено записей: {removed}")
