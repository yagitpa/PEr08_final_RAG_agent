"""
Команда `search` — поиск по базе знаний без генерации.

Показывает ровно то, что уйдёт в модель: какие фрагменты отобраны, с каким
сходством и откуда. Отдельная команда нужна потому, что качество поиска и
качество ответа — разные вещи, и отлаживать их по одному выводу невозможно:
плохой ответ на хорошем контексте лечится промптом, а на плохом — параметрами
поиска.
"""

from __future__ import annotations

import click

from zerocoder_assistant.cli.options import retrieval_options
from zerocoder_assistant.config.settings import get_settings
from zerocoder_assistant.errors import AssistantError
from zerocoder_assistant.reporting import render_search_result


@click.command(name="search")
@click.argument("query", metavar="ЗАПРОС")
@retrieval_options
@click.option("--full", is_flag=True, help="Показать фрагменты целиком, а не первые строки.")
def search(
    query: str,
    lessons: tuple[str, ...],
    modules: tuple[int, ...],
    content_types: tuple[str, ...],
    top_k: int | None,
    no_cache: bool,
    full: bool,
) -> None:
    """Найти фрагменты базы знаний по ЗАПРОСУ."""
    from zerocoder_assistant.cache import SqliteCache
    from zerocoder_assistant.retrieval import Retriever, build_where

    settings = get_settings()
    where = build_where(lessons=lessons, modules=modules, content_types=content_types)
    cache = None if no_cache else SqliteCache(settings.cache_db)

    try:
        with Retriever(settings, cache=cache) as retriever:
            result = retriever.retrieve(query, top_k=top_k, where=where, use_cache=not no_cache)
    except AssistantError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(render_search_result(result, full=full))
