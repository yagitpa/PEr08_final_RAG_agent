"""
Фильтры поиска по метаданным.

Сужают область до конкретных уроков, модулей или типов содержимого. Это не
украшение: один ассистент обслуживает вопросы «что было в PEr06» и «что вообще
известно про кэширование», и первый без фильтра отвечает хуже.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def build_where(
    lessons: Sequence[str] | None = None,
    modules: Sequence[int] | None = None,
    content_types: Sequence[str] | None = None,
) -> dict[str, Any] | None:
    """Собирает выражение `where` для Chroma.

    Одно условие передаётся напрямую, несколько — через `$and`: Chroma не
    принимает словарь с двумя ключами верхнего уровня.
    Возвращает None, если фильтровать не по чему.
    """
    conditions: list[dict[str, Any]] = []
    if lessons:
        conditions.append(_match("lesson_id", list(lessons)))
    if modules:
        conditions.append(_match("module_num", list(modules)))
    if content_types:
        conditions.append(_match("content_type", list(content_types)))

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _match(field: str, values: list[Any]) -> dict[str, Any]:
    """Условие равенства для одного значения и `$in` для нескольких."""
    if len(values) == 1:
        return {field: values[0]}
    return {field: {"$in": values}}
