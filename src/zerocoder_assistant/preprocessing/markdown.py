"""
Разбор структуры Markdown: frontmatter, секции, блоки, предложения.

Весь разбор ведётся с учётом fenced-блоков. Это не мелочь: в корпусе 552 маркера
``` и внутри листингов свободно встречаются символы `#` и пустые строки. Наивный
построчный разбор принял бы комментарий Python за заголовок и разорвал бы листинг
пополам.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import yaml

from zerocoder_assistant.config.constants import (
    FENCE_PATTERN,
    HEADING_PATTERN,
    SENTENCE_ABBREVIATIONS,
    SENTENCE_BOUNDARY_PATTERN,
)
from zerocoder_assistant.preprocessing.models import Block, Section

logger = logging.getLogger(__name__)

_FRONTMATTER_PATTERN = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n?", flags=re.DOTALL)

#: Последний «токен» перед точкой — кандидат в сокращение.
_TRAILING_TOKEN_PATTERN = re.compile(r"(\S+)\.\s*$")
_LEADING_NONWORD_PATTERN = re.compile(r"^\W+", flags=re.UNICODE)


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Отделяет YAML-frontmatter от тела документа.

    Существующие конспекты frontmatter не имеют — метаданные для них выводятся из
    пути и заголовка. Но конвейер авторства (P1) его проставляет, и тогда он имеет
    приоритет над выведенными значениями.
    """
    match = _FRONTMATTER_PATTERN.match(text)
    if not match:
        return {}, text

    try:
        data = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        logger.warning("Не удалось разобрать frontmatter (%s), он будет проигнорирован", exc)
        return {}, text

    if not isinstance(data, dict):
        return {}, text
    return data, text[match.end() :]


def _read_fence(line: str) -> tuple[str, str] | None:
    """Маркер и информационная строка fence-линии, либо None."""
    match = FENCE_PATTERN.match(line)
    if match is None:
        return None
    return match.group(2), match.group(3).strip()


def _closes_fence(marker: str, info: str, opened_with: str) -> bool:
    """Закрывает ли эта fence-линия ранее открытый блок.

    По спецификации CommonMark закрывающий маркер — того же типа, не короче
    открывающего и без информационной строки.
    """
    return marker[0] == opened_with[0] and len(marker) >= len(opened_with) and not info


def split_sections(markdown: str) -> list[Section]:
    """Делит документ на секции по заголовкам, сохраняя путь вложенности.

    Текст до первого заголовка возвращается отдельной секцией с `heading=None`.
    """
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    heading: str | None = None
    level = 0
    path: tuple[str, ...] = ()
    buffer: list[str] = []
    fence: str | None = None

    def emit() -> None:
        body = "\n".join(buffer).strip()
        if heading is None and not body:
            return
        sections.append(Section(heading, level, path, body, len(sections)))

    for line in markdown.splitlines():
        fence_info = _read_fence(line)
        if fence_info is not None:
            marker, info = fence_info
            if fence is None:
                fence = marker
            elif _closes_fence(marker, info, fence):
                fence = None
            buffer.append(line)
            continue

        heading_match = HEADING_PATTERN.match(line) if fence is None else None
        if heading_match is not None:
            emit()
            buffer = []
            level = len(heading_match.group(1))
            heading = heading_match.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
            path = tuple(title for _, title in stack)
            continue

        buffer.append(line)

    emit()
    return sections


def split_blocks(body: str) -> list[Block]:
    """Делит тело секции на неделимые блоки.

    Блоком становится абзац, список или таблица (они не содержат пустых строк
    внутри, поэтому разделяются естественно) и целиком — каждый fenced-листинг.
    """
    blocks: list[Block] = []
    text_lines: list[str] = []
    code_lines: list[str] = []
    fence: str | None = None

    def flush_text() -> None:
        text = "\n".join(text_lines).strip()
        if text:
            blocks.append(Block(text, is_code=False))
        text_lines.clear()

    for line in body.splitlines():
        fence_info = _read_fence(line)
        if fence_info is not None:
            marker, info = fence_info
            if fence is None:
                flush_text()
                fence = marker
                code_lines.append(line)
                continue
            code_lines.append(line)
            if _closes_fence(marker, info, fence):
                blocks.append(Block("\n".join(code_lines), is_code=True))
                code_lines.clear()
                fence = None
            continue

        if fence is not None:
            code_lines.append(line)
            continue

        if not line.strip():
            flush_text()
            continue

        text_lines.append(line)

    if code_lines:  # незакрытый fence — сохраняем как код, а не теряем
        blocks.append(Block("\n".join(code_lines), is_code=True))
    flush_text()
    return blocks


def _is_false_boundary(prefix: str) -> bool:
    """Точка в конце `prefix` не заканчивает предложение?

    Отсекает два случая: известные сокращения («и т.д. Затем») и номер пункта
    списка («1. Текст»), где точка структурная, а не предложенческая.
    """
    if not prefix.endswith("."):
        return False

    match = _TRAILING_TOKEN_PATTERN.search(prefix)
    if match is None:
        return False

    token = _LEADING_NONWORD_PATTERN.sub("", match.group(1)).lower()
    if not token:
        return False
    return token.isdigit() or token in SENTENCE_ABBREVIATIONS


def split_sentences(text: str) -> list[str]:
    """Делит текст на предложения с поправкой на русские сокращения.

    Нужно для двух вещей: дробления слишком длинных абзацев и набора перекрытия
    целыми предложениями, а не обрубками.
    """
    if not text.strip():
        return []

    sentences: list[str] = []
    start = 0
    for match in SENTENCE_BOUNDARY_PATTERN.finditer(text):
        if _is_false_boundary(text[start : match.start()]):
            continue
        fragment = text[start : match.start()].strip()
        if fragment:
            sentences.append(fragment)
        start = match.end()

    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences
