"""
Очистка текста перед векторизацией.

Важное архитектурное решение: очистка применяется здесь, а не при сохранении
конспекта. Файл `.md` остаётся полноценным читаемым документом с заголовками,
списками и акцентами; «раздевается» только та копия текста, которая уходит в
индекс. Благодаря этому правила очистки обратимы — их можно переписать и
переиндексировать, ничего не потеряв.

Второе решение: очистка работает поблочно, уже после того как `split_blocks`
отделил листинги от прозы. Поэтому не нужны плейсхолдеры для защиты кода —
блок кода просто обрабатывается по своим правилам.
"""

from __future__ import annotations

import re

from zerocoder_assistant.config.constants import (
    ADMIN_LINE_PREFIXES,
    BOILERPLATE_MARKERS,
    EMOJI_PATTERN,
    HTML_TAG_PATTERN,
    MD_BLOCKQUOTE_PATTERN,
    MD_EMPHASIS_PASSES,
    MD_EMPHASIS_PATTERN,
    MD_HRULE_PATTERN,
    MD_IMAGE_PATTERN,
    MD_LINK_PATTERN,
    MD_LIST_MARKER_PATTERN,
    MD_TABLE_DIVIDER_PATTERN,
    MODULE_TITLE_BLOCK_PATTERN,
)
from zerocoder_assistant.preprocessing.headings import TRANSLATION_TABLE, normalize_for_match
from zerocoder_assistant.preprocessing.models import Block

_INLINE_SPACE_PATTERN = re.compile(r"[ \t]{2,}")


class TextCleaner:
    """Приводит блок текста к виду, пригодному для эмбеддинга."""

    def is_boilerplate(self, text: str) -> bool:
        """Шаблонный ли это блок, повторяющийся во всех конспектах.

        Два случая: дисклеймер студенту и строка с названием модуля. Оба не несут
        знаний и в масштабе корпуса дают десятки почти одинаковых чанков.
        """
        if any(marker in normalize_for_match(text) for marker in BOILERPLATE_MARKERS):
            return True
        return bool(MODULE_TITLE_BLOCK_PATTERN.fullmatch(text.strip()))

    def drop_admin_lines(self, text: str) -> str:
        """Убирает строки административной шапки урока.

        Отбор идёт построчно: «Время чтения» встречается и внутри содержательных
        секций, где выбрасывать надо одну строку, а не всю секцию.
        """
        kept = [line for line in text.splitlines() if not self._is_admin_line(line)]
        return "\n".join(kept).strip()

    @staticmethod
    def _is_admin_line(line: str) -> bool:
        stripped = line.strip().lstrip("-").strip().lower()
        return stripped.startswith(ADMIN_LINE_PREFIXES)

    def clean_block(self, block: Block) -> Block:
        """Очищает блок по правилам его типа."""
        if block.is_code:
            return Block(self.clean_code(block.text), is_code=True)
        return Block(self.clean_text(block.text), is_code=False)

    def clean_code(self, text: str) -> str:
        """Минимальная обработка листинга.

        Внутрь кода не лезем: удаление «шума» сломало бы строковые литералы и
        отступы. Ограничиваемся невидимыми символами и хвостовыми пробелами,
        а сами fence-маркеры сохраняем — они говорят модели, что это код.
        """
        normalized = text.translate(TRANSLATION_TABLE)
        lines = [line.rstrip() for line in normalized.splitlines()]
        return "\n".join(lines).strip("\n")

    def clean_text(self, text: str) -> str:
        """Полная очистка прозы, списка или таблицы."""
        result = text.translate(TRANSLATION_TABLE)
        result = HTML_TAG_PATTERN.sub(" ", result)

        # Подпись ссылки осмысленна, URL для эмбеддинга — нет.
        result = MD_IMAGE_PATTERN.sub(r"\1", result)
        result = MD_LINK_PATTERN.sub(r"\1", result)

        # Порядок принципиален: сначала выделение, потом эмодзи. В обратном
        # порядке «**Текст ↘️**» превращается в «**Текст  **», где перед
        # закрывающим маркером пробел — паттерн выделения его уже не видит,
        # и в тексте остаются осиротевшие звёздочки.
        result = self._strip_emphasis(result)
        result = EMOJI_PATTERN.sub(" ", result)

        result = MD_BLOCKQUOTE_PATTERN.sub("", result)
        result = MD_TABLE_DIVIDER_PATTERN.sub("", result)
        result = MD_HRULE_PATTERN.sub("", result)
        result = MD_LIST_MARKER_PATTERN.sub(r"\1- ", result)

        return self._collapse_whitespace(result)

    @staticmethod
    def _strip_emphasis(text: str) -> str:
        """Снимает маркеры выделения послойно, до стабилизации."""
        for _ in range(MD_EMPHASIS_PASSES):
            stripped = MD_EMPHASIS_PATTERN.sub(r"\2", text)
            if stripped == text:
                break
            text = stripped
        return text

    @staticmethod
    def _collapse_whitespace(text: str) -> str:
        """Схлопывает пробелы и выбрасывает опустевшие строки.

        Переводы строк внутри блока сохраняются: они держат структуру списка и
        таблицы, а стоят один токен.
        """
        lines = (_INLINE_SPACE_PATTERN.sub(" ", line).strip() for line in text.splitlines())
        return "\n".join(line for line in lines if line).strip()
