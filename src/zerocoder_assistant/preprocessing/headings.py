"""
Нормализация текста и заголовков, классификация содержимого.

Заголовки в корпусе оформлены непоследовательно: где-то разделы нумеруются как
`## 1. Введение`, где-то как `### 1. ...`, в тексте попадаются эмодзи-маркеры и
разная пунктуация. Сопоставлять такие заголовки со списками из констант можно
только после приведения к канонической форме.

Здесь же живёт таблица посимвольных замен: она нужна и очистке тела, и санации
заголовков, поэтому собирается в одном месте.
"""

from __future__ import annotations

import re

from zerocoder_assistant.config.constants import (
    ACTUALIZATION_MARKER,
    BULLET_REPLACEMENT,
    BULLET_VARIANTS,
    CODE_DOMINANCE_RATIO,
    CONTENT_TYPE_ACTUALIZATION,
    CONTENT_TYPE_CODE,
    CONTENT_TYPE_PRACTICE,
    CONTENT_TYPE_SUMMARY,
    CONTENT_TYPE_THEORY,
    DASH_REPLACEMENT,
    DASH_VARIANTS,
    EMOJI_PATTERN,
    EXCLUDED_SECTIONS,
    HEADING_NUMBERING_PATTERN,
    INVISIBLE_TRANSLATIONS,
    PRACTICE_HEADING_MARKERS,
    QUOTE_TRANSLATIONS,
    SUMMARY_SECTIONS,
)

#: Пунктуация выбрасывается, дефис сохраняется: «no-code» — одно слово.
_PUNCTUATION_PATTERN = re.compile(r"[^\w\s-]", flags=re.UNICODE)
_WHITESPACE_PATTERN = re.compile(r"\s+")

#: «ё» и «е» в заголовках корпуса чередуются свободно («подведём»/«подведем»),
#: поэтому в канонической форме остаётся только «е».
_YO_TABLE = str.maketrans({"ё": "е", "Ё": "Е"})


def _build_translation_table() -> dict[int, str]:
    """Единая таблица посимвольных замен: тире, кавычки, буллеты, невидимые."""
    table: dict[int, str] = {}
    table.update({ord(char): DASH_REPLACEMENT for char in DASH_VARIANTS})
    table.update({ord(char): BULLET_REPLACEMENT for char in BULLET_VARIANTS})
    table.update({ord(char): value for char, value in QUOTE_TRANSLATIONS.items()})
    table.update({ord(char): value for char, value in INVISIBLE_TRANSLATIONS.items()})
    return table


#: Собирается один раз: замены применяются за один проход вместо четырёх.
TRANSLATION_TABLE = _build_translation_table()


def sanitize_heading(text: str) -> str:
    """Заголовок в виде, пригодном для показа и эмбеддинга.

    В отличие от `normalize_heading`, сохраняет регистр и пунктуацию: этот текст
    попадает в contextual header чанка и читается человеком и моделью. Снимаются
    только эмодзи-маркеры вроде «↘️» и разнобой в символах.
    """
    cleaned = text.translate(TRANSLATION_TABLE)
    cleaned = EMOJI_PATTERN.sub(" ", cleaned)
    return _WHITESPACE_PATTERN.sub(" ", cleaned).strip()


def normalize_heading(text: str) -> str:
    """Каноническая форма заголовка для сопоставления со списками констант.

    Нижний регистр, без «ё», без эмодзи, без ведущей нумерации и пунктуации.
    """
    cleaned = text.translate(TRANSLATION_TABLE).translate(_YO_TABLE).lower()
    cleaned = EMOJI_PATTERN.sub(" ", cleaned)
    cleaned = HEADING_NUMBERING_PATTERN.sub("", cleaned.strip())
    cleaned = _PUNCTUATION_PATTERN.sub(" ", cleaned)
    return _WHITESPACE_PATTERN.sub(" ", cleaned).strip()


def normalize_for_match(text: str) -> str:
    """Каноническая форма произвольного текста (для поиска шаблонных абзацев)."""
    cleaned = text.translate(TRANSLATION_TABLE).translate(_YO_TABLE).lower()
    cleaned = EMOJI_PATTERN.sub(" ", cleaned)
    cleaned = _PUNCTUATION_PATTERN.sub(" ", cleaned)
    return _WHITESPACE_PATTERN.sub(" ", cleaned).strip()


def is_excluded_section(heading: str | None) -> bool:
    """Служебная ли это секция (домашнее задание, анонс, список материалов).

    Неизвестный заголовок сохраняется: жёсткий список молча съел бы содержательные
    разделы, поскольку оформление в корпусе неоднородно.
    """
    if heading is None:
        return False
    return normalize_heading(heading) in EXCLUDED_SECTIONS


def classify_content_type(heading: str | None, text: str, code_ratio: float = 0.0) -> str:
    """Тип содержимого чанка.

    Порядок проверок задаёт приоритет меток. «Актуализация» идёт первой:
    такие блоки часто содержат листинг с пинами версий, но ценны именно как
    свежая фактура, а не как код.
    """
    normalized_text = normalize_for_match(text)
    if ACTUALIZATION_MARKER in normalized_text:
        return CONTENT_TYPE_ACTUALIZATION

    if code_ratio >= CODE_DOMINANCE_RATIO:
        return CONTENT_TYPE_CODE

    normalized_heading = normalize_heading(heading) if heading else ""
    if normalized_heading in SUMMARY_SECTIONS:
        return CONTENT_TYPE_SUMMARY
    if any(marker in normalized_heading for marker in PRACTICE_HEADING_MARKERS):
        return CONTENT_TYPE_PRACTICE

    return CONTENT_TYPE_THEORY
