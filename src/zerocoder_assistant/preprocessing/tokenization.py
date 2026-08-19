"""
Подсчёт токенов.

Размер чанка задаётся в токенах, а не в символах: у эмбеддера и LLM бюджет
токенный, и для кириллицы расхождение с посимвольной оценкой примерно
трёхкратное. Считать «по 400 символов» и называть это «400 токенов» — обычная
ошибка учебных проектов, из-за которой чанки выходят втрое мельче заявленного.
"""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

#: Запасная оценка, если tiktoken недоступен (нет пакета или не скачался словарь).
#: Для русского технического текста один токен — примерно два-три символа;
#: берём консервативную нижнюю границу, чтобы скорее раздробить, чем переполнить.
FALLBACK_CHARS_PER_TOKEN = 2.5


class TokenCounter:
    """Счётчик токенов конкретной кодировки.

    Кодировка задаётся настройкой, а не константой: у разных моделей эмбеддингов
    разные токенайзеры, и при смене модели считать надо её мерой.
    """

    def __init__(self, encoding: str) -> None:
        self.encoding = encoding
        self._encoder = _load_encoder(encoding)

    @property
    def is_exact(self) -> bool:
        """False, если работает запасная эвристика вместо настоящего токенайзера."""
        return self._encoder is not None

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._encoder is None:
            return max(1, round(len(text) / FALLBACK_CHARS_PER_TOKEN))
        return len(self._encoder.encode(text, disallowed_special=()))


@lru_cache(maxsize=4)
def _load_encoder(encoding: str):
    """Кодировщик tiktoken или None, если его не удалось получить.

    Возвращаемый тип намеренно не аннотирован: он существует только при
    установленном tiktoken, а модуль обязан импортироваться и без него.

    Кешируется: загрузка словаря BPE стоит заметного времени, а счётчик создаётся
    на каждый файл конспекта.
    """
    try:
        import tiktoken
    except ImportError:
        logger.warning(
            "tiktoken не установлен — размеры чанков считаются приблизительно. "
            "Установите зависимости из requirements.txt."
        )
        return None

    try:
        return tiktoken.get_encoding(encoding)
    except Exception as exc:  # словарь качается из сети при первом обращении
        logger.warning(
            "Не удалось загрузить кодировку %r (%s) — размеры чанков считаются приблизительно.",
            encoding,
            exc,
        )
        return None


@lru_cache(maxsize=4)
def get_token_counter(encoding: str) -> TokenCounter:
    """Счётчик токенов для кодировки (создаётся один раз на кодировку)."""
    return TokenCounter(encoding)
