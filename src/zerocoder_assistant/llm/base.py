"""
Контракт языковой модели.

Отделён от контракта эмбеддингов сознательно. Это две независимые замены:
модель генерации меняется свободно, а смена модели эмбеддингов обязывает
пересобрать индекс. Держать их одним интерфейсом значило бы скрыть эту разницу
ровно там, где о ней важнее всего помнить.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol, TypedDict, runtime_checkable

Role = Literal["system", "user", "assistant"]


class ChatMessage(TypedDict):
    """Сообщение в диалоге с моделью."""

    role: Role
    content: str


@runtime_checkable
class LLMProvider(Protocol):
    """Генерирует ответ по набору сообщений."""

    @property
    def model_id(self) -> str:
        """Имя модели — попадает в логи, метрики и ключ кеша ответов."""

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Ответ модели одним текстом."""
