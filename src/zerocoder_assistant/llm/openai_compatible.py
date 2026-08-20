"""Генерация через OpenAI-совместимый API (OpenAI и ProxyAPI)."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from openai import OpenAI, OpenAIError

from zerocoder_assistant.config.settings import ProviderCredentials
from zerocoder_assistant.errors import ProviderRequestError
from zerocoder_assistant.llm.base import ChatMessage

logger = logging.getLogger(__name__)


class OpenAICompatibleLLM:
    """Клиент чат-модели поверх OpenAI SDK."""

    def __init__(
        self,
        credentials: ProviderCredentials,
        model: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1000,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = OpenAI(
            api_key=credentials.api_key.get_secret_value(),
            base_url=credentials.base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    @property
    def model_id(self) -> str:
        return self._model

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=list(messages),
                temperature=self._temperature if temperature is None else temperature,
                max_tokens=self._max_tokens if max_tokens is None else max_tokens,
            )
        except OpenAIError as exc:
            # Повторы SDK уже исчерпаны — дальше это внешнее обстоятельство,
            # а не поломка программы, и показывать его надо строкой.
            raise ProviderRequestError(f"Модель {self._model} не ответила", exc) from exc

        content = response.choices[0].message.content
        if content is None:
            # Пустой ответ — не исключение API, но и не результат: сообщаем явно,
            # иначе дальше по конвейеру поедет None вместо текста.
            raise ProviderRequestError(f"Модель {self._model}", "вернула пустой ответ")
        return content.strip()
