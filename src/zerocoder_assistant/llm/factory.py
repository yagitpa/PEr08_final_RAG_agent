"""Сборка клиента языковой модели по настройкам."""

from __future__ import annotations

import logging

from zerocoder_assistant.config.settings import Settings, get_settings
from zerocoder_assistant.embeddings.factory import OPENAI_COMPATIBLE
from zerocoder_assistant.errors import UnknownProviderError
from zerocoder_assistant.llm.base import LLMProvider
from zerocoder_assistant.llm.openai_compatible import OpenAICompatibleLLM

logger = logging.getLogger(__name__)


def build_llm_provider(settings: Settings | None = None) -> LLMProvider:
    """Клиент модели, заданный `LLM_PROVIDER`.

    В отличие от эмбеддингов, менять здесь провайдера можно свободно: на
    собранный индекс это никак не влияет.
    """
    settings = settings or get_settings()
    provider = settings.llm_provider.lower()
    credentials = settings.credentials(provider)

    if provider not in OPENAI_COMPATIBLE:
        # Не NotImplementedError: для того, кто настраивает .env, это ровно
        # то же самое, что опечатка в имени, — и показывать это надо так же.
        raise UnknownProviderError(provider, tuple(sorted(OPENAI_COMPATIBLE)))

    logger.debug("LLM: %s, модель %s", provider, settings.llm_model)
    return OpenAICompatibleLLM(
        credentials,
        settings.llm_model,
        temperature=settings.temperature,
        timeout=settings.request_timeout,
        max_retries=settings.max_retries,
    )
