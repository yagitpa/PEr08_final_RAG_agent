"""Сборка поставщика эмбеддингов по настройкам."""

from __future__ import annotations

import logging

from zerocoder_assistant.config.settings import Settings, get_settings
from zerocoder_assistant.embeddings.base import EmbeddingProvider
from zerocoder_assistant.embeddings.openai_compatible import OpenAICompatibleEmbeddings
from zerocoder_assistant.errors import UnknownProviderError

logger = logging.getLogger(__name__)

#: Провайдеры, говорящие на протоколе OpenAI. Для них хватает одного клиента,
#: различие сводится к base_url и ключу.
OPENAI_COMPATIBLE: frozenset[str] = frozenset({"openai", "proxyapi"})


def build_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    """Поставщик эмбеддингов, заданный `EMBED_PROVIDER`.

    Точка расширения: провайдер с собственным протоколом (GigaChat, YandexGPT)
    добавляется веткой здесь и отдельным модулем рядом — ядро не меняется.
    Помнить при этом надо одно: смена поставщика эмбеддингов означает пересборку
    индекса, потому что векторы разных моделей несравнимы.
    """
    settings = settings or get_settings()
    provider = settings.embed_provider.lower()
    credentials = settings.credentials(provider)

    if provider not in OPENAI_COMPATIBLE:
        # Не NotImplementedError: для того, кто настраивает .env, это ровно
        # то же самое, что опечатка в имени, — и показывать это надо так же.
        raise UnknownProviderError(provider, tuple(sorted(OPENAI_COMPATIBLE)))

    logger.debug("Эмбеддинги: %s, модель %s", provider, settings.embed_model)
    return OpenAICompatibleEmbeddings(
        credentials,
        settings.embed_model,
        batch_size=settings.embed_batch_size,
        timeout=settings.request_timeout,
        max_retries=settings.max_retries,
    )
