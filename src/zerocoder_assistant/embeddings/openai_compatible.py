"""
Эмбеддинги через OpenAI-совместимый API.

Одна реализация обслуживает и OpenAI, и ProxyAPI: у второго тот же протокол и
формат ответа, отличается только `base_url`. Заводить ради этого отдельный класс
значило бы дублировать код, а различие спрятать в тип вместо конфигурации.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from typing import Final

from openai import OpenAI, OpenAIError

from zerocoder_assistant.config.settings import ProviderCredentials
from zerocoder_assistant.errors import ProviderRequestError

logger = logging.getLogger(__name__)

#: Размерности известных моделей — чтобы не тратить запрос на их выяснение.
#: Неизвестная модель определяется одним пробным вызовом.
KNOWN_DIMENSIONS: Final[dict[str, int]] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

#: Текст пробного запроса, если размерность модели неизвестна.
PROBE_TEXT: Final[str] = "проверка размерности"


class OpenAICompatibleEmbeddings:
    """Поставщик эмбеддингов поверх OpenAI SDK."""

    def __init__(
        self,
        credentials: ProviderCredentials,
        model: str,
        *,
        batch_size: int = 64,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self._model = model
        self._batch_size = batch_size
        self._dimension: int | None = KNOWN_DIMENSIONS.get(model)
        # Повторы при сетевых сбоях и 429 берёт на себя сам SDK.
        self._client = OpenAI(
            api_key=credentials.api_key.get_secret_value(),
            base_url=credentials.base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        """Размерность вектора; для незнакомой модели выясняется один раз."""
        if self._dimension is None:
            self._dimension = len(self.embed_query(PROBE_TEXT))
            logger.info(
                "Размерность модели %s определена опытным путём: %d", self._model, self._dimension
            )
        return self._dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Векторы для набора текстов, в исходном порядке."""
        if not texts:
            return []

        vectors: list[list[float]] = []
        for batch in self._batches(texts):
            try:
                response = self._client.embeddings.create(model=self._model, input=list(batch))
            except OpenAIError as exc:
                raise ProviderRequestError(
                    f"Векторизация через {self._model} не удалась", exc
                ) from exc
            # Порядок в ответе не гарантирован контрактом — сортируем по index.
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend(item.embedding for item in ordered)

        if len(vectors) != len(texts):
            raise ProviderRequestError(
                f"Векторизация через {self._model}",
                f"на {len(texts)} текстов вернулось {len(vectors)} векторов — "
                "полнота выдачи нарушена, индексировать нельзя",
            )

        if self._dimension is None and vectors:
            self._dimension = len(vectors[0])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def _batches(self, texts: Sequence[str]) -> Iterator[Sequence[str]]:
        for start in range(0, len(texts), self._batch_size):
            yield texts[start : start + self._batch_size]
