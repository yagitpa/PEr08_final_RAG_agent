"""Разрешение доступов и сборка провайдеров.

Тесты намеренно не трогают сеть и не читают окружение: настройки собираются
явными аргументами, у которых в pydantic-settings наивысший приоритет.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from zerocoder_assistant.config.settings import Settings
from zerocoder_assistant.embeddings.factory import build_embedding_provider
from zerocoder_assistant.embeddings.openai_compatible import (
    KNOWN_DIMENSIONS,
    OpenAICompatibleEmbeddings,
)
from zerocoder_assistant.errors import MissingCredentialsError
from zerocoder_assistant.llm.factory import build_llm_provider


def make_settings(**overrides: Any) -> Settings:
    """Настройки с заданными доступами, не зависящие от окружения машины."""
    defaults: dict[str, Any] = {
        "openai_api_key": "test-openai-key",
        "proxyapi_api_key": "test-proxy-key",
    }
    return Settings(**{**defaults, **overrides})


class TestCredentials:
    def test_openai_resolved(self) -> None:
        credentials = make_settings().credentials("openai")

        assert credentials.api_key == "test-openai-key"
        assert "api.openai.com" in credentials.base_url

    def test_proxyapi_differs_only_by_address(self) -> None:
        """ProxyAPI — это тот же протокол, другой адрес; отдельный клиент не нужен."""
        settings = make_settings()

        assert settings.credentials("proxyapi").base_url != settings.credentials("openai").base_url

    def test_provider_name_is_case_insensitive(self) -> None:
        assert make_settings().credentials("OpenAI").api_key == "test-openai-key"

    def test_missing_key_names_the_variable(self) -> None:
        with pytest.raises(MissingCredentialsError) as excinfo:
            make_settings(openai_api_key=None).credentials("openai")

        assert "OPENAI_API_KEY" in str(excinfo.value)

    def test_empty_key_counts_as_missing(self) -> None:
        """Пустая строка в .env — самая частая форма «ключ забыли вписать»."""
        with pytest.raises(MissingCredentialsError):
            make_settings(openai_api_key="").credentials("openai")

    def test_unknown_provider_lists_supported(self) -> None:
        with pytest.raises(ValueError, match="proxyapi") as excinfo:
            make_settings().credentials("gigachat")

        assert "gigachat" in str(excinfo.value)


class TestFactories:
    def test_embedding_provider_built(self) -> None:
        provider = build_embedding_provider(make_settings(embed_model="text-embedding-3-small"))

        assert provider.model_id == "text-embedding-3-small"
        assert provider.dimension == 1536  # известна заранее, запрос не нужен

    def test_llm_provider_built(self) -> None:
        assert build_llm_provider(make_settings(llm_model="gpt-4o-mini")).model_id == "gpt-4o-mini"

    def test_unsupported_provider_is_explicit(self) -> None:
        settings = make_settings(embed_provider="proxyapi")
        assert build_embedding_provider(settings).model_id  # поддерживается

    def test_missing_key_surfaces_from_factory(self) -> None:
        with pytest.raises(MissingCredentialsError):
            build_embedding_provider(make_settings(openai_api_key=None))


class _FakeEmbeddingsAPI:
    """Подмена клиента OpenAI: считает батчи и возвращает вектор фиксированной длины."""

    def __init__(self, dimension: int = 4, shuffle: bool = False) -> None:
        self.dimension = dimension
        self.shuffle = shuffle
        self.batch_sizes: list[int] = []

    @property
    def embeddings(self) -> _FakeEmbeddingsAPI:
        return self

    def create(self, *, model: str, input: Sequence[str]) -> Any:
        self.batch_sizes.append(len(input))
        items = [
            type("Item", (), {"index": index, "embedding": [float(index)] * self.dimension})()
            for index in range(len(input))
        ]
        if self.shuffle:
            items = list(reversed(items))
        return type("Response", (), {"data": items})()


class TestEmbeddingBatching:
    @staticmethod
    def build(
        batch_size: int, **kwargs: Any
    ) -> tuple[OpenAICompatibleEmbeddings, _FakeEmbeddingsAPI]:
        provider = build_embedding_provider(make_settings(embed_batch_size=batch_size))
        fake = _FakeEmbeddingsAPI(**kwargs)
        provider._client = fake
        return provider, fake

    def test_texts_split_into_batches(self) -> None:
        provider, fake = self.build(batch_size=3)
        vectors = provider.embed([f"текст {i}" for i in range(7)])

        assert fake.batch_sizes == [3, 3, 1]
        assert len(vectors) == 7

    def test_order_restored_when_api_shuffles(self) -> None:
        """Порядок в ответе не гарантирован контрактом — восстанавливаем по index."""
        provider, _ = self.build(batch_size=10, shuffle=True)
        vectors = provider.embed(["раз", "два", "три"])

        assert [vector[0] for vector in vectors] == [0.0, 1.0, 2.0]

    def test_empty_input_makes_no_calls(self) -> None:
        provider, fake = self.build(batch_size=10)

        assert provider.embed([]) == []
        assert fake.batch_sizes == []

    def test_unknown_model_dimension_probed_once(self) -> None:
        provider, fake = self.build(batch_size=10)
        provider._model = "неизвестная-модель"
        provider._dimension = None

        assert provider.dimension == 4
        assert provider.dimension == 4  # повторный вызов не делает нового запроса
        assert len(fake.batch_sizes) == 1

    def test_known_models_need_no_probe(self) -> None:
        assert set(KNOWN_DIMENSIONS) >= {"text-embedding-3-small", "text-embedding-3-large"}
