"""Поставщики эмбеддингов."""

from zerocoder_assistant.embeddings.base import EmbeddingProvider
from zerocoder_assistant.embeddings.factory import build_embedding_provider

__all__ = ["EmbeddingProvider", "build_embedding_provider"]
