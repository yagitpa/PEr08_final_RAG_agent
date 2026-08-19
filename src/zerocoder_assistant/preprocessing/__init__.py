"""
Препроцессинг конспектов: Markdown -> очищенные чанки с метаданными.

Публичный контракт пакета — `NotePreprocessor` и `iter_note_files`. Остальные
модули (разбор Markdown, очистка, чанкинг) доступны, но вызывать их напрямую
нужно только в тестах и при отладке.
"""

from zerocoder_assistant.preprocessing.chunker import Chunker
from zerocoder_assistant.preprocessing.cleaner import TextCleaner
from zerocoder_assistant.preprocessing.models import (
    Block,
    Chunk,
    NoteMetadata,
    ProcessedNote,
    Section,
    content_hash,
)
from zerocoder_assistant.preprocessing.pipeline import NotePreprocessor, iter_note_files
from zerocoder_assistant.preprocessing.tokenization import TokenCounter, get_token_counter

__all__ = [
    "Block",
    "Chunk",
    "Chunker",
    "NoteMetadata",
    "NotePreprocessor",
    "ProcessedNote",
    "Section",
    "TextCleaner",
    "TokenCounter",
    "content_hash",
    "get_token_counter",
    "iter_note_files",
]
