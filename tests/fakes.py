"""
Заглушки провайдеров и фабрики тестовых данных.

Вынесены из тестов поиска, когда те же подделки понадобились тестам генерации:
эмбеддер с управляемым сходством и модель с заранее заданным ответом нужны
обоим, и держать две копии значит однажды поправить только одну.
"""

from __future__ import annotations

from collections.abc import Sequence

from zerocoder_assistant.llm.base import ChatMessage
from zerocoder_assistant.preprocessing.models import Chunk, NoteMetadata, content_hash


class FakeEmbedder:
    """Возвращает заранее заданный вектор для каждого запроса."""

    def __init__(self, vectors: dict[str, list[float]] | None = None) -> None:
        self.vectors = vectors or {}
        self.calls = 0

    @property
    def model_id(self) -> str:
        return "fake-embed-v1"

    @property
    def dimension(self) -> int:
        return 2

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += len(texts)
        return [self.vectors.get(text, [1.0, 0.0]) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class FakeLLM:
    """Отдаёт заготовленный ответ и запоминает, с чем её позвали."""

    def __init__(self, reply: str = "Ответ по фрагменту [1].") -> None:
        self.reply = reply
        self.calls: list[list[ChatMessage]] = []

    @property
    def model_id(self) -> str:
        return "fake-llm-v1"

    @property
    def last_messages(self) -> list[ChatMessage]:
        return self.calls[-1]

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append(list(messages))
        return self.reply


def make_chunk(index: int, text: str, lesson_id: str = "PEr08") -> Chunk:
    """Чанк с заполненными метаданными — ровно то, что кладётся в хранилище."""
    note = NoteMetadata(
        source_file=f"{lesson_id}.md",
        lesson_title=f"{lesson_id}. Урок",
        lesson_id=lesson_id,
        module_num=5,
    )
    return Chunk(
        chunk_id=f"{lesson_id}:s000:c{index:02d}",
        text=text,
        note=note,
        heading_path=(f"{lesson_id}. Урок", "Теория"),
        section_title="Теория",
        chunk_index=index,
        chunks_in_section=1,
        content_type="theory",
        token_count=len(text),
        hash=content_hash(text),
    )
