"""Модели данных препроцессинга: от файла конспекта до готового чанка."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from zerocoder_assistant.config.constants import HEADING_PATH_SEPARATOR

#: Длина усечённого sha256, используемого как content_hash.
#: 16 hex-символов = 64 бита: коллизия на корпусе в тысячи чанков исключена.
CONTENT_HASH_LENGTH = 16


def content_hash(text: str) -> str:
    """Устойчивый хеш содержимого — основа идемпотентной переиндексации."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:CONTENT_HASH_LENGTH]


@dataclass(frozen=True, slots=True)
class NoteMetadata:
    """Паспорт конспекта: то, что известно о файле до разбора его содержимого."""

    source_file: str
    lesson_title: str
    lesson_id: str | None = None
    module: str | None = None
    module_num: int | None = None
    notes_version: int = 1

    def as_dict(self) -> dict[str, Any]:
        """Плоское представление без None — для метаданных векторного хранилища."""
        data: dict[str, Any] = {
            "source_file": self.source_file,
            "lesson_title": self.lesson_title,
            "notes_version": self.notes_version,
        }
        if self.lesson_id is not None:
            data["lesson_id"] = self.lesson_id
        if self.module is not None:
            data["module"] = self.module
        if self.module_num is not None:
            data["module_num"] = self.module_num
        return data


@dataclass(frozen=True, slots=True)
class Block:
    """Неделимая единица текста внутри секции: абзац, список, таблица или код.

    Блок кода атомарен: разрыв внутри ``` даёт бессмысленный чанк, поэтому такой
    блок допускается превысить максимальный размер, но не допускается разрезать.
    """

    text: str
    is_code: bool = False


@dataclass(frozen=True, slots=True)
class Section:
    """Раздел конспекта, ограниченный заголовком."""

    heading: str | None
    level: int
    heading_path: tuple[str, ...]
    body: str
    ordinal: int

    @property
    def title(self) -> str | None:
        return self.heading


@dataclass(frozen=True, slots=True)
class Chunk:
    """Готовый к векторизации фрагмент."""

    chunk_id: str
    text: str
    note: NoteMetadata
    heading_path: tuple[str, ...]
    section_title: str | None
    chunk_index: int
    chunks_in_section: int
    content_type: str
    token_count: int
    hash: str

    def fingerprint(self) -> str:
        """Отпечаток всего, что уезжает в хранилище: текста И паспорта.

        Идемпотентность сравнивает чанки по отпечатку, а в базу пишется не один
        текст. Пока отпечаток считался только по тексту, любая правка
        метаданных без правки текста навсегда оставалась в файле и не доезжала
        до индекса: сборка честно докладывала `unchanged`, `embedded=0` — и это
        выглядело как успешный повтор.

        Проявлялось это двумя способами. Дописанный frontmatter (`lesson_id`,
        `module_num`) не менял ничего, и фильтры `--lesson` / `--module`
        продолжали искать по старым значениям. А выросшая секция оставляла
        первому чанку прежний `chunks_in_section`, и в списке источников
        появлялось «часть 1/2» рядом с «часть 2/5».

        Из отпечатка исключён сам `content_hash` — иначе он зависел бы от себя.
        """
        payload = {key: value for key, value in self.as_metadata().items() if key != "content_hash"}
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return content_hash(f"{serialized}\n{self.text}")

    def as_metadata(self) -> dict[str, Any]:
        """Плоские метаданные для векторного хранилища.

        Chroma принимает только скаляры, поэтому путь заголовков сериализуется
        в строку тем же разделителем, что используется в contextual header.
        """
        data = self.note.as_dict()
        data.update(
            {
                "heading_path": HEADING_PATH_SEPARATOR.join(self.heading_path),
                "chunk_index": self.chunk_index,
                "chunks_in_section": self.chunks_in_section,
                "content_type": self.content_type,
                "token_count": self.token_count,
                "content_hash": self.hash,
            }
        )
        if self.section_title is not None:
            data["section_title"] = self.section_title
        return data


@dataclass(frozen=True, slots=True)
class ProcessedNote:
    """Результат препроцессинга одного файла конспекта."""

    note: NoteMetadata
    chunks: list[Chunk]
    skipped_sections: list[str] = field(default_factory=list)
    dropped_boilerplate: int = 0
