"""
Паспорт конспекта: что выводится из пути, что берётся из frontmatter.

Ни один конспект корпуса frontmatter не имеет, поэтому вывод из имени файла и
папки — не запасной путь, а основной. Ошибка здесь тихая: неверный `lesson_id`
не ломает индексацию, он ломает фильтры и ответ на вопрос «в каком уроке это
было», причём заметно это станет далеко от места ошибки.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from zerocoder_assistant.preprocessing.metadata import (
    build_note_metadata,
    extract_lesson_id,
    extract_module_number,
)
from zerocoder_assistant.preprocessing.models import Section

NOTES = Path("C:/корпус")


def section(heading: str, level: int = 1) -> Section:
    return Section(heading=heading, level=level, heading_path=(heading,), body="текст", ordinal=0)


class TestLessonId:
    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("PEr08_практика", "PEr08"),
            ("PEs06.1_разбор", "PEs06.1"),
            ("SMPb01_основы", "SMPb01"),
            ("PEnB10_интеграции", "PEnB10"),
            ("PEW02_право", "PEW02"),
        ],
    )
    def test_recognised_forms(self, filename: str, expected: str) -> None:
        assert extract_lesson_id(filename) == expected

    @pytest.mark.parametrize("filename", ["заметка", "2024_отчёт", ""])
    def test_no_identifier(self, filename: str) -> None:
        assert extract_lesson_id(filename) is None


class TestModuleNumber:
    @pytest.mark.parametrize(
        ("folder", "expected"),
        [
            ("Модуль 5. Ассистенты с памятью", 5),
            ("модуль 0. Вводно-правовой", 0),
            ("Модуля 12", 12),
        ],
    )
    def test_number_parsed(self, folder: str, expected: int) -> None:
        assert extract_module_number(folder) == expected

    @pytest.mark.parametrize("folder", ["Бонусный модуль", None, "", "Разное"])
    def test_number_absent(self, folder: str | None) -> None:
        """У бонусного модуля номера нет, и придумывать его нельзя."""
        assert extract_module_number(folder) is None


class TestNoteMetadata:
    def test_derived_from_the_path(self) -> None:
        note = build_note_metadata(
            NOTES / "Модуль 5. Ассистенты" / "PEr08_практика.md", NOTES, {}, []
        )

        assert note.lesson_id == "PEr08"
        assert note.module_num == 5
        assert note.module == "Модуль 5. Ассистенты"
        assert note.source_file == "Модуль 5. Ассистенты/PEr08_практика.md"

    def test_frontmatter_wins_over_the_path(self) -> None:
        """Иначе конспекты, порождённые P1, нельзя было бы разложить иначе."""
        note = build_note_metadata(
            NOTES / "Модуль 5. Ассистенты" / "PEr08_практика.md",
            NOTES,
            {"lesson_id": "PEr09", "module_num": 7, "title": "Своё название"},
            [section("Заголовок файла")],
        )

        assert note.lesson_id == "PEr09"
        assert note.module_num == 7
        assert note.lesson_title == "Своё название"

    def test_title_falls_back_to_the_first_heading(self) -> None:
        note = build_note_metadata(
            NOTES / "PEr08_практика.md", NOTES, {}, [section("Практика: ассистент")]
        )

        assert note.lesson_title == "Практика: ассистент"

    def test_title_falls_back_to_the_filename(self) -> None:
        """Ни frontmatter, ни заголовка — остаётся имя файла, но человеческое."""
        note = build_note_metadata(NOTES / "PEr08_практика_по_RAG.md", NOTES, {}, [])

        assert note.lesson_title == "практика по RAG"

    def test_second_level_heading_is_not_the_title(self) -> None:
        note = build_note_metadata(
            NOTES / "PEr08_практика.md", NOTES, {}, [section("Теория на сегодня", level=2)]
        )

        assert note.lesson_title == "практика"

    def test_file_outside_the_corpus_does_not_crash(self) -> None:
        """Путь вне NOTES_DIR — повод работать с тем, что есть, а не падать."""
        note = build_note_metadata(Path("D:/другое/PEr08_практика.md"), NOTES, {}, [])

        assert note.source_file == "PEr08_практика.md"
        assert note.module is None
        assert note.lesson_id == "PEr08"

    def test_notes_version_defaults_to_one(self) -> None:
        assert build_note_metadata(NOTES / "PEr08.md", NOTES, {}, []).notes_version == 1
