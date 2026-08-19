"""
Определение паспорта конспекта.

Ни один из существующих конспектов не имеет YAML-frontmatter — все начинаются
сразу с заголовка первого уровня. Поэтому метаданные выводятся из расположения
и имени файла, а frontmatter, если он есть, только переопределяет выведенное.
Так конвейер работает и с уже написанными конспектами, и с теми, что будет
порождать P1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from zerocoder_assistant.config.constants import LESSON_ID_PATTERN, MODULE_NUMBER_PATTERN
from zerocoder_assistant.preprocessing.models import NoteMetadata, Section

#: Уровень заголовка, который считается названием урока.
TITLE_HEADING_LEVEL = 1

DEFAULT_NOTES_VERSION = 1


def extract_lesson_id(filename: str) -> str | None:
    """Идентификатор урока из имени файла: PEr08, PEs06.1, SMPb01, PEnB10."""
    match = LESSON_ID_PATTERN.match(filename)
    return match.group(1) if match else None


def extract_module_number(module: str | None) -> int | None:
    """Номер модуля из имени папки. У «Бонусного модуля» номера нет."""
    if not module:
        return None
    match = MODULE_NUMBER_PATTERN.search(module)
    return int(match.group(1)) if match else None


def _relative_parts(path: Path, notes_dir: Path) -> tuple[str, tuple[str, ...]]:
    """Путь файла относительно корня конспектов и промежуточные папки."""
    try:
        relative = path.resolve().relative_to(notes_dir.resolve())
    except ValueError:
        # Файл вне notes_dir — работаем с тем, что есть, вместо падения.
        return path.name, ()
    return relative.as_posix(), relative.parts[:-1]


def _humanize_filename(stem: str, lesson_id: str | None) -> str:
    """Запасное название урока, если в файле нет ни frontmatter, ни заголовка."""
    title = stem
    if lesson_id and title.startswith(lesson_id):
        title = title[len(lesson_id) :]
    return title.replace("_", " ").strip(" -") or stem


def _title_from_sections(sections: list[Section]) -> str | None:
    for section in sections:
        if section.level == TITLE_HEADING_LEVEL and section.heading:
            return section.heading
    return None


def build_note_metadata(
    path: Path,
    notes_dir: Path,
    frontmatter: dict[str, Any],
    sections: list[Section],
) -> NoteMetadata:
    """Собирает паспорт конспекта: frontmatter поверх выведенного из пути."""
    source_file, folders = _relative_parts(path, notes_dir)

    lesson_id = frontmatter.get("lesson_id") or extract_lesson_id(path.stem)
    module = frontmatter.get("module") or (folders[0] if folders else None)
    module_num = frontmatter.get("module_num")
    if module_num is None:
        module_num = extract_module_number(module)

    title = (
        frontmatter.get("title")
        or _title_from_sections(sections)
        or _humanize_filename(path.stem, lesson_id)
    )

    return NoteMetadata(
        source_file=source_file,
        lesson_title=str(title),
        lesson_id=str(lesson_id) if lesson_id else None,
        module=str(module) if module else None,
        module_num=int(module_num) if module_num is not None else None,
        notes_version=int(frontmatter.get("notes_version", DEFAULT_NOTES_VERSION)),
    )
