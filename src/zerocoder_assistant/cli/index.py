"""
Команды группы `index` — всё, что относится к превращению конспектов в индекс.

Сейчас здесь одна команда, `preview`: сухой прогон препроцессинга без обращения
к эмбеддеру и хранилищу. Это инструмент верификации этапа 1 (docs/architecture.md,
«Верификация», п. 1) и одновременно способ подбирать параметры чанкинга, не платя
за векторизацию. Команды `build` и `stats` появятся на этапе 3 в этом же файле.
"""

from __future__ import annotations

import json
import logging
import random
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Final

import click

from zerocoder_assistant.config.settings import get_settings
from zerocoder_assistant.reporting import (
    build_corpus_stats,
    render_chunk_sample,
    render_histogram,
    render_stats,
)

if TYPE_CHECKING:  # pragma: no cover - только для аннотаций
    from collections.abc import Iterable, Iterator, Sequence

    from zerocoder_assistant.preprocessing import NotePreprocessor
    from zerocoder_assistant.preprocessing.models import Chunk, ProcessedNote

logger = logging.getLogger(__name__)

#: Сколько примеров чанков показывать без явного `--samples`.
#: Верификация требует просмотреть десяток чанков; по умолчанию берём меньше,
#: чтобы сводка не тонула в тексте, а полный просмотр запрашивался осознанно.
DEFAULT_SAMPLE_COUNT: Final[int] = 3

#: Зерно генератора для выбора примеров. Фиксировано, чтобы два прогона с одними
#: параметрами давали один отчёт: иначе изменения в выводе невозможно приписать
#: изменению кода.
SAMPLE_SEED: Final[int] = 20250819

RULE: Final[str] = "-" * 72


@click.group(name="index")
def index_group() -> None:
    """Индексация конспектов: препроцессинг, сборка и статистика индекса."""


@index_group.command(name="preview")
@click.option(
    "--notes-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Каталог конспектов; по умолчанию NOTES_DIR из настроек.",
)
@click.option(
    "--lesson",
    "lessons",
    multiple=True,
    metavar="LESSON_ID",
    help="Оставить только указанные уроки (можно повторять): --lesson PEr06.",
)
@click.option(
    "--limit",
    type=click.IntRange(min=1),
    default=None,
    help="Прочитать только первые N файлов (применяется до фильтра --lesson).",
)
@click.option(
    "--samples",
    type=click.IntRange(min=0),
    default=DEFAULT_SAMPLE_COUNT,
    show_default=True,
    help="Сколько случайных чанков показать целиком.",
)
@click.option(
    "--export",
    "export_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Выгрузить все чанки в JSONL (id, text, metadata).",
)
@click.option("--no-histogram", is_flag=True, help="Не печатать гистограмму размеров.")
def preview(
    notes_dir: Path | None,
    lessons: tuple[str, ...],
    limit: int | None,
    samples: int,
    export_path: Path | None,
    no_histogram: bool,
) -> None:
    """Сухой прогон препроцессинга: статистика, гистограмма и примеры чанков.

    Ничего не эмбеддит и ничего не пишет в хранилище — только читает `.md`.
    """
    # Ядро препроцессинга импортируется лениво: оно тянет за собой tiktoken со
    # словарём BPE, и платить за это на `zassist --help` незачем.
    from zerocoder_assistant.preprocessing import NotePreprocessor, iter_note_files

    settings = get_settings()
    root = (notes_dir or settings.notes_dir).expanduser()
    if not root.is_dir():
        raise click.ClickException(f"Каталог конспектов не найден: {root}")

    config = settings.chunking
    preprocessor = NotePreprocessor(config)

    notes, failures = _process_notes(preprocessor, iter_note_files(root), root, limit)
    processed_count = len(notes)
    if lessons:
        notes = _filter_by_lesson(notes, lessons)

    click.echo(f"Конспекты: {root}")
    click.echo(RULE)

    if not notes:
        _report_failures(failures)
        if processed_count:
            raise click.ClickException(
                f"Фильтр --lesson не оставил ни одного конспекта из {processed_count}."
            )
        raise click.ClickException("Не обработано ни одного конспекта — отчёт пуст.")

    stats = build_corpus_stats(notes, config)
    click.echo(render_stats(stats, config))

    if not no_histogram:
        click.echo("\nРаспределение размеров чанков (токены):\n")
        click.echo(render_histogram(stats.token_counts))

    chunks = [chunk for note in notes for chunk in note.chunks]
    _echo_samples(chunks, samples)

    if export_path is not None:
        exported = _export_jsonl(chunks, export_path)
        click.echo(f"\nВыгружено чанков: {exported} -> {export_path}")

    _report_failures(failures)


# ---------------------------------------------------------------------------
# Шаги команды
# ---------------------------------------------------------------------------


def _process_notes(
    preprocessor: NotePreprocessor,
    paths: Iterator[Path],
    notes_dir: Path,
    limit: int | None,
) -> tuple[list[ProcessedNote], list[Path]]:
    """Обработать конспекты, не роняя прогон из-за одного файла.

    Смысл превью — увидеть корпус целиком, поэтому битый файл попадает в список
    сбоев и в лог, а обход продолжается.
    """
    notes: list[ProcessedNote] = []
    failures: list[Path] = []
    for path in islice(paths, limit):
        try:
            notes.append(preprocessor.process_file(path, notes_dir))
        except Exception as exc:
            # Трассировка уходит только в DEBUG: на корпусе в полсотни файлов
            # десяток стектрейсов вытесняет из терминала сам отчёт.
            logger.error("Не удалось обработать конспект %s: %s", path, exc)
            logger.debug("Подробности сбоя на %s", path, exc_info=True)
            failures.append(path)
    return notes, failures


def _filter_by_lesson(
    notes: Sequence[ProcessedNote], lessons: Iterable[str]
) -> list[ProcessedNote]:
    """Отбор по lesson_id.

    Сравнение без учёта регистра: в именах файлов встречаются и `PEr08`, и `per08`,
    а идентификатор из frontmatter — единственный достоверный источник, поэтому
    фильтр применяется после разбора, а не по имени файла.
    """
    wanted = {lesson.casefold() for lesson in lessons}
    return [
        note for note in notes if note.note.lesson_id and note.note.lesson_id.casefold() in wanted
    ]


def _echo_samples(chunks: Sequence[Chunk], samples: int) -> None:
    """Напечатать несколько случайных чанков для ручной проверки."""
    if not samples or not chunks:
        return
    rng = random.Random(SAMPLE_SEED)
    selected = rng.sample(list(chunks), k=min(samples, len(chunks)))
    click.echo(f"\nПримеры чанков ({len(selected)} из {len(chunks)}):")
    for chunk in selected:
        click.echo(RULE)
        click.echo(render_chunk_sample(chunk))


def _export_jsonl(chunks: Sequence[Chunk], path: Path) -> int:
    """Выгрузить чанки в JSONL.

    `ensure_ascii=False` обязателен: иначе весь русский корпус превращается в
    `\\uXXXX` и файл нельзя просмотреть глазами, ради чего выгрузка и делается.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for chunk in chunks:
            record = {
                "id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": chunk.as_metadata(),
            }
            stream.write(json.dumps(record, ensure_ascii=False))
            stream.write("\n")
    return len(chunks)


def _report_failures(failures: Sequence[Path]) -> None:
    if not failures:
        return
    click.echo(f"\nСбоев при обработке: {len(failures)}", err=True)
    for path in failures:
        click.echo(f"  {path}", err=True)
