"""
Конвейер препроцессинга: файл конспекта -> список чанков.

Порядок стадий: разбор -> отбор секций -> очистка -> слияние мелких секций ->
чанкинг. Очистка идёт после разбора не случайно: к этому моменту листинги уже
отделены от прозы, и каждый блок обрабатывается по своим правилам. Слияние — до
чанкинга, чтобы чанкер видел уже укрупнённые секции и не плодил обрывки.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

from zerocoder_assistant.config.settings import ChunkingConfig, get_settings
from zerocoder_assistant.preprocessing.chunker import Chunker
from zerocoder_assistant.preprocessing.cleaner import TextCleaner
from zerocoder_assistant.preprocessing.headings import is_excluded_section, sanitize_heading
from zerocoder_assistant.preprocessing.markdown import (
    split_blocks,
    split_frontmatter,
    split_sections,
)
from zerocoder_assistant.preprocessing.metadata import build_note_metadata
from zerocoder_assistant.preprocessing.models import (
    Block,
    Chunk,
    CleanedSection,
    ProcessedNote,
    Section,
)
from zerocoder_assistant.preprocessing.section_merger import PreparedSection, SectionMerger
from zerocoder_assistant.preprocessing.tokenization import get_token_counter

logger = logging.getLogger(__name__)

NOTE_GLOB = "*.md"


def iter_note_files(notes_dir: Path) -> Iterator[Path]:
    """Все файлы конспектов под указанным корнем, в устойчивом порядке."""
    yield from sorted(notes_dir.rglob(NOTE_GLOB))


def _as_cleaned(item: PreparedSection) -> CleanedSection:
    """Очищенные блоки секции, склеенные обратно в связный текст.

    Блоки разделяются пустой строкой — тем же, чем они были разделены в
    исходнике. Иначе абзац и следующий за ним листинг слипаются в одну строку,
    и посмотреть глазами, ради чего всё затевалось, не получится.
    """
    return CleanedSection(
        # Заголовки санируются так же, как в чанкере: дамп обязан показывать
        # то, что уедет в индекс. Иначе «Результат дня в эмодзи» стоял бы в
        # дампе с картинкой, а в contextual header чанка — без неё.
        heading_path=tuple(sanitize_heading(part) for part in item.section.heading_path),
        text="\n\n".join(block.text for block in item.blocks),
        tokens=item.tokens,
    )


class NotePreprocessor:
    """Превращает файл конспекта в набор чанков, готовых к векторизации."""

    def __init__(
        self,
        config: ChunkingConfig | None = None,
        cleaner: TextCleaner | None = None,
        chunker: Chunker | None = None,
        merger: SectionMerger | None = None,
    ) -> None:
        self._config = config or get_settings().chunking
        self._counter = get_token_counter(self._config.encoding)
        self._cleaner = cleaner or TextCleaner()
        self._chunker = chunker or Chunker(self._config, self._counter)
        self._merger = merger or SectionMerger(self._config, self._counter)

    def process_file(self, path: Path, notes_dir: Path) -> ProcessedNote:
        """Читает и обрабатывает файл конспекта."""
        return self.process_text(path.read_text(encoding="utf-8"), path, notes_dir)

    def process_text(self, text: str, path: Path, notes_dir: Path) -> ProcessedNote:
        """Обрабатывает содержимое конспекта (точка входа для тестов)."""
        frontmatter, body = split_frontmatter(text)
        sections = split_sections(body)
        note = build_note_metadata(path, notes_dir, frontmatter, sections)

        prepared: list[PreparedSection] = []
        skipped: list[str] = []
        dropped = 0

        for section in sections:
            excluded = self._excluded_heading(section)
            if excluded is not None:
                skipped.append(excluded)
                continue

            blocks, boilerplate = self._prepare_blocks(section)
            dropped += boilerplate
            if not blocks:
                continue
            tokens = sum(self._counter.count(block.text) for block in blocks)
            prepared.append(PreparedSection(section=section, blocks=blocks, tokens=tokens))

        chunks: list[Chunk] = []
        for item in self._merger.merge(prepared):
            chunks.extend(self._chunker.chunk_section(item.blocks, item.section, note))

        logger.debug(
            "%s: %d чанков, пропущено секций %d, шаблонных абзацев %d",
            note.source_file,
            len(chunks),
            len(skipped),
            dropped,
        )
        return ProcessedNote(
            note=note,
            chunks=chunks,
            skipped_sections=skipped,
            dropped_boilerplate=dropped,
            cleaned_sections=tuple(_as_cleaned(item) for item in prepared),
        )

    @staticmethod
    def _excluded_heading(section: Section) -> str | None:
        """Заголовок, из-за которого секция отбрасывается, либо None.

        Проверяется весь путь, а не только собственный заголовок: подраздел
        внутри «Домашнего задания» тоже служебный.
        """
        for heading in section.heading_path:
            if is_excluded_section(heading):
                return heading
        return None

    def _prepare_blocks(self, section: Section) -> tuple[list[Block], int]:
        """Блоки секции после отсева шаблонов, очистки и снятия админ-строк.

        Второе значение — сколько блоков отброшено *осмысленно*: как шаблон или
        как целиком административные. Блоки, опустевшие от обычной очистки
        (разделители, строки из одних эмодзи), сюда не попадают: это оформление,
        а не потерянное содержимое, и смешивать их в одном счётчике значило бы
        рапортовать о сотнях удалённых шаблонов там, где их десятки.
        """
        prepared: list[Block] = []
        dropped = 0

        for block in split_blocks(section.body):
            if not block.is_code and self._cleaner.is_boilerplate(block.text):
                dropped += 1
                continue

            cleaned = self._cleaner.clean_block(block)
            if cleaned.is_code:
                prepared.append(cleaned)
                continue

            without_admin = self._cleaner.drop_admin_lines(cleaned.text)
            if without_admin.strip():
                prepared.append(Block(without_admin, is_code=False))
            elif cleaned.text.strip():
                dropped += 1  # блок состоял только из административных строк

        return prepared, dropped
