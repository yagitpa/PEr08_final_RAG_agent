"""
Разбиение секции на чанки.

Стратегия — структурная, а не «нарезать текст каждые N символов»:

1. Единица — секция конспекта. Уместилась в лимит — стала одним чанком.
2. Не уместилась — набирается из блоков (абзац, список, таблица, листинг).
3. Блок длиннее лимита дробится по предложениям, а не по произвольной границе.
4. Листинг неделим: разрыв внутри ``` даёт бессмысленный фрагмент.
5. Перекрытие набирается целыми предложениями с конца предыдущего чанка.

К тексту чанка спереди приписывается путь заголовков (contextual header).
Без него фрагмент — безымянный кусок текста; с ним и эмбеддер, и модель видят,
из какого урока и раздела он взят.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from zerocoder_assistant.config.constants import HEADING_PATH_SEPARATOR
from zerocoder_assistant.config.settings import ChunkingConfig
from zerocoder_assistant.preprocessing.headings import classify_content_type, sanitize_heading
from zerocoder_assistant.preprocessing.markdown import split_sentences
from zerocoder_assistant.preprocessing.models import (
    Block,
    Chunk,
    NoteMetadata,
    Section,
    content_hash,
)
from zerocoder_assistant.preprocessing.tokenization import TokenCounter, get_token_counter

#: Разделитель между contextual header и телом чанка.
BODY_SEPARATOR = "\n\n"

#: Склейка соседних блоков и соседних предложений одного блока.
BLOCK_GLUE = "\n\n"
SENTENCE_GLUE = " "

#: Нижняя граница бюджета: длинный путь заголовков не должен съесть чанк целиком.
MIN_BUDGET_TOKENS = 50


@dataclass(frozen=True, slots=True)
class _Atom:
    """Минимальная неразрывная единица набора: блок целиком или предложение."""

    text: str
    is_code: bool
    glue: str


@dataclass
class _Draft:
    """Накопитель чанка в процессе набора."""

    atoms: list[_Atom] = field(default_factory=list)
    tokens: int = 0
    code_chars: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.atoms

    @property
    def last_is_code(self) -> bool:
        return bool(self.atoms) and self.atoms[-1].is_code

    def add(self, atom: _Atom, tokens: int) -> None:
        self.atoms.append(atom)
        self.tokens += tokens
        if atom.is_code:
            self.code_chars += len(atom.text)

    def render(self) -> str:
        parts: list[str] = []
        for index, atom in enumerate(self.atoms):
            if index:
                parts.append(atom.glue)
            parts.append(atom.text)
        return "".join(parts).strip()


class Chunker:
    """Собирает чанки из очищенных блоков секции."""

    def __init__(self, config: ChunkingConfig, counter: TokenCounter | None = None) -> None:
        self._config = config
        self._counter = counter or get_token_counter(config.encoding)

    def chunk_section(
        self,
        blocks: list[Block],
        section: Section,
        note: NoteMetadata,
    ) -> list[Chunk]:
        """Чанки одной секции. Пустая секция даёт пустой список."""
        usable = [block for block in blocks if block.text.strip()]
        if not usable:
            return []

        # Заголовки санируются один раз: этот текст попадёт и в contextual header,
        # и в метаданные, поэтому эмодзи-маркерам вроде «↘️» там не место.
        heading_path = tuple(sanitize_heading(item) for item in section.heading_path)
        section_title = sanitize_heading(section.heading) if section.heading else None

        header = self._build_header(heading_path, note)
        header_tokens = self._counter.count(header + BODY_SEPARATOR) if header else 0
        target = max(MIN_BUDGET_TOKENS, self._config.target_tokens - header_tokens)
        maximum = max(MIN_BUDGET_TOKENS, self._config.max_tokens - header_tokens)

        drafts = self._pack(self._to_atoms(usable, target), target)
        drafts = self._merge_undersized(drafts, maximum)

        return [
            self._build_chunk(
                draft=draft,
                index=index,
                total=len(drafts),
                header=header,
                heading_path=heading_path,
                section_title=section_title,
                ordinal=section.ordinal,
                note=note,
            )
            for index, draft in enumerate(drafts)
        ]

    # -- сборка ------------------------------------------------------------

    @staticmethod
    def _build_header(heading_path: tuple[str, ...], note: NoteMetadata) -> str:
        """Contextual header: «PEr08. Практика > Теория на сегодня > 4. Параметры RAG»."""
        if heading_path:
            return HEADING_PATH_SEPARATOR.join(heading_path)
        return note.lesson_title

    def _to_atoms(self, blocks: list[Block], target: int) -> list[_Atom]:
        """Раскладывает блоки в неразрывные единицы набора.

        Листинг остаётся целым при любом размере. Слишком длинный текстовый блок
        распадается на предложения, но первое из них сохраняет «абзацную» склейку,
        чтобы граница абзаца не потерялась при сборке.
        """
        atoms: list[_Atom] = []
        for block in blocks:
            if block.is_code or self._counter.count(block.text) <= target:
                atoms.append(_Atom(block.text, block.is_code, BLOCK_GLUE))
                continue

            sentences = split_sentences(block.text)
            if not sentences:
                atoms.append(_Atom(block.text, False, BLOCK_GLUE))
                continue

            atoms.append(_Atom(sentences[0], False, BLOCK_GLUE))
            atoms.extend(_Atom(text, False, SENTENCE_GLUE) for text in sentences[1:])
        return atoms

    def _pack(self, atoms: list[_Atom], target: int) -> list[_Draft]:
        """Жадно набирает чанки до целевого размера, добавляя перекрытие."""
        drafts: list[_Draft] = []
        current = _Draft()

        for atom in atoms:
            tokens = self._counter.count(atom.text)

            if current.is_empty or current.tokens + tokens <= target:
                current.add(atom, tokens)
                continue

            drafts.append(current)
            current = _Draft()
            for carried in self._overlap_atoms(drafts[-1]):
                current.add(carried, self._counter.count(carried.text))
            current.add(atom, tokens)

        if not current.is_empty:
            drafts.append(current)
        return drafts

    def _overlap_atoms(self, draft: _Draft) -> list[_Atom]:
        """Хвост предыдущего чанка, переносимый в следующий.

        Переносятся только целые предложения и только если чанк не заканчивается
        листингом: дублировать код бессмысленно. Перенести весь чанк целиком
        нельзя — иначе набор зациклится на копировании самого себя.
        """
        budget = self._config.overlap_tokens
        if budget <= 0 or draft.last_is_code:
            return []

        picked: list[_Atom] = []
        total = 0
        for atom in reversed(draft.atoms):
            if atom.is_code:
                break
            tokens = self._counter.count(atom.text)
            if total + tokens > budget:
                break
            picked.insert(0, atom)
            total += tokens

        if len(picked) == len(draft.atoms):
            picked = picked[1:]
        return picked

    def _merge_undersized(self, drafts: list[_Draft], maximum: int) -> list[_Draft]:
        """Приклеивает слишком мелкие чанки к предыдущему, если тот выдержит.

        Мелкий чанк не выбрасывается: потерять содержимое хуже, чем оставить
        короткий фрагмент. Если склейка не влезает — он остаётся как есть и
        попадает в отчёт как выход за коридор размеров.
        """
        if len(drafts) < 2:
            return drafts

        merged: list[_Draft] = [drafts[0]]
        for draft in drafts[1:]:
            previous = merged[-1]
            fits = previous.tokens + draft.tokens <= maximum
            if draft.tokens < self._config.min_tokens and fits:
                for atom in draft.atoms:
                    previous.add(atom, self._counter.count(atom.text))
                continue
            merged.append(draft)
        return merged

    def _build_chunk(
        self,
        *,
        draft: _Draft,
        index: int,
        total: int,
        header: str,
        heading_path: tuple[str, ...],
        section_title: str | None,
        ordinal: int,
        note: NoteMetadata,
    ) -> Chunk:
        body = draft.render()
        text = f"{header}{BODY_SEPARATOR}{body}" if header else body
        code_ratio = draft.code_chars / len(body) if body else 0.0

        return Chunk(
            chunk_id=f"{note.source_file}:s{ordinal:03d}:c{index:02d}",
            text=text,
            note=note,
            heading_path=heading_path,
            section_title=section_title,
            chunk_index=index,
            chunks_in_section=total,
            content_type=classify_content_type(section_title, body, code_ratio),
            token_count=self._counter.count(text),
            hash=content_hash(text),
        )
