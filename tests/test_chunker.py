"""Чанкинг: размеры, атомарность кода, перекрытие, слияние секций."""

from __future__ import annotations

import pytest

from zerocoder_assistant.config.settings import ChunkingConfig
from zerocoder_assistant.preprocessing.chunker import Chunker
from zerocoder_assistant.preprocessing.models import Block, NoteMetadata, Section
from zerocoder_assistant.preprocessing.section_merger import PreparedSection, SectionMerger
from zerocoder_assistant.preprocessing.tokenization import get_token_counter

SENTENCE = "Векторный поиск находит фрагменты по смысловой близости, а не по словам."


@pytest.fixture
def note() -> NoteMetadata:
    return NoteMetadata(
        source_file="Модуль 5/PEr08_Практика.md",
        lesson_title="PEr08. Практика",
        lesson_id="PEr08",
        module="Модуль 5",
        module_num=5,
    )


DEFAULT_PATH = ("PEr08. Практика", "Теория")


def make_section(ordinal: int = 3, path: tuple[str, ...] = DEFAULT_PATH) -> Section:
    return Section(
        heading=path[-1] if path else None,
        level=len(path),
        heading_path=path,
        body="",
        ordinal=ordinal,
    )


@pytest.fixture
def chunker(chunking: ChunkingConfig) -> Chunker:
    return Chunker(chunking, get_token_counter(chunking.encoding))


class TestChunkSection:
    def test_short_section_becomes_single_chunk(self, chunker: Chunker, note: NoteMetadata) -> None:
        chunks = chunker.chunk_section([Block(SENTENCE)], make_section(), note)

        assert len(chunks) == 1
        assert chunks[0].chunks_in_section == 1

    def test_contextual_header_prefixes_text(self, chunker: Chunker, note: NoteMetadata) -> None:
        chunks = chunker.chunk_section([Block(SENTENCE)], make_section(), note)

        assert chunks[0].text.startswith("PEr08. Практика > Теория")

    def test_header_falls_back_to_lesson_title(self, chunker: Chunker, note: NoteMetadata) -> None:
        chunks = chunker.chunk_section([Block(SENTENCE)], make_section(path=()), note)

        assert chunks[0].text.startswith("PEr08. Практика")

    def test_long_section_splits_within_limit(
        self, chunker: Chunker, chunking: ChunkingConfig, note: NoteMetadata
    ) -> None:
        blocks = [Block(" ".join([SENTENCE] * 12)) for _ in range(6)]
        chunks = chunker.chunk_section(blocks, make_section(), note)

        assert len(chunks) > 1
        assert all(chunk.token_count <= chunking.max_tokens for chunk in chunks)
        assert all(chunk.chunks_in_section == len(chunks) for chunk in chunks)

    def test_empty_blocks_produce_nothing(self, chunker: Chunker, note: NoteMetadata) -> None:
        assert chunker.chunk_section([Block("   ")], make_section(), note) == []

    def test_chunk_ids_unique_and_file_scoped(self, chunker: Chunker, note: NoteMetadata) -> None:
        """Идентификатор включает путь файла: lesson_id в корпусе не уникален."""
        blocks = [Block(" ".join([SENTENCE] * 12)) for _ in range(6)]
        chunks = chunker.chunk_section(blocks, make_section(), note)

        assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
        assert all(chunk.chunk_id.startswith(note.source_file) for chunk in chunks)


class TestCodeAtomicity:
    def test_code_block_is_never_split(
        self, chunker: Chunker, chunking: ChunkingConfig, note: NoteMetadata
    ) -> None:
        listing = "```python\n" + "\n".join(f"value_{i} = {i}" for i in range(300)) + "\n```"
        chunks = chunker.chunk_section([Block(listing, is_code=True)], make_section(), note)

        assert len(chunks) == 1
        assert chunks[0].token_count > chunking.max_tokens  # атомарность важнее лимита
        assert chunks[0].text.count("```") == 2

    def test_fences_stay_balanced_across_chunks(self, chunker: Chunker, note: NoteMetadata) -> None:
        listing = "```python\nx = 1\n```"
        blocks = [Block(" ".join([SENTENCE] * 10)), Block(listing, is_code=True)] * 4
        chunks = chunker.chunk_section(blocks, make_section(), note)

        assert all(chunk.text.count("```") % 2 == 0 for chunk in chunks)


class TestOverlap:
    def test_consecutive_chunks_share_tail(self, chunker: Chunker, note: NoteMetadata) -> None:
        sentences = [f"Предложение номер {i} про устройство RAG-системы." for i in range(60)]
        chunks = chunker.chunk_section([Block(" ".join(sentences))], make_section(), note)

        assert len(chunks) > 1
        tail_words = set(chunks[0].text.split())
        assert tail_words & set(chunks[1].text.split())

    def test_zero_overlap_disables_carryover(self, note: NoteMetadata) -> None:
        config = ChunkingConfig(overlap_pct=0)
        chunker = Chunker(config, get_token_counter(config.encoding))
        sentences = [f"Предложение номер {i} про устройство RAG-системы." for i in range(60)]
        chunks = chunker.chunk_section([Block(" ".join(sentences))], make_section(), note)

        first_tail = chunks[0].text.strip().rsplit(".", 2)[-2]
        assert first_tail not in chunks[1].text


class TestSectionMerger:
    @pytest.fixture
    def merger(self, chunking: ChunkingConfig) -> SectionMerger:
        return SectionMerger(chunking, get_token_counter(chunking.encoding))

    @staticmethod
    def prepared(name: str, tokens: int, parent: str = "Теория") -> PreparedSection:
        section = Section(
            heading=name,
            level=3,
            heading_path=("Урок", parent, name),
            body="",
            ordinal=hash(name) % 100,
        )
        return PreparedSection(section=section, blocks=[Block(SENTENCE)], tokens=tokens)

    def test_small_siblings_are_merged(self, merger: SectionMerger) -> None:
        result = merger.merge([self.prepared("Промпты", 60), self.prepared("Чат", 70)])

        assert len(result) == 1
        assert result[0].section.heading == "Теория"

    def test_merged_body_keeps_child_headings(self, merger: SectionMerger) -> None:
        result = merger.merge([self.prepared("Промпты", 60), self.prepared("Чат", 70)])
        texts = [block.text for block in result[0].blocks]

        assert "Промпты" in texts
        assert "Чат" in texts

    def test_different_parents_not_merged(self, merger: SectionMerger) -> None:
        result = merger.merge(
            [self.prepared("Промпты", 60), self.prepared("Кэш", 70, parent="Практика")]
        )

        assert len(result) == 2

    def test_large_section_left_alone(self, merger: SectionMerger) -> None:
        result = merger.merge([self.prepared("Большая", 450), self.prepared("Чат", 70)])

        assert len(result) == 2

    def test_group_does_not_exceed_target(
        self, merger: SectionMerger, chunking: ChunkingConfig
    ) -> None:
        result = merger.merge([self.prepared(f"Раздел {i}", 150) for i in range(6)])

        assert all(item.tokens <= chunking.target_tokens for item in result)

    def test_top_level_sections_not_merged(self, merger: SectionMerger) -> None:
        """У секций верхнего уровня нет общего родителя, кроме документа."""
        shallow = [
            PreparedSection(
                section=Section(heading=name, level=1, heading_path=(name,), body="", ordinal=i),
                blocks=[Block(SENTENCE)],
                tokens=50,
            )
            for i, name in enumerate(("A", "B"))
        ]

        assert len(merger.merge(shallow)) == 2
