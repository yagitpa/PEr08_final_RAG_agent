"""Разбор структуры Markdown."""

from __future__ import annotations

from zerocoder_assistant.preprocessing.markdown import (
    split_blocks,
    split_frontmatter,
    split_sections,
    split_sentences,
)


class TestSplitSections:
    def test_heading_path_tracks_nesting(self) -> None:
        text = "# Урок\n\nВступление.\n\n## Теория\n\nТело.\n\n### 1. Раздел\n\nЕщё тело."
        sections = split_sections(text)

        assert [item.heading for item in sections] == ["Урок", "Теория", "1. Раздел"]
        assert sections[-1].heading_path == ("Урок", "Теория", "1. Раздел")

    def test_sibling_heading_pops_stack(self) -> None:
        text = "# У\n\n## A\n\n### A1\n\nx\n\n## B\n\ny"
        paths = [item.heading_path for item in split_sections(text)]

        assert paths[-1] == ("У", "B")

    def test_hash_inside_code_is_not_a_heading(self) -> None:
        """Комментарий Python не должен разрывать секцию."""
        text = "# Урок\n\n```python\n# это комментарий, а не заголовок\nx = 1\n```\n\nПосле."
        sections = split_sections(text)

        assert len(sections) == 1
        assert "# это комментарий" in sections[0].body

    def test_preamble_before_first_heading(self) -> None:
        sections = split_sections("Текст до заголовка.\n\n# Урок\n\nТело.")

        assert sections[0].heading is None
        assert sections[0].heading_path == ()

    def test_empty_preamble_is_dropped(self) -> None:
        assert split_sections("# Урок\n\nТело.")[0].heading == "Урок"


class TestSplitBlocks:
    def test_code_block_stays_whole(self) -> None:
        body = "Абзац.\n\n```python\nx = 1\n\ny = 2\n```\n\nЕщё абзац."
        blocks = split_blocks(body)

        assert [block.is_code for block in blocks] == [False, True, False]
        assert "y = 2" in blocks[1].text

    def test_blank_line_inside_fence_does_not_split(self) -> None:
        blocks = split_blocks("```\nа\n\nб\n```")

        assert len(blocks) == 1
        assert blocks[0].is_code

    def test_unclosed_fence_is_kept_as_code(self) -> None:
        """Потерять содержимое хуже, чем сохранить незакрытый блок."""
        blocks = split_blocks("Текст.\n\n```python\nx = 1")

        assert len(blocks) == 2
        assert blocks[1].is_code

    def test_table_is_single_block(self) -> None:
        body = "| a | b |\n| --- | --- |\n| 1 | 2 |"

        assert len(split_blocks(body)) == 1


class TestSplitFrontmatter:
    def test_parses_yaml(self) -> None:
        data, body = split_frontmatter("---\nlesson_id: PEr08\n---\n# Урок")

        assert data == {"lesson_id": "PEr08"}
        assert body.startswith("# Урок")

    def test_absent_frontmatter_returns_text_intact(self) -> None:
        data, body = split_frontmatter("# Урок\n\nТело.")

        assert data == {}
        assert body.startswith("# Урок")

    def test_broken_yaml_is_ignored_not_raised(self) -> None:
        source = "---\n: : :\n---\n# Урок"
        data, body = split_frontmatter(source)

        assert data == {}
        assert body == source


class TestSplitSentences:
    def test_splits_on_terminators(self) -> None:
        result = split_sentences("Первое. Второе! Третье?")

        assert result == ["Первое.", "Второе!", "Третье?"]

    def test_abbreviation_does_not_split(self) -> None:
        """«и т.д. Затем» — одно предложение, а не два."""
        result = split_sentences("Берём чанки, эмбеддинги и т.д. Затем ищем.")

        assert len(result) == 1

    def test_list_numbering_does_not_split(self) -> None:
        """«1. Текст» — структурная точка, а не конец предложения."""
        result = split_sentences("1. Первый пункт списка")

        assert len(result) == 1

    def test_empty_text(self) -> None:
        assert split_sentences("   ") == []
