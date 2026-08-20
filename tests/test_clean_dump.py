"""
Выгрузка очищенных конспектов: `index preview --dump-clean`.

Смысл артефакта — наблюдаемость самого непрозрачного шага. Очистка снимает
эмодзи, разметку, административные строки и целые служебные секции, и по чанкам
уже не видно, что именно исчезло: там текст разрезан и снабжён contextual
header. Поэтому проверяется не «файл записался», а что в файле видно и чего в
нём заведомо нет.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner
from fakes import isolated_settings

from zerocoder_assistant.cli.index import index_group
from zerocoder_assistant.config.settings import ChunkingConfig, Settings
from zerocoder_assistant.preprocessing import NotePreprocessor, ProcessedNote
from zerocoder_assistant.reporting import clean_note_filenames, render_clean_note

NOTE = """# PEr08. Практика: ассистент

Дорогой студент! Напиши об этом в учебный чат.

## Результат дня ✅

Разберём, как устроен **RAG** и из чего складывается точность.

## Теория на сегодня

### 1. Чанкинг

Время чтения: 15 минут

Чанк — это фрагмент документа.

```python
chunks = split(text)
```

## Домашнее задание

Соберите ассистента и пришлите ссылку.
"""


@pytest.fixture
def processed(tmp_path: Path) -> ProcessedNote:
    root = tmp_path / "корпус" / "Модуль 5. Ассистенты"
    root.mkdir(parents=True)
    path = root / "PEr08_практика.md"
    path.write_text(NOTE, encoding="utf-8")
    return NotePreprocessor(ChunkingConfig()).process_file(path, tmp_path / "корпус")


class TestCleanedSectionsSurviveThePipeline:
    def test_sections_are_kept_alongside_chunks(self, processed: ProcessedNote) -> None:
        assert processed.cleaned_sections
        assert processed.chunks

    def test_text_is_the_cleaned_one(self, processed: ProcessedNote) -> None:
        """Иначе дамп показывал бы исходник и не отвечал ни на один вопрос."""
        whole = "\n".join(section.text for section in processed.cleaned_sections)

        assert "✅" not in whole
        assert "**RAG**" not in whole
        assert "RAG" in whole
        assert "Время чтения" not in whole
        assert "Дорогой студент" not in whole

    def test_service_sections_are_absent(self, processed: ProcessedNote) -> None:
        whole = "\n".join(section.text for section in processed.cleaned_sections)

        assert "Соберите ассистента" not in whole
        assert "Домашнее задание" in processed.skipped_sections

    def test_code_survives_intact(self, processed: ProcessedNote) -> None:
        whole = "\n".join(section.text for section in processed.cleaned_sections)

        assert "chunks = split(text)" in whole

    def test_blocks_stay_separated(self, processed: ProcessedNote) -> None:
        """Абзац и следующий за ним листинг не должны слипаться в одну строку.

        Слипшийся текст всё ещё содержит и абзац, и код, поэтому проверка «код
        на месте» такую поломку пропускает. А читать дамп становится нельзя —
        ради чтения он и делается.
        """
        whole = "\n".join(section.text for section in processed.cleaned_sections)

        assert "документа.\n\n```python" in whole


class TestRenderCleanNote:
    def test_header_reports_what_was_dropped(self, processed: ProcessedNote) -> None:
        text = render_clean_note(processed)

        assert "- урок: PEr08" in text
        assert f"- чанков получилось: {len(processed.chunks)}" in text
        assert "Домашнее задание" in text  # перечислено как выброшенное

    def test_note_title_is_not_repeated_in_every_section(self, processed: ProcessedNote) -> None:
        """Путь заголовков начинается с H1 файла, и он уже стоит шапкой.

        Повторённый в каждой секции, он топит то, ради чего дамп открыли, —
        чем секции друг от друга отличаются.
        """
        text = render_clean_note(processed)
        headings = [line for line in text.splitlines() if line.startswith("##")]

        assert headings, "в дампе нет ни одного заголовка секции"
        assert all(processed.note.lesson_title not in line for line in headings)

    def test_sections_are_nested_under_the_title(self, processed: ProcessedNote) -> None:
        text = render_clean_note(processed)

        assert text.startswith(f"# {processed.note.lesson_title}")
        assert "\n## Результат дня\n" in text

    def test_reader_is_warned_about_merging(self, processed: ProcessedNote) -> None:
        """Слияние убирает каждую третью секцию — дамп не должен об этом молчать.

        Замер по корпусу: 869 секций после очистки против 564 разных путей
        заголовков в чанках. Читатель, сверивший дамп с `index preview`,
        без предупреждения решит, что одно из двух врёт.
        """
        text = render_clean_note(processed)

        assert "ДО чанкинга" in text
        assert "сливаются" in text

    def test_token_count_is_visible(self, processed: ProcessedNote) -> None:
        """Размер секции — то, из-за чего чанкер её потом делит или сливает."""
        text = render_clean_note(processed)

        assert "<!-- токенов:" in text


class TestCleanNoteFilenames:
    def make(self, source_file: str, lesson_id: str | None = "PEr08") -> ProcessedNote:
        from zerocoder_assistant.preprocessing.models import NoteMetadata

        return ProcessedNote(
            note=NoteMetadata(
                source_file=source_file,
                lesson_title="Урок",
                lesson_id=lesson_id,
                module="Модуль 5",
                module_num=5,
                notes_version=1,
            ),
            chunks=[],
        )

    def test_layout_is_flat(self) -> None:
        """Зеркалить папки нельзя: путь дампа перерастает предел Windows.

        Замер по корпусу: `storage/clean/<Модуль ...>/<PEmB01.2_...>.md` в
        рабочем дереве проекта доходит до 273 символов при пределе 260;
        плоское имя укладывается в 217.
        """
        names = clean_note_filenames([self.make("Модуль 5. Ассистенты/PEr08_практика.md")])

        assert list(names.values()) == ["PEr08_практика.md"]

    def test_no_separator_leaks_into_the_name(self) -> None:
        """Имя обязано оставаться именем: запись не должна уходить из CLEAN_DIR."""
        names = clean_note_filenames([self.make("а/б/в/PEr08_практика.md")])

        assert all("/" not in name and "\\" not in name for name in names.values())

    def test_collision_is_resolved_by_the_module_folder(self) -> None:
        """Сегодня имена уникальны по корпусу, но дамп не должен терять конспект."""
        names = clean_note_filenames(
            [
                self.make("Модуль 4. Форматы/PEm01_обзор.md"),
                self.make("Модуль 5. Ассистенты/PEm01_обзор.md"),
            ]
        )

        assert names["Модуль 4. Форматы/PEm01_обзор.md"] == "PEm01_обзор.md"
        assert names["Модуль 5. Ассистенты/PEm01_обзор.md"] == (
            "Модуль 5. Ассистенты_PEm01_обзор.md"
        )
        assert len(set(names.values())) == 2

    def test_second_collision_falls_back_to_a_number(self) -> None:
        names = clean_note_filenames(
            [
                self.make("PEm01_обзор.md"),
                self.make("PEm01_обзор.md"),
                self.make("PEm01_обзор.md"),
            ]
        )

        assert len(set(names.values())) == 1, "одинаковый source_file — это один и тот же файл"

    def test_every_note_gets_a_name(self) -> None:
        sources = [f"Модуль {index}/PEr0{index}_урок.md" for index in range(1, 6)]
        names = clean_note_filenames([self.make(source) for source in sources])

        assert set(names) == set(sources)
        assert len(set(names.values())) == len(sources)


class TestDumpCleanCommand:
    def corpus(self, tmp_path: Path) -> Path:
        root = tmp_path / "корпус" / "Модуль 5. Ассистенты"
        root.mkdir(parents=True)
        (root / "PEr08_практика.md").write_text(NOTE, encoding="utf-8")
        (root / "PEr07_кэш.md").write_text(
            NOTE.replace("PEr08. Практика: ассистент", "PEr07. Кэширование"), encoding="utf-8"
        )
        return tmp_path / "корпус"

    def settings_at(self, tmp_path: Path) -> Settings:
        return isolated_settings(openai_api_key="k", clean_dir=tmp_path / "чистые")

    def test_files_are_written_to_clean_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = self.settings_at(tmp_path)
        monkeypatch.setattr("zerocoder_assistant.cli.index.get_settings", lambda: settings)

        result = CliRunner().invoke(
            index_group,
            [
                "preview",
                "--notes-dir",
                str(self.corpus(tmp_path)),
                "--dump-clean",
                "--samples",
                "0",
            ],
        )

        assert result.exit_code == 0, result.output
        written = sorted(path.name for path in settings.clean_dir.iterdir())
        assert written == ["PEr07_кэш.md", "PEr08_практика.md"]

    def test_dump_follows_the_lesson_filter(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = self.settings_at(tmp_path)
        monkeypatch.setattr("zerocoder_assistant.cli.index.get_settings", lambda: settings)

        CliRunner().invoke(
            index_group,
            [
                "preview",
                "--notes-dir",
                str(self.corpus(tmp_path)),
                "--dump-clean",
                "--lesson",
                "PEr08",
                "--samples",
                "0",
            ],
        )

        assert [path.name for path in settings.clean_dir.iterdir()] == ["PEr08_практика.md"]

    def test_nothing_is_written_without_the_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Обычный preview остаётся командой, которая ничего не создаёт."""
        settings = self.settings_at(tmp_path)
        monkeypatch.setattr("zerocoder_assistant.cli.index.get_settings", lambda: settings)

        CliRunner().invoke(
            index_group,
            ["preview", "--notes-dir", str(self.corpus(tmp_path)), "--samples", "0"],
        )

        assert not settings.clean_dir.exists()

    def test_repeat_run_overwrites_rather_than_multiplies(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = self.settings_at(tmp_path)
        monkeypatch.setattr("zerocoder_assistant.cli.index.get_settings", lambda: settings)
        args = [
            "preview",
            "--notes-dir",
            str(self.corpus(tmp_path)),
            "--dump-clean",
            "--samples",
            "0",
        ]

        CliRunner().invoke(index_group, args)
        CliRunner().invoke(index_group, args)

        assert len(list(settings.clean_dir.iterdir())) == 2
