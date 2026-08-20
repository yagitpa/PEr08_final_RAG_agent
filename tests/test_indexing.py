"""
Сборка индекса: идемпотентность, частичная сборка, защита манифеста.

Сеть не используется: эмбеддер подменяется заглушкой, которая заодно считает,
сколько текстов её просили векторизовать. Именно это число и проверяется —
оно равно объёму платной работы.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from zerocoder_assistant.config.settings import Settings
from zerocoder_assistant.errors import IndexBuildError, IndexMismatchError
from zerocoder_assistant.indexing import IndexBuilder
from zerocoder_assistant.vectorstore.chroma_store import ChromaVectorStore
from zerocoder_assistant.vectorstore.manifest import IndexManifest

NOTE_TEMPLATE = """# {lesson_id}. {title}

## Теория на сегодня

### 1. Первый раздел

{body}

### 2. Второй раздел

Векторный поиск находит фрагменты по смысловой близости, а не по совпадению слов.
Это позволяет отвечать на вопрос, сформулированный другими словами.

## Домашнее задание

Это служебная секция, она не должна попасть в индекс.
"""


class FakeEmbedder:
    """Эмбеддер без сети. Считает, сколько текстов его просили векторизовать."""

    def __init__(self, model: str = "fake-embed-v1", dimension: int = 3) -> None:
        self._model = model
        self._dimension = dimension
        self.embedded: list[str] = []

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return [[float(len(text) % 97), 1.0, 0.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


def write_note(notes_dir: Path, lesson_id: str, body: str = "Текст первого раздела.") -> Path:
    path = notes_dir / f"{lesson_id}_Урок.md"
    path.write_text(
        NOTE_TEMPLATE.format(lesson_id=lesson_id, title="Урок", body=body),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def notes_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "notes"
    directory.mkdir()
    return directory


@pytest.fixture
def settings(tmp_path: Path, notes_dir: Path) -> Settings:
    return Settings(
        openai_api_key="test-key",
        notes_dir=notes_dir,
        chroma_dir=tmp_path / "chroma",
    )


def build(settings: Settings, embedder: FakeEmbedder, **kwargs: object) -> object:
    return IndexBuilder(settings, embedder=embedder).build(**kwargs)  # type: ignore[arg-type]


class TestFreshBuild:
    def test_all_chunks_added(self, settings: Settings, notes_dir: Path) -> None:
        write_note(notes_dir, "PEr01")
        embedder = FakeEmbedder()
        report = build(settings, embedder)

        assert report.notes == 1
        assert report.chunks > 0
        assert report.added == report.chunks
        assert report.embedded == report.chunks
        assert len(embedder.embedded) == report.chunks

    def test_service_sections_not_indexed(self, settings: Settings, notes_dir: Path) -> None:
        write_note(notes_dir, "PEr01")
        embedder = FakeEmbedder()
        build(settings, embedder)

        assert not any("служебная секция" in text for text in embedder.embedded)

    def test_manifest_written(self, settings: Settings, notes_dir: Path) -> None:
        write_note(notes_dir, "PEr01")
        report = build(settings, FakeEmbedder())
        manifest = IndexManifest.load(settings.chroma_dir)

        assert manifest is not None
        assert manifest.embed_model == "fake-embed-v1"
        assert manifest.chunks == report.chunks


class TestIdempotency:
    def test_second_build_embeds_nothing(self, settings: Settings, notes_dir: Path) -> None:
        """Повторный запуск без правок не должен стоить ни одного запроса."""
        write_note(notes_dir, "PEr01")
        build(settings, FakeEmbedder())

        second = FakeEmbedder()
        report = build(settings, second)

        assert report.embedded == 0
        assert second.embedded == []
        assert report.unchanged == report.chunks

    def test_only_changed_note_reembedded(self, settings: Settings, notes_dir: Path) -> None:
        write_note(notes_dir, "PEr01")
        write_note(notes_dir, "PEr02")
        first = FakeEmbedder()
        build(settings, first)

        write_note(notes_dir, "PEr02", body="Совершенно другой текст первого раздела.")
        second = FakeEmbedder()
        report = build(settings, second)

        assert report.updated > 0
        assert report.unchanged > 0
        assert len(second.embedded) == report.updated
        assert all("Совершенно другой" in text or "PEr02" in text for text in second.embedded)

    def test_removed_note_purges_its_chunks(self, settings: Settings, notes_dir: Path) -> None:
        write_note(notes_dir, "PEr01")
        path = write_note(notes_dir, "PEr02")
        first = build(settings, FakeEmbedder())

        path.unlink()
        report = build(settings, FakeEmbedder())

        assert report.removed > 0
        assert report.chunks < first.chunks

        with ChromaVectorStore(settings.chroma_dir, "fake-embed-v1") as store:
            assert store.count() == report.chunks


class TestPartialBuild:
    def test_lesson_filter_limits_work(self, settings: Settings, notes_dir: Path) -> None:
        write_note(notes_dir, "PEr01")
        write_note(notes_dir, "PEr02")
        embedder = FakeEmbedder()
        report = build(settings, embedder, lessons=["PEr02"])

        assert report.notes == 1
        assert all("PEr02" in text for text in embedder.embedded)

    def test_partial_build_never_deletes(self, settings: Settings, notes_dir: Path) -> None:
        """Иначе сборка одного урока снесла бы из индекса все остальные."""
        write_note(notes_dir, "PEr01")
        write_note(notes_dir, "PEr02")
        full = build(settings, FakeEmbedder())

        report = build(settings, FakeEmbedder(), lessons=["PEr02"])

        assert report.removed == 0
        with ChromaVectorStore(settings.chroma_dir, "fake-embed-v1") as store:
            assert store.count() == full.chunks

    def test_lesson_filter_is_case_insensitive(self, settings: Settings, notes_dir: Path) -> None:
        write_note(notes_dir, "PEr01")
        report = build(settings, FakeEmbedder(), lessons=["per01"])

        assert report.notes == 1


class TestDryRun:
    def test_nothing_written_and_nothing_embedded(
        self, settings: Settings, notes_dir: Path
    ) -> None:
        write_note(notes_dir, "PEr01")
        embedder = FakeEmbedder()
        report = build(settings, embedder, dry_run=True)

        assert report.dry_run
        assert report.added > 0  # план показан
        assert embedder.embedded == []  # но не оплачен
        assert IndexManifest.load(settings.chroma_dir) is None

    def test_dry_run_after_build_shows_no_changes(
        self, settings: Settings, notes_dir: Path
    ) -> None:
        write_note(notes_dir, "PEr01")
        build(settings, FakeEmbedder())
        report = build(settings, FakeEmbedder(), dry_run=True)

        assert report.added == 0
        assert report.unchanged == report.chunks


class TestManifestProtection:
    def test_model_change_refused(self, settings: Settings, notes_dir: Path) -> None:
        write_note(notes_dir, "PEr01")
        build(settings, FakeEmbedder(model="fake-embed-v1"))

        with pytest.raises(IndexMismatchError, match="fake-embed-v2"):
            build(settings, FakeEmbedder(model="fake-embed-v2"))

    def test_refusal_leaves_no_orphan_collection(self, settings: Settings, notes_dir: Path) -> None:
        """Сверка идёт до открытия хранилища, иначе остаётся пустая коллекция."""
        write_note(notes_dir, "PEr01")
        build(settings, FakeEmbedder(model="fake-embed-v1"))

        with pytest.raises(IndexMismatchError):
            build(settings, FakeEmbedder(model="fake-embed-v2"))

        with ChromaVectorStore(settings.chroma_dir, "fake-embed-v2") as store:
            assert store.count() == 0
        assert IndexManifest.load(settings.chroma_dir).embed_model == "fake-embed-v1"

    def test_rebuild_overrides_mismatch(self, settings: Settings, notes_dir: Path) -> None:
        write_note(notes_dir, "PEr01")
        build(settings, FakeEmbedder(model="fake-embed-v1"))

        report = build(settings, FakeEmbedder(model="fake-embed-v2"), rebuild=True)

        assert report.embedded == report.chunks
        assert IndexManifest.load(settings.chroma_dir).embed_model == "fake-embed-v2"

    def test_rebuild_reembeds_everything(self, settings: Settings, notes_dir: Path) -> None:
        write_note(notes_dir, "PEr01")
        build(settings, FakeEmbedder())

        embedder = FakeEmbedder()
        report = build(settings, embedder, rebuild=True)

        assert report.unchanged == 0
        assert len(embedder.embedded) == report.chunks


class TestResilience:
    def test_broken_file_does_not_stop_build(self, settings: Settings, notes_dir: Path) -> None:
        write_note(notes_dir, "PEr01")
        # Последовательность, недопустимая в UTF-8: чтение файла упадёт.
        (notes_dir / "PEr02_битый.md").write_bytes(b"\xff\xfe\x80\x81\x82")

        report = build(settings, FakeEmbedder())

        assert report.notes == 1
        assert len(report.failed_files) == 1
        assert report.chunks > 0

    def test_empty_corpus_is_not_a_crash(self, settings: Settings) -> None:
        report = build(settings, FakeEmbedder())

        assert report.notes == 0
        assert report.chunks == 0
        assert report.embedded == 0


class TestDestructiveGuards:
    """Операции, которые уносят содержимое базы, а выглядят успешными.

    Обе проверки написаны после код-ревью: в обоих случаях чанки исчезали из
    индекса, отчёт показывал `removed=0`, а файлы на диске оставались целыми.
    """

    def test_rebuild_with_lesson_filter_is_refused(
        self, settings: Settings, notes_dir: Path
    ) -> None:
        """Поодиночке оба флага безопасны, вместе — стирают чужие уроки.

        `--rebuild` чистит коллекцию целиком, `--lesson` наполняет её только
        отобранным. Отчёт при этом честно сообщает `removed=0`: плановых
        удалений не было, содержимое исчезло раньше.
        """
        write_note(notes_dir, "PEr01")
        write_note(notes_dir, "PEr02")
        build(settings, FakeEmbedder())

        with pytest.raises(IndexBuildError, match="несовместима"):
            build(settings, FakeEmbedder(), rebuild=True, lessons=["PEr01"])

    def test_other_lessons_survive_the_refused_rebuild(
        self, settings: Settings, notes_dir: Path
    ) -> None:
        """Отказ должен случиться ДО очистки, а не после."""
        write_note(notes_dir, "PEr01")
        write_note(notes_dir, "PEr02")
        before = build(settings, FakeEmbedder()).chunks

        with pytest.raises(IndexBuildError):
            build(settings, FakeEmbedder(), rebuild=True, lessons=["PEr01"])

        assert build(settings, FakeEmbedder(), dry_run=True).unchanged == before

    def test_unreadable_file_does_not_delete_its_chunks(
        self, settings: Settings, notes_dir: Path
    ) -> None:
        """Сбой чтения и «конспект удалён» — разные вещи.

        Файл цел на диске, но не читается (битая кодировка, несинхронизированный
        плейсхолдер OneDrive, блокировка антивирусом). Прежде все его чанки
        уходили из базы, и ассистент начинал отвечать «в конспектах этого нет».
        """
        write_note(notes_dir, "PEr01")
        broken = write_note(notes_dir, "PEr02")
        first = build(settings, FakeEmbedder())
        assert first.notes == 2

        broken.write_bytes(b"\xff\xfe\x80\x81\x82")
        second = build(settings, FakeEmbedder())

        assert second.failed_files
        assert second.removed == 0

    def test_deleted_file_still_removes_its_chunks(
        self, settings: Settings, notes_dir: Path
    ) -> None:
        """Осторожность не должна отменить настоящее удаление."""
        write_note(notes_dir, "PEr01")
        removable = write_note(notes_dir, "PEr02")
        build(settings, FakeEmbedder())

        removable.unlink()
        report = build(settings, FakeEmbedder())

        assert report.removed > 0


class TestMetadataReachesTheIndex:
    def test_frontmatter_change_is_reindexed(self, settings: Settings, notes_dir: Path) -> None:
        """Правка паспорта без правки текста обязана доехать до индекса.

        Пока отпечаток считался только по тексту, такая правка давала
        `unchanged`, `embedded=0` — и фильтры `--lesson` / `--module` продолжали
        искать по старым значениям, ничем себя не выдавая.
        """
        path = write_note(notes_dir, "PEr01")
        build(settings, FakeEmbedder())

        original = path.read_text(encoding="utf-8")
        path.write_text(
            "---\nlesson_id: PEr99\nmodule_num: 9\n---\n\n" + original, encoding="utf-8"
        )
        report = build(settings, FakeEmbedder())

        assert report.updated > 0
        assert report.embedded > 0

    def test_identical_note_is_still_free(self, settings: Settings, notes_dir: Path) -> None:
        """Отпечаток стал шире — но повтор без правок обязан остаться бесплатным."""
        write_note(notes_dir, "PEr01")
        build(settings, FakeEmbedder())

        embedder = FakeEmbedder()
        report = build(settings, embedder)

        assert report.embedded == 0
        assert embedder.embedded == []
