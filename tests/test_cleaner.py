"""Очистка текста и классификация содержимого."""

from __future__ import annotations

import pytest

from zerocoder_assistant.config.constants import (
    CONTENT_TYPE_ACTUALIZATION,
    CONTENT_TYPE_CODE,
    CONTENT_TYPE_PRACTICE,
    CONTENT_TYPE_SUMMARY,
    CONTENT_TYPE_THEORY,
)
from zerocoder_assistant.preprocessing.cleaner import TextCleaner
from zerocoder_assistant.preprocessing.headings import (
    classify_content_type,
    is_excluded_section,
    normalize_heading,
    sanitize_heading,
)
from zerocoder_assistant.preprocessing.models import Block


@pytest.fixture
def cleaner() -> TextCleaner:
    return TextCleaner()


class TestCleanText:
    def test_emoji_removed_together_with_emphasis(self, cleaner: TextCleaner) -> None:
        """Порядок операций: эмодзи после выделения, иначе остаются звёздочки."""
        assert cleaner.clean_text("**Финальная сборка ↘️**") == "Финальная сборка"

    def test_emphasis_stripped_but_content_kept(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean_text("**жирный с `кодом` внутри**") == "жирный с кодом внутри"

    @pytest.mark.parametrize("dash", ["—", "–", "−", "‑"])
    def test_dashes_normalized_not_deleted(self, cleaner: TextCleaner, dash: str) -> None:
        """Удаление тире сломало бы «что-то» и диапазоны — только приведение."""
        assert cleaner.clean_text(f"RAG {dash} это паттерн") == "RAG - это паттерн"

    def test_hyphenated_words_survive(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean_text("что-то из-за RAG-системы") == "что-то из-за RAG-системы"

    def test_multiplication_is_not_emphasis(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean_text("Берём top_k * 3 кандидатов") == "Берём top_k * 3 кандидатов"

    def test_links_reduced_to_caption(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean_text("см. [проект](https://example.com/x)") == "см. проект"

    def test_html_tags_removed(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean_text("<b>Жирный</b> текст") == "Жирный текст"

    def test_blockquote_marker_removed(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean_text("> ⚠️ Предупреждение") == "Предупреждение"

    def test_table_divider_removed_header_kept(self, cleaner: TextCleaner) -> None:
        result = cleaner.clean_text("| Параметр | Описание |\n| --- | --- |\n| top_k | Сколько |")

        assert "---" not in result
        assert "Параметр" in result
        assert "top_k" in result

    def test_whitespace_collapsed(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean_text("много     пробелов") == "много пробелов"

    def test_case_preserved(self, cleaner: TextCleaner) -> None:
        """Нижний регистр ломает имена моделей и термины."""
        assert "ChromaDB" in cleaner.clean_text("Используем ChromaDB")


class TestCleanCode:
    def test_code_content_untouched(self, cleaner: TextCleaner) -> None:
        source = '```python\nx = "**не выделение**"\n# 💡 комментарий\n```'
        result = cleaner.clean_code(source)

        assert "**не выделение**" in result
        assert "```python" in result

    def test_clean_block_dispatches_by_type(self, cleaner: TextCleaner) -> None:
        code = cleaner.clean_block(Block("```\n**x**\n```", is_code=True))
        text = cleaner.clean_block(Block("**x**", is_code=False))

        assert "**x**" in code.text
        assert text.text == "x"


class TestBoilerplate:
    def test_disclaimer_detected(self, cleaner: TextCleaner) -> None:
        assert cleaner.is_boilerplate("Дорогой студент! Интерфейсы могут меняться.")

    def test_regular_text_is_not_boilerplate(self, cleaner: TextCleaner) -> None:
        assert not cleaner.is_boilerplate("RAG - это архитектурный паттерн.")

    @pytest.mark.parametrize(
        "block",
        [
            "**Модуль 5. Ассистенты с памятью: базы знаний, RAG и Relevance**",
            "Модуль 3. No-code и API-интеграции",
        ],
    )
    def test_module_title_block_dropped(self, cleaner: TextCleaner, block: str) -> None:
        """Название модуля дублирует метаданные и забивает top_k."""
        assert cleaner.is_boilerplate(block)

    @pytest.mark.parametrize(
        "block",
        [
            "**Модуль 1: rag.py (Реализация RAG).** Это ядро нашей системы. "
            "Класс RAGAssistant объединяет поиск и генерацию.",
            "**Модуль 2: embeddings.py (Работа с хранилищем).** Этот модуль отвечает "
            "за создание эмбеддингов.",
        ],
    )
    def test_module_prefixed_prose_is_kept(self, cleaner: TextCleaner, block: str) -> None:
        """Разбор кода проекта начинается так же, но это содержимое, а не шапка."""
        assert not cleaner.is_boilerplate(block)


class TestAdminLines:
    def test_admin_header_removed_entirely(self, cleaner: TextCleaner) -> None:
        header = (
            "Курс: Промпт-инжиниринг 3.0 - Модуль 3\n"
            "Статус: Обязательно для заполнения\n"
            "Время чтения и просмотра: ~ 1 час\n"
            "Время выполнения задания: ~ 1,5 часа"
        )

        assert cleaner.drop_admin_lines(header) == ""

    def test_only_admin_lines_removed(self, cleaner: TextCleaner) -> None:
        text = "Время чтения и просмотра: ~ 1 час\nRAG - это архитектурный паттерн."

        assert cleaner.drop_admin_lines(text) == "RAG - это архитектурный паттерн."

    def test_teacher_line_is_kept(self, cleaner: TextCleaner) -> None:
        """«Кто вёл урок» — осмысленный вопрос, эта строка не административная."""
        line = "Преподаватель: Даниил Соболев, Fullstack разработчик"

        assert cleaner.drop_admin_lines(line) == line

    def test_content_untouched(self, cleaner: TextCleaner) -> None:
        text = "Стоимость зависит от качества: ~$0.167 за изображение."

        assert cleaner.drop_admin_lines(text) == text


class TestHeadings:
    def test_sanitize_strips_emoji_keeps_case(self) -> None:
        assert sanitize_heading("Подготовка: права доступа ↘️") == "Подготовка: права доступа"

    def test_normalize_strips_numbering_and_punctuation(self) -> None:
        """На вход приходит текст заголовка без решёток — их снимает парсер."""
        assert normalize_heading("1. Оценка качества: критерии") == "оценка качества критерии"

    @pytest.mark.parametrize(
        "heading",
        ["1. Введение", "10.2 Подраздел", "Этап 3. Проверка", "Шаг 1) Настройка"],
    )
    def test_numbering_styles_stripped(self, heading: str) -> None:
        """В корпусе разделы нумеруются по-разному — канон должен быть один."""
        assert not normalize_heading(heading)[0].isdigit()

    def test_normalize_folds_yo(self) -> None:
        assert normalize_heading("Подведём итоги") == normalize_heading("Подведем итоги")

    @pytest.mark.parametrize(
        "heading",
        ["Домашнее задание", "На следующем занятии", "Что мы умеем", "Полезные ссылки"],
    )
    def test_service_sections_excluded(self, heading: str) -> None:
        assert is_excluded_section(heading)

    @pytest.mark.parametrize("heading", ["Результат дня", "Итоги урока", "1. Что такое RAG"])
    def test_content_sections_kept(self, heading: str) -> None:
        assert not is_excluded_section(heading)

    def test_unknown_heading_is_kept(self) -> None:
        """Неизвестный заголовок сохраняется: оформление в корпусе неоднородно."""
        assert not is_excluded_section("Совершенно новый раздел")


class TestClassifyContentType:
    def test_actualization_wins_over_code(self) -> None:
        """Блок «Актуализация» ценен как свежая фактура, даже если содержит листинг."""
        result = classify_content_type("1. Введение", "Актуализация: версии", code_ratio=0.9)

        assert result == CONTENT_TYPE_ACTUALIZATION

    def test_code_dominant_chunk(self) -> None:
        assert classify_content_type("Пример", "x = 1", code_ratio=0.8) == CONTENT_TYPE_CODE

    def test_summary_heading(self) -> None:
        assert classify_content_type("Результат дня", "текст") == CONTENT_TYPE_SUMMARY

    def test_practice_heading(self) -> None:
        assert classify_content_type("5. Практика: запуск", "текст") == CONTENT_TYPE_PRACTICE

    def test_default_is_theory(self) -> None:
        assert classify_content_type("2. Про векторы", "текст") == CONTENT_TYPE_THEORY
