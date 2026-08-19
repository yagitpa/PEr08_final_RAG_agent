"""
Генерация: промпт, сборка контекста, проверка ссылок, кэш ответов.

Модель подменяется заглушкой — проверяется поведение конвейера, а не качество
конкретной модели. То, что проверить без живой модели нельзя (следует ли она
инструкциям промпта), выносится в golden set этапа 7.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fakes import FakeEmbedder, FakeLLM

from zerocoder_assistant.cache.sqlite_cache import SqliteCache
from zerocoder_assistant.config.settings import Settings
from zerocoder_assistant.generation import (
    NO_CONTEXT_ANSWER,
    Answerer,
    ContextBuilder,
    Prompt,
    PromptNotFoundError,
    cited_numbers,
    render_user_message,
    unknown_citations,
)
from zerocoder_assistant.generation.context_builder import FRAGMENTS_HEADER, QUESTION_HEADER
from zerocoder_assistant.memory import SessionHistory
from zerocoder_assistant.preprocessing.tokenization import get_token_counter
from zerocoder_assistant.retrieval import Retriever, build_where
from zerocoder_assistant.retrieval.retriever import RetrievedChunk
from zerocoder_assistant.vectorstore.chroma_store import ChromaVectorStore

PROMPT_TEXT = "Отвечай только по фрагментам."


def make_hit(number: int, text: str, similarity: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"PEr08:s000:c{number:02d}",
        text=text,
        similarity=similarity,
        metadata={"lesson_id": "PEr08", "section_title": f"Секция {number}"},
    )


def make_long_chunk(number: int):
    """Чанк, который в одиночку занимает заметную часть бюджета контекста."""
    from fakes import make_chunk

    return make_chunk(number, f"фрагмент {number}. " + "длинное слово " * 60)


@pytest.fixture
def prompt(settings: Settings) -> Prompt:
    """Промпт на диске: путь тот же, что используется приложением."""
    settings.answer_prompt_path.parent.mkdir(parents=True, exist_ok=True)
    settings.answer_prompt_path.write_text(PROMPT_TEXT, encoding="utf-8")
    return Prompt.load(settings.answer_prompt_path)


class TestPrompt:
    def test_version_follows_content(self, tmp_path: Path) -> None:
        """Версия — хеш содержимого: правку промпта нельзя забыть отметить."""
        path = tmp_path / "prompt.md"
        path.write_text("первая редакция", encoding="utf-8")
        first = Prompt.load(path)

        path.write_text("вторая редакция", encoding="utf-8")
        second = Prompt.load(path)

        assert first.version != second.version

    def test_missing_file_is_predictable_failure(self, tmp_path: Path) -> None:
        with pytest.raises(PromptNotFoundError):
            Prompt.load(tmp_path / "нет.md")

    def test_empty_file_is_predictable_failure(self, tmp_path: Path) -> None:
        """Пустой промпт молча сменил бы все правила ответа на умолчания модели."""
        path = tmp_path / "prompt.md"
        path.write_text("   \n", encoding="utf-8")

        with pytest.raises(PromptNotFoundError):
            Prompt.load(path)


class TestContextBuilder:
    def test_numbering_starts_at_one(self) -> None:
        context = ContextBuilder(1000).build([make_hit(0, "первый"), make_hit(1, "второй")])

        assert context.text.startswith("[1] первый")
        assert "[2] второй" in context.text
        assert context.used == 2

    def test_budget_drops_weakest_whole_fragments(self) -> None:
        """Режем целыми фрагментами: обрезанный чанк теряет конец мысли."""
        hits = [make_hit(number, "слово " * 100) for number in range(5)]

        context = ContextBuilder(300).build(hits)

        assert context.used < len(hits)
        assert context.dropped == len(hits) - context.used
        assert context.tokens <= 300
        assert not context.truncated

    def test_single_oversized_fragment_is_trimmed_not_dropped(self) -> None:
        """Бюджет меньше одного чанка — отдать пустоту хуже, чем отдать урезанный."""
        context = ContextBuilder(200).build([make_hit(0, "слово " * 500)])

        assert context.used == 1
        assert context.truncated
        assert context.tokens <= 200

    def test_empty_input(self) -> None:
        context = ContextBuilder(1000).build([])

        assert context.is_empty
        assert context.text == ""

    def test_question_goes_after_fragments(self) -> None:
        message = render_user_message("[1] текст", "как это работает")

        assert message.index(FRAGMENTS_HEADER) < message.index(QUESTION_HEADER)
        assert message.rstrip().endswith("как это работает")


class TestCitations:
    def test_numbers_in_order_of_first_mention(self) -> None:
        assert cited_numbers("Сначала [2], потом [1], снова [2].") == [2, 1]

    def test_no_citations(self) -> None:
        assert cited_numbers("Ответ без ссылок.") == []

    def test_invented_citation_is_detected(self) -> None:
        """Сослаться на [7] при пяти фрагментах можно только придумав его."""
        assert unknown_citations("По [1] и [7].", available=5) == [7]

    def test_all_citations_valid(self) -> None:
        assert unknown_citations("По [1] и [3].", available=3) == []


def build_answerer(
    settings: Settings,
    store: ChromaVectorStore,
    prompt: Prompt,
    *,
    llm: FakeLLM,
    cache: SqliteCache | None = None,
    embedder: FakeEmbedder | None = None,
) -> Answerer:
    retriever = Retriever(settings, embedder=embedder or FakeEmbedder(), store=store, cache=cache)
    return Answerer(settings, retriever=retriever, llm=llm, cache=cache, prompt=prompt)


class TestAnswerer:
    def test_answers_from_found_fragments(
        self, settings: Settings, store: ChromaVectorStore, prompt: Prompt
    ) -> None:
        llm = FakeLLM("Кэш работает так [1].")
        answerer = build_answerer(settings, store, prompt, llm=llm)

        answer = answerer.answer("как работает кэш", use_cache=False)

        assert answer.text == "Кэш работает так [1]."
        assert answer.grounded
        assert answer.used_fragments == settings.top_k
        assert answer.sources

    def test_empty_retrieval_skips_the_model(
        self, settings: Settings, store: ChromaVectorStore, prompt: Prompt
    ) -> None:
        """Порог имеет смысл ровно потому, что после него можно не звать модель."""
        llm = FakeLLM()
        embedder = FakeEmbedder({"столица Австралии": [0.0, -1.0]})
        answerer = build_answerer(settings, store, prompt, llm=llm, embedder=embedder)

        answer = answerer.answer("столица Австралии", use_cache=False)

        assert answer.text == NO_CONTEXT_ANSWER
        assert not answer.grounded
        assert llm.calls == []

    def test_fragments_travel_as_data_not_as_instruction(
        self, settings: Settings, store: ChromaVectorStore, prompt: Prompt
    ) -> None:
        """Конспекты по промпт-инжинирингу полны чужих инструкций.

        Системное сообщение — это то, что модель исполняет; фрагментам там не
        место, даже когда так короче.
        """
        llm = FakeLLM()
        answerer = build_answerer(settings, store, prompt, llm=llm)

        answerer.answer("как работает кэш", use_cache=False)

        system = [message for message in llm.last_messages if message["role"] == "system"]
        assert len(system) == 1
        assert system[0]["content"] == PROMPT_TEXT
        assert "фрагмент про кэширование" not in system[0]["content"]
        assert "фрагмент про кэширование" in llm.last_messages[-1]["content"]

    def test_invented_citation_is_reported_not_silently_fixed(
        self, settings: Settings, store: ChromaVectorStore, prompt: Prompt
    ) -> None:
        llm = FakeLLM("По фрагменту [9] всё понятно.")
        answerer = build_answerer(settings, store, prompt, llm=llm)

        answer = answerer.answer("как работает кэш", use_cache=False)

        assert answer.unknown_citations == [9]
        assert answer.text == "По фрагменту [9] всё понятно."

    def test_filters_reach_the_store(
        self, settings: Settings, store: ChromaVectorStore, prompt: Prompt
    ) -> None:
        llm = FakeLLM()
        answerer = build_answerer(settings, store, prompt, llm=llm)

        answer = answerer.answer(
            "как работает кэш", where=build_where(lessons=["PEr01"]), use_cache=False
        )

        assert answer.text == NO_CONTEXT_ANSWER  # PEr01 лежит ниже порога
        assert llm.calls == []


class TestAnswerCache:
    def test_repeat_question_skips_the_model(
        self, settings: Settings, store: ChromaVectorStore, prompt: Prompt
    ) -> None:
        cache = SqliteCache(settings.cache_db)
        llm = FakeLLM("Ответ [1].")
        answerer = build_answerer(settings, store, prompt, llm=llm, cache=cache)

        first = answerer.answer("как работает кэш")
        second = answerer.answer("как работает кэш")

        assert not first.from_cache
        assert second.from_cache
        assert second.text == first.text
        assert second.sources == first.sources
        assert len(llm.calls) == 1

    def test_history_disables_the_answer_cache(
        self, settings: Settings, store: ChromaVectorStore, prompt: Prompt
    ) -> None:
        """Ответ на «а подробнее?» зависит от прошлой реплики, а ключ о ней не знает."""
        cache = SqliteCache(settings.cache_db)
        llm = FakeLLM("Ответ [1].")
        answerer = build_answerer(settings, store, prompt, llm=llm, cache=cache)
        history = SessionHistory(5)
        history.add("что такое кэш", "хранилище готовых результатов")

        answerer.answer("как работает кэш", history=history)
        answerer.answer("как работает кэш", history=history)

        assert len(llm.calls) == 2
        assert cache.stats().answers == 0

    def test_empty_history_still_caches(
        self, settings: Settings, store: ChromaVectorStore, prompt: Prompt
    ) -> None:
        cache = SqliteCache(settings.cache_db)
        answerer = build_answerer(settings, store, prompt, llm=FakeLLM(), cache=cache)

        answerer.answer("как работает кэш", history=SessionHistory(5))

        assert cache.stats().answers == 1

    def test_different_top_k_misses_the_cache(
        self, settings: Settings, store: ChromaVectorStore, prompt: Prompt
    ) -> None:
        cache = SqliteCache(settings.cache_db)
        llm = FakeLLM("Ответ [1].")
        answerer = build_answerer(settings, store, prompt, llm=llm, cache=cache)

        answerer.answer("как работает кэш")
        answerer.answer("как работает кэш", top_k=1)

        assert len(llm.calls) == 2

    def test_edited_prompt_misses_the_cache(
        self, settings: Settings, store: ChromaVectorStore, prompt: Prompt
    ) -> None:
        """Версия промпта — в ключе, иначе правка правил ответа ничего не меняет."""
        cache = SqliteCache(settings.cache_db)
        llm = FakeLLM("Ответ [1].")
        answerer = build_answerer(settings, store, prompt, llm=llm, cache=cache)
        answerer.answer("как работает кэш")

        edited = Prompt(name=prompt.name, text="Другие правила.", version="deadbeef")
        answerer = build_answerer(settings, store, edited, llm=llm, cache=cache)
        answerer.answer("как работает кэш")

        assert len(llm.calls) == 2


class TestCitationsInCode:
    """Конспекты по промпт-инжинирингу полны примеров на Python.

    Индексация массива — не ссылка на фрагмент. Без этого различия индикатор
    галлюцинаций мерил бы долю ответов, в которых есть код.
    """

    @pytest.mark.parametrize(
        "answer",
        [
            "Смотрите: `items[0]` возвращает первый элемент.",
            "Пишите data[10] и data[42], как в конспекте.",
            "Обращение вида results[3] к списку.",
            'Ключ достаётся так: config["hosts"][0].',
        ],
    )
    def test_array_indexing_is_not_a_citation(self, answer: str) -> None:
        assert unknown_citations(answer, available=2) == []

    def test_real_citation_next_to_code_still_counts(self) -> None:
        assert cited_numbers("Пример `items[0]` разобран во фрагменте [2].") == [2]

    def test_long_number_is_not_missed(self) -> None:
        """Ограничение в три цифры пропускало бы [1234] как несуществующий номер."""
        assert unknown_citations("По фрагменту [1234].", available=5) == [1234]


class TestContextBudgetEdges:
    def test_fit_terminates_on_impossible_budget(self) -> None:
        """Зависание — худшая реакция на плохой аргумент: класс публичный."""
        context = ContextBuilder(1).build([make_hit(0, "слово " * 50)])

        assert context.truncated
        assert context.used == 1

    def test_reported_tokens_match_the_text(self) -> None:
        """Разделителей на n блоков ровно n-1, иначе --verbose печатает завышенное."""
        counter = get_token_counter("o200k_base")
        hits = [make_hit(number, f"фрагмент номер {number}") for number in range(3)]

        context = ContextBuilder(1000, counter=counter).build(hits)

        assert context.tokens == counter.count(context.text)


class TestAnswerCacheFingerprint:
    """Всё, что влияет на текст ответа, обязано входить в ключ.

    Иначе правка настройки отдаёт вчерашний ответ как свежий — ровно тот
    сценарий, который домашние задания модуля просят продемонстрировать.
    """

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("temperature", 0.9),
            ("max_answer_tokens", 500),
            ("max_context_tokens", 900),
            # Кодировка решает, сколько фрагментов влезет в бюджет: один и тот
            # же чанк весит по-разному в разных токенайзерах.
            ("tokenizer_encoding", "cl100k_base"),
        ],
    )
    def test_changed_setting_misses_the_cache(
        self,
        settings: Settings,
        store: ChromaVectorStore,
        prompt: Prompt,
        field: str,
        value: object,
    ) -> None:
        cache = SqliteCache(settings.cache_db)
        llm = FakeLLM("Ответ [1].")

        build_answerer(settings, store, prompt, llm=llm, cache=cache).answer("как работает кэш")
        changed = settings.model_copy(update={field: value})
        build_answerer(changed, store, prompt, llm=llm, cache=cache).answer("как работает кэш")

        assert len(llm.calls) == 2, f"{field} не входит в ключ кэша ответов"

    def test_cache_hit_keeps_the_citation_warning(
        self, settings: Settings, store: ChromaVectorStore, prompt: Prompt
    ) -> None:
        """Текст тот же — значит и предупреждение о выдумке должно быть тем же."""
        cache = SqliteCache(settings.cache_db)
        answerer = build_answerer(
            settings, store, prompt, llm=FakeLLM("По фрагменту [9]."), cache=cache
        )

        first = answerer.answer("как работает кэш")
        second = answerer.answer("как работает кэш")

        assert second.from_cache
        assert second.unknown_citations == first.unknown_citations == [9]


class TestSourcesMatchContext:
    def test_sources_cover_only_what_reached_the_model(
        self, settings: Settings, tmp_path: Path, prompt: Prompt
    ) -> None:
        """Ссылка [3] обязана быть третьей строкой списка, а не третьим найденным."""
        narrow = settings.model_copy(update={"top_k": 3, "max_context_tokens": 200})
        llm = FakeLLM("Ответ [1].")

        with ChromaVectorStore(tmp_path / "wide", "fake-embed-v1") as store:
            store.upsert(
                [make_long_chunk(number) for number in range(3)],
                [[1.0, 0.0], [0.99, 0.01], [0.98, 0.02]],
            )
            answer = build_answerer(narrow, store, prompt, llm=llm).answer(
                "длинный вопрос", use_cache=False
            )

        assert answer.dropped_fragments > 0
        assert len(answer.sources) == answer.used_fragments
