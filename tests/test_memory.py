"""
Память диалога: окно истории, распознавание уточнений, `/clear`.

Проверяется не «история хранится», а то, ради чего она нужна: уточняющий вопрос
должен попасть в поиск с предметными словами из предыдущей реплики, иначе искать
в базе будет нечего.
"""

from __future__ import annotations

import pytest

from zerocoder_assistant.memory import SessionHistory, is_follow_up


class TestFollowUpDetection:
    @pytest.mark.parametrize(
        "question",
        [
            "а сколько его ставить?",
            "почему так?",
            "и что дальше",
            "а это обязательно",
            "зачем он нужен",
            "расскажи про них подробнее",
        ],
    )
    def test_referential_questions(self, question: str) -> None:
        assert is_follow_up(question)

    @pytest.mark.parametrize(
        "question",
        [
            "что такое overlap",
            "как собрать индекс",
            "какие метрики считает RAGAS",
            "чем chunk отличается от секции",
        ],
    )
    def test_self_contained_questions(self, question: str) -> None:
        assert not is_follow_up(question)

    @pytest.mark.parametrize(
        "question",
        [
            "что такое overlap и зачем он нужен",
            "а как задать порог релевантности в конфиге проекта",
        ],
    )
    def test_long_question_stands_on_its_own(self, question: str) -> None:
        """Отсылка есть, но предметных слов хватает и без предыдущей реплики."""
        assert not is_follow_up(question)

    def test_empty_question(self) -> None:
        assert not is_follow_up("   ")


class TestSessionHistory:
    def test_starts_empty(self) -> None:
        assert SessionHistory(5).is_empty

    def test_window_drops_oldest(self) -> None:
        history = SessionHistory(2)
        for number in range(4):
            history.add(f"вопрос {number}", f"ответ {number}")

        assert [turn.question for turn in history.turns] == ["вопрос 2", "вопрос 3"]

    def test_zero_pairs_disables_memory(self) -> None:
        """HISTORY_PAIRS=0 — это «памяти нет», а не «память без ограничения»."""
        history = SessionHistory(0)
        history.add("вопрос", "ответ")

        assert history.is_empty
        assert history.as_messages() == []

    def test_messages_alternate_roles(self) -> None:
        history = SessionHistory(5)
        history.add("что такое overlap", "перекрытие соседних чанков")

        assert history.as_messages() == [
            {"role": "user", "content": "что такое overlap"},
            {"role": "assistant", "content": "перекрытие соседних чанков"},
        ]

    def test_clear_reports_forgotten_pairs(self) -> None:
        history = SessionHistory(5)
        history.add("первый", "ответ")
        history.add("второй", "ответ")

        assert history.clear() == 2
        assert history.is_empty


class TestSearchQuery:
    def test_self_contained_question_goes_as_is(self) -> None:
        history = SessionHistory(5)
        history.add("что такое overlap", "перекрытие соседних чанков")

        assert history.search_query("как собрать индекс") == "как собрать индекс"

    def test_follow_up_picks_up_previous_question(self) -> None:
        """Ради этого история и заведена: «а сколько его ставить?» само не ищется."""
        history = SessionHistory(5)
        history.add("что такое overlap", "перекрытие соседних чанков")

        query = history.search_query("а сколько его ставить?")

        assert "overlap" in query
        assert query.endswith("а сколько его ставить?")

    def test_empty_history_leaves_question_alone(self) -> None:
        assert SessionHistory(5).search_query("а сколько его ставить?") == "а сколько его ставить?"

    def test_lookback_zero_disables_rewriting(self) -> None:
        history = SessionHistory(5, lookback=0)
        history.add("что такое overlap", "перекрытие соседних чанков")

        assert history.search_query("а сколько его ставить?") == "а сколько его ставить?"

    def test_lookback_limits_how_much_is_dragged_in(self) -> None:
        """Чем больше чужого текста в запросе, тем дальше выдача от текущего вопроса."""
        history = SessionHistory(5, lookback=1)
        history.add("что такое чанкинг", "разбиение текста")
        history.add("что такое overlap", "перекрытие соседних чанков")

        query = history.search_query("а сколько его ставить?")

        assert "overlap" in query
        assert "чанкинг" not in query


class TestFollowUpChain:
    """Цепочка уточнений не должна терять предмет разговора."""

    def test_third_reply_keeps_the_topic(self) -> None:
        """«а почему так?» после «а сколько его ставить?» — предметных слов ноль."""
        history = SessionHistory(5)
        history.add("что такое overlap", "перекрытие соседних чанков")
        history.add("а сколько его ставить", "10-20%")

        query = history.search_query("а почему так")

        assert "overlap" in query
        assert query.endswith("а почему так")

    def test_query_does_not_grow_with_the_chain(self) -> None:
        """Держим тему разговора, а не его историю: иначе запрос растёт без предела."""
        history = SessionHistory(5)
        history.add("что такое overlap", "перекрытие")
        for question in ("а сколько его ставить", "а почему так", "а если больше"):
            query = history.search_query(question)
            history.add(question, "ответ")

        assert query.count("\n") == 1
        assert "overlap" in query

    def test_lookback_two_takes_two_topics(self) -> None:
        """Проверка с lookback=2: при lookback=1 срез неотличим от жёсткого [-1:]."""
        history = SessionHistory(5, lookback=2)
        history.add("что такое чанкинг", "разбиение текста")
        history.add("что такое overlap", "перекрытие")

        query = history.search_query("а сколько его ставить")

        assert "чанкинг" in query
        assert "overlap" in query
        assert query.endswith("а сколько его ставить")

    def test_dialogue_started_with_a_follow_up(self) -> None:
        """Опоры нет — берём последний заданный вопрос, лучше неточная, чем никакой."""
        history = SessionHistory(5)
        history.add("а это точно так", "да")

        assert history.search_query("а почему") == "а это точно так\nа почему"
