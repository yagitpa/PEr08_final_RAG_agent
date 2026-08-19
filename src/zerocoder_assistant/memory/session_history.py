"""
Короткая память диалога в пределах сессии.

Хранится в оперативной памяти и умирает вместе с процессом. Это не упущение:
долгоживущая память между запусками — отдельная задача с отдельными вопросами
(чей это диалог, когда его забывать, что делать при смене индекса), и
подмешивать её к RAG на этом этапе значит получить два недоделанных механизма
вместо одного работающего.

История ограничена сверху `history_pairs` парами реплик. Ограничение нужно не
ради экономии токенов, а ради качества: чем длиннее хвост диалога в запросе, тем
охотнее модель отвечает по нему, а не по фрагментам базы знаний.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from zerocoder_assistant.llm.base import ChatMessage
from zerocoder_assistant.memory.follow_up import is_follow_up

#: Разделитель между предыдущим и текущим вопросом в поисковом запросе.
#: Перевод строки, а не пробел: два вопроса не должны склеиться в одну фразу.
QUERY_JOIN = "\n"


@dataclass(frozen=True, slots=True)
class Turn:
    """Одна пара реплик: вопрос студента и ответ ассистента."""

    question: str
    answer: str


class SessionHistory:
    """Последние пары реплик текущей сессии."""

    def __init__(self, max_pairs: int = 5, *, lookback: int = 1) -> None:
        # maxlen=0 — это и есть «память выключена»: очередь молча отбрасывает
        # всё, что в неё кладут, и отдельной ветки для history_pairs=0 не нужно.
        self._turns: deque[Turn] = deque(maxlen=max_pairs)
        self._lookback = lookback

    def __len__(self) -> int:
        return len(self._turns)

    @property
    def is_empty(self) -> bool:
        """Пустая история — единственное условие, при котором ответ кешируется."""
        return not self._turns

    @property
    def turns(self) -> tuple[Turn, ...]:
        return tuple(self._turns)

    def add(self, question: str, answer: str) -> None:
        """Запомнить пару реплик."""
        self._turns.append(Turn(question=question, answer=answer))

    def clear(self) -> int:
        """Забыть диалог. Возвращает число забытых пар — команде `/clear` есть что сказать."""
        forgotten = len(self)
        self._turns.clear()
        return forgotten

    def as_messages(self) -> list[ChatMessage]:
        """История в виде сообщений для модели.

        Ответы ассистента едут как есть, вместе со ссылками `[1]`, `[2]`. Номера
        в них относятся к прошлым наборам фрагментов и в текущем недействительны —
        системный промпт про это предупреждает отдельным требованием.
        """
        messages: list[ChatMessage] = []
        for turn in self.turns:
            messages.append({"role": "user", "content": turn.question})
            messages.append({"role": "assistant", "content": turn.answer})
        return messages

    def search_query(self, question: str) -> str:
        """Текст, который пойдёт в поиск.

        Совпадает с вопросом, пока вопрос самодостаточен. Уточняющий вопрос
        дополняется предыдущими — иначе искать в базе будет попросту нечего.
        """
        if self._lookback <= 0 or self.is_empty or not is_follow_up(question):
            return question

        previous = [turn.question for turn in self.turns[-self._lookback :]]
        return QUERY_JOIN.join([*previous, question])
