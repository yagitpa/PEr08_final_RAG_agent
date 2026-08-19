"""
Генерация ответа по найденным фрагментам.

Полная цепочка запроса:

    вопрос -> поисковый запрос (с учётом истории)
           -> [L3] кэш ответа, только при пустой истории
           -> поиск (кэши L1 и L2 внутри)
           -> пусто? -> отказ БЕЗ вызова модели
           -> сборка контекста -> модель -> проверка ссылок
           -> [L3] сохранение, только при пустой истории

Два правила определяют весь класс:

**Пустая выдача — ответ, а не ошибка.** Порог релевантности имеет смысл ровно
потому, что после него можно честно сказать «в конспектах этого нет». Позвать
модель с пустым контекстом означало бы попросить её ответить по общим знаниям —
то есть отменить всё, ради чего строился RAG.

**Ответ с историей не кешируется.** Реплика «а подробнее?» осмысленна только
после предыдущей, а ключ кэша про предыдущую ничего не знает. Сохранить такой
ответ значит отдать его в следующий раз, когда та же фраза прозвучит в другом
разговоре. Уровни L1 и L2 от истории не зависят и работают всегда.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from zerocoder_assistant.cache.keys import answer_key
from zerocoder_assistant.cache.sqlite_cache import SqliteCache
from zerocoder_assistant.config.settings import Settings, get_settings
from zerocoder_assistant.generation.context_builder import (
    ContextBuilder,
    render_user_message,
    unknown_citations,
)
from zerocoder_assistant.generation.prompts import Prompt
from zerocoder_assistant.llm.base import ChatMessage, LLMProvider
from zerocoder_assistant.llm.factory import build_llm_provider
from zerocoder_assistant.memory.session_history import SessionHistory
from zerocoder_assistant.observability.timing import Stopwatch
from zerocoder_assistant.preprocessing.tokenization import get_token_counter
from zerocoder_assistant.retrieval.retriever import RetrievalResult, Retriever

logger = logging.getLogger(__name__)

#: Ответ, когда после порога релевантности не осталось ни одного фрагмента.
#: Формулировка обещает ровно то, что система проверила: смотрели конспекты,
#: ничего достаточно близкого не нашли. Про «такого не существует» речи нет.
NO_CONTEXT_ANSWER = (
    "В конспектах ничего подходящего не нашлось.\n"
    "Попробуйте переформулировать вопрос ближе к словам из урока или снимите "
    "фильтры, если задавали их."
)


@dataclass(frozen=True, slots=True)
class Answer:
    """Ответ ассистента вместе с тем, как он получен."""

    question: str
    text: str
    sources: list[str] = field(default_factory=list)
    from_cache: bool = False
    grounded: bool = True
    used_fragments: int = 0
    dropped_fragments: int = 0
    context_tokens: int = 0
    unknown_citations: list[int] = field(default_factory=list)
    search_query: str | None = None
    retrieval: RetrievalResult | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def rewritten_search(self) -> bool:
        """Искали не тем текстом, который ввёл студент, — по истории диалога."""
        return self.search_query is not None and self.search_query != self.question


class Answerer:
    """Отвечает на вопрос по базе знаний."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        retriever: Retriever | None = None,
        llm: LLMProvider | None = None,
        cache: SqliteCache | None = None,
        prompt: Prompt | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._cache = cache
        # Порядок важен: промпт и клиент модели читаются с диска и из настроек,
        # падают чаще всего (нет файла, нет ключа) и стоят дёшево. Открытое до
        # них хранилище пришлось бы закрывать из __init__, потому что `with`
        # получает объект только после его возврата.
        self._llm = llm or build_llm_provider(self._settings)
        self._prompt = prompt or Prompt.load(self._settings.answer_prompt_path)
        self._context = ContextBuilder(
            self._settings.max_context_tokens,
            counter=get_token_counter(self._settings.tokenizer_encoding),
        )
        self._retriever = retriever or Retriever(self._settings, cache=cache)
        self._owns_retriever = retriever is None

    # -- жизненный цикл ----------------------------------------------------

    def close(self) -> None:
        if self._owns_retriever:
            self._retriever.close()

    def __enter__(self) -> Answerer:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- ответ -------------------------------------------------------------

    def answer(
        self,
        question: str,
        *,
        history: SessionHistory | None = None,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> Answer:
        """Ответить на вопрос, опираясь только на базу знаний."""
        watch = Stopwatch()
        search_query = history.search_query(question) if history else question

        # Кэш ответов подключается, только когда история пуста: ключ не знает о
        # предыдущих репликах и на «а подробнее?» отдал бы чужое продолжение.
        cache = self._cache if use_cache and (history is None or history.is_empty) else None
        key = self._answer_key(question, top_k, where) if cache is not None else None

        if cache is not None and key is not None:
            cached = cache.get_answer(key)
            if cached is not None:
                text, sources = cached
                logger.debug("Ответ: попадание в кэш L3")
                # Ссылки пересчитываются, а не берутся из записи: иначе повтор
                # вопроса показывал бы галлюцинацию как проверенный ответ —
                # текст тот же, а предупреждение исчезло. Источников ровно
                # столько, сколько фрагментов ушло в контекст, так что счёт
                # восстанавливается по ним.
                return Answer(
                    question=question,
                    text=text,
                    sources=sources,
                    from_cache=True,
                    used_fragments=len(sources),
                    unknown_citations=unknown_citations(text, len(sources)),
                    search_query=search_query,
                    timings_ms=watch.finish(),
                )

        retrieval = self._retriever.retrieve(
            search_query, top_k=top_k, where=where, use_cache=use_cache
        )
        if retrieval.is_empty:
            logger.info("Ответ: контекст пуст, модель не вызывается")
            return Answer(
                question=question,
                text=NO_CONTEXT_ANSWER,
                grounded=False,
                search_query=search_query,
                retrieval=retrieval,
                timings_ms={**retrieval.timings_ms, **watch.finish()},
            )

        with watch.stage("context"):
            context = self._context.build(retrieval.chunks)
            messages = self._messages(question, context.text, history)

        with watch.stage("llm"):
            text = self._llm.chat(messages, max_tokens=self._settings.max_answer_tokens)

        invented = unknown_citations(text, context.used)
        if invented:
            logger.warning(
                "Модель сослалась на несуществующие фрагменты: %s (в контексте их %d)",
                ", ".join(f"[{number}]" for number in invented),
                context.used,
            )

        # Список источников идёт 1:1 с нумерацией фрагментов: ссылка [3] в
        # ответе должна находиться третьей строкой, иначе номер бесполезен.
        sources = [chunk.source for chunk in retrieval.chunks[: context.used]]
        if cache is not None and key is not None:
            cache.set_answer(key, question, self._params_label(), text, sources)

        return Answer(
            question=question,
            text=text,
            sources=sources,
            used_fragments=context.used,
            dropped_fragments=context.dropped,
            context_tokens=context.tokens,
            unknown_citations=invented,
            search_query=search_query,
            retrieval=retrieval,
            timings_ms={**retrieval.timings_ms, **watch.finish()},
        )

    # -- шаги ---------------------------------------------------------------

    def _messages(
        self, question: str, context: str, history: SessionHistory | None
    ) -> list[ChatMessage]:
        """Системный промпт, история диалога и вопрос с фрагментами.

        Фрагменты едут в сообщении пользователя, а не в системном. Это не
        оформление: в системное сообщение попадает то, что модель исполняет, а
        конспекты — данные, и в корпусе по промпт-инжинирингу они буквально
        полны чужих инструкций.
        """
        messages: list[ChatMessage] = [{"role": "system", "content": self._prompt.text}]
        if history is not None:
            messages.extend(history.as_messages())
        messages.append({"role": "user", "content": render_user_message(context, question)})
        return messages

    def _answer_key(self, question: str, top_k: int | None, where: dict[str, Any] | None) -> str:
        return answer_key(
            question,
            self._retriever.params(top_k, where),
            self._generation_params(),
        )

    def _generation_params(self) -> dict[str, Any]:
        """Всё, что влияет на текст ответа помимо найденных фрагментов."""
        return {
            "llm_model": self._llm.model_id,
            "temperature": self._settings.temperature,
            "max_answer_tokens": self._settings.max_answer_tokens,
            "max_context_tokens": self._settings.max_context_tokens,
            # Кодировка решает, сколько фрагментов влезет в бюджет: один и тот
            # же чанк — 177 токенов по o200k_base и 264 по cl100k_base. Без неё
            # в ключе смена TOKENIZER_ENCODING отдавала бы ответ, посчитанный
            # по другому числу фрагментов.
            "tokenizer_encoding": self._settings.tokenizer_encoding,
            "prompt": self._prompt.name,
            "prompt_version": self._prompt.version,
        }

    def _params_label(self) -> str:
        """Читаемая подпись параметров генерации — для колонки `params` в кэше.

        Ключ и так покрывает всё, что влияет на ответ, но по хешу не видно, чем
        именно запись отличается от соседней; эта строка отвечает на вопрос
        «каким промптом и какой моделью это посчитано» прямо в SQLite.
        """
        return f"{self._llm.model_id} | {self._prompt.name}@{self._prompt.version}"
