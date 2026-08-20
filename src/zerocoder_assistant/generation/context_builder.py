"""
Сборка контекста для модели.

Фрагменты нумеруются `[1]`, `[2]`, ... — на эти номера модель ссылается в
ответе, и по ним же потом проверяется, не сослалась ли она на несуществующий
фрагмент. Нумерация идёт по порядку выдачи поиска, то есть по убыванию сходства.

Бюджет контекста режется **целыми фрагментами**, начиная с самых слабых. Обрезка
фрагмента посередине выглядит экономнее, но отрезает ровно то, чего в чанке и
так меньше всего, — конец мысли; в результате модель видит начало определения
без его сути и договаривает сама.

Вопрос ставится **после** фрагментов. Инструкция, оказавшаяся перед длинным
блоком данных, конкурирует с ним за внимание модели, а вопрос в конце ещё и
задаёт, зачем всё это было прочитано.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from zerocoder_assistant.preprocessing.models import content_hash
from zerocoder_assistant.preprocessing.tokenization import TokenCounter

if TYPE_CHECKING:  # pragma: no cover - только для аннотаций
    from zerocoder_assistant.retrieval.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

FRAGMENTS_HEADER = "Фрагменты конспектов:"
QUESTION_HEADER = "Вопрос студента:"
FRAGMENT_SEPARATOR = "\n\n"
BLOCK_SEPARATOR = "\n\n"

#: Листинги и инлайн-код. Изымаются ДО разбора ссылок: индексация в примере
#: на Python — не ссылка на фрагмент, а `x = [256]` не выдуманный источник.
#: Корпус — конспекты по промпт-инжинирингу, кода в ответах много, и без этого
#: индикатор галлюцинаций мерил бы долю ответов с листингами.
#:
#: Незакрытая ограда (```python без парной) не удаляется: обрезать до конца
#: текста опаснее, чем разобрать лишний фрагмент как прозу.
CODE_SPAN_PATTERN = re.compile(r"```.*?```|~~~.*?~~~|`[^`\n]*`", flags=re.DOTALL)

#: Ссылка на фрагмент в ответе модели. Формат тот же, что и в нумерации выше:
#: один источник истины на два направления — как пишем номера в контекст и как
#: читаем их обратно из ответа.
#:
#: Просмотр назад различает ссылку и индексацию по алфавиту: в этом корпусе
#: код латинский, а проза кириллическая. `items[0]` и `data[42]` — индексация,
#: `векторов[1]` — ссылка, приписанная к слову вплотную. Прежний вариант
#: запрещал перед скобкой любую букву и такую ссылку терял.
#:
#: Число не ограничено тремя цифрами намеренно: `[1234]` — тоже выдумка,
#: и её надо поймать.
CITATION_PATTERN = re.compile(r"(?<![A-Za-z_\]\)\'\"`])\[(\d+)\]")


def without_code(text: str) -> str:
    """Текст без листингов и инлайн-кода.

    Пробел вместо вырезанного куска, а не пустая строка: иначе слова по краям
    склеиваются и разбиение на предложения ошибается.
    """
    return CODE_SPAN_PATTERN.sub(" ", text)


@dataclass(frozen=True, slots=True)
class BuiltContext:
    """Готовый контекст и то, что от него пришлось отрезать."""

    text: str
    used: int
    dropped: int
    tokens: int
    truncated: bool = False

    @property
    def is_empty(self) -> bool:
        return self.used == 0


class ContextBuilder:
    """Превращает найденные фрагменты в сообщение для модели."""

    def __init__(self, max_tokens: int, *, counter: TokenCounter | None = None) -> None:
        self._max_tokens = max_tokens
        self._counter = counter or TokenCounter("o200k_base")

    def build(self, chunks: Sequence[RetrievedChunk]) -> BuiltContext:
        """Пронумерованные фрагменты, помещающиеся в бюджет."""
        blocks: list[str] = []
        tokens = 0
        truncated = False

        separator_cost = self._counter.count(FRAGMENT_SEPARATOR)

        for number, chunk in enumerate(chunks, 1):
            block = f"[{number}] {chunk.text}"
            # Разделителей на n блоков ровно n-1: перед первым его нет.
            cost = self._counter.count(block) + (separator_cost if blocks else 0)

            if tokens + cost <= self._max_tokens:
                blocks.append(block)
                tokens += cost
                continue

            if not blocks:
                # Первый же фрагмент не помещается: бюджет меньше одного чанка.
                # Отдать пустой контекст здесь означало бы отказ на вопрос, ответ
                # на который найден, — режем текст, но фрагмент отдаём.
                block = self._fit(block)
                blocks.append(block)
                tokens = self._counter.count(block)
                truncated = True
                logger.warning(
                    "MAX_CONTEXT_TOKENS=%d меньше одного фрагмента — фрагмент [1] обрезан",
                    self._max_tokens,
                )
            break

        dropped = len(chunks) - len(blocks)
        if dropped:
            logger.debug("Контекст: %d фрагментов, %d не поместилось", len(blocks), dropped)

        return BuiltContext(
            text=FRAGMENT_SEPARATOR.join(blocks),
            used=len(blocks),
            dropped=dropped,
            tokens=tokens,
            truncated=truncated,
        )

    def _fit(self, block: str) -> str:
        """Урезать текст до бюджета, ориентируясь по средней длине токена.

        Строка обязана иметь право стать пустой. Иначе при бюджете, в который
        не влезает даже один символ, срез упирается в длину 1, условие выхода
        не наступает никогда и цикл крутится вечно. Через настройки такой
        бюджет недостижим (`max_context_tokens` не меньше 200), но класс
        публичный, и зависание — худшая из возможных реакций на плохой аргумент.
        """
        while block and self._counter.count(block) > self._max_tokens:
            excess = self._counter.count(block) - self._max_tokens
            cut = max(1, len(block) * excess // max(1, self._counter.count(block)))
            block = block[: len(block) - cut]
        return block


#: Отпечаток обёртки, в которой фрагменты и вопрос едут модели.
#:
#: Входит в ключ кэша ответов наравне с версией системного промпта. Причина та
#: же: заголовки и порядок блоков — часть инструкции, которую видит модель, и
#: перестановка вопроса вперёд фрагментов меняет ответ. Пока отпечатка не было,
#: правка этих строк молча продолжала отдавать ответы, посчитанные по прежней
#: обёртке, — ровно тот тихий обман, ради которого версия промпта и заведена.
USER_TEMPLATE_VERSION = content_hash(
    "\x00".join([FRAGMENTS_HEADER, QUESTION_HEADER, BLOCK_SEPARATOR, FRAGMENT_SEPARATOR])
)


def render_user_message(context_text: str, question: str) -> str:
    """Сообщение пользователя: сначала фрагменты, в конце вопрос."""
    return BLOCK_SEPARATOR.join([FRAGMENTS_HEADER, context_text, QUESTION_HEADER, question.strip()])


def cited_numbers(answer: str) -> list[int]:
    """Номера фрагментов, на которые сослалась модель (в порядке первого упоминания)."""
    seen: dict[int, None] = {}
    for match in CITATION_PATTERN.finditer(without_code(answer)):
        seen.setdefault(int(match.group(1)), None)
    return list(seen)


def unknown_citations(answer: str, available: int) -> list[int]:
    """Ссылки на фрагменты, которых модели не давали.

    Дешёвая проверка обоснованности: сослаться на `[7]`, когда фрагментов пять,
    можно только придумав его. Ответ при этом не правится — правка чужого текста
    по регулярному выражению испортит его вернее, чем лишняя ссылка, — но факт
    попадает и в лог, и в отчёт, и на этапе 7 станет метрикой.
    """
    return [number for number in cited_numbers(answer) if not 1 <= number <= available]
