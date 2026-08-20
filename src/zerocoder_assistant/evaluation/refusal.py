"""
Опознание отказа в тексте ответа.

Нужно оценке: порог релевантности держит только вопросы, далёкие от корпуса
целиком, а на смежной теме фрагменты находятся с высоким сходством, и отказать
может лишь сама модель. Проверить это можно, только заглянув в текст.

Правило требует **двух** признаков в одном предложении: упоминания границ базы
знаний и отрицания. Проверка по одному слову («в конспектах») не работает —
системный промпт сам велит модели употреблять этот оборот в утвердительном
ответе, и метрика начинает засчитывать галлюцинацию как отказ.
"""

from __future__ import annotations

from zerocoder_assistant.config.constants import REFUSAL_NEGATIONS, REFUSAL_SCOPE_WORDS
from zerocoder_assistant.preprocessing.markdown import split_sentences


def looks_like_refusal(text: str) -> bool:
    """Сказано ли в ответе, что нужного в базе знаний нет.

    Предложение — правильная единица проверки. «В конспектах есть [1], но
    порядок шагов не описан» и «в конспектах этого нет» отличаются не набором
    слов, а тем, стоят ли отрицание и упоминание базы рядом.
    """
    return any(_is_refusing(sentence) for sentence in split_sentences(text))


def _is_refusing(sentence: str) -> bool:
    lowered = sentence.lower()
    return any(word in lowered for word in REFUSAL_SCOPE_WORDS) and any(
        negation in lowered for negation in REFUSAL_NEGATIONS
    )
