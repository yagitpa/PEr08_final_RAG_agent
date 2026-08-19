"""Генерация ответа по найденным фрагментам."""

from zerocoder_assistant.generation.answerer import NO_CONTEXT_ANSWER, Answer, Answerer
from zerocoder_assistant.generation.context_builder import (
    BuiltContext,
    ContextBuilder,
    cited_numbers,
    render_user_message,
    unknown_citations,
)
from zerocoder_assistant.generation.prompts import Prompt, PromptNotFoundError

__all__ = [
    "NO_CONTEXT_ANSWER",
    "Answer",
    "Answerer",
    "BuiltContext",
    "ContextBuilder",
    "Prompt",
    "PromptNotFoundError",
    "cited_numbers",
    "render_user_message",
    "unknown_citations",
]
