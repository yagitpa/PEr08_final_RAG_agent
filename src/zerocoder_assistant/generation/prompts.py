"""
Загрузка системного промпта из файла.

Промпт лежит в `prompts/`, а не в исходниках: его правят чаще, чем код, и
править его должно быть можно, не разыскивая строковый литерал среди классов.

Версия промпта — хеш его содержимого, а не номер в имени файла. Номер надо
помнить поднять, а хеш меняется сам. Он входит в ключ кэша ответов, поэтому
правка промпта автоматически промахивается мимо ответов, посчитанных по прежней
редакции, — вместо того чтобы месяц отдавать их как свежие.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from zerocoder_assistant.errors import AssistantError
from zerocoder_assistant.preprocessing.models import content_hash

logger = logging.getLogger(__name__)


class PromptNotFoundError(AssistantError):
    """Файл системного промпта отсутствует или пуст."""

    def __init__(self, path: Path, reason: str) -> None:
        self.path = path
        super().__init__(f"{reason}: {path}\nПромпты лежат в PROMPTS_DIR (см. .env.example).")


@dataclass(frozen=True, slots=True)
class Prompt:
    """Системный промпт вместе с отпечатком его содержимого."""

    name: str
    text: str
    version: str

    @classmethod
    def load(cls, path: Path) -> Prompt:
        """Прочитать промпт с диска.

        Отсутствующий файл — предсказуемый отказ, а не сбой: без промпта
        ассистент не «работает чуть хуже», он не работает вовсе, и подставлять
        вместо него умолчание из кода значило бы тихо сменить все правила ответа.
        """
        if not path.is_file():
            raise PromptNotFoundError(path, "Файл системного промпта не найден")

        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise PromptNotFoundError(path, "Файл системного промпта пуст")

        prompt = cls(name=path.name, text=text, version=content_hash(text))
        logger.debug("Промпт %s, версия %s, %d символов", prompt.name, prompt.version, len(text))
        return prompt
