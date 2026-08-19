"""
Golden set: вопросы с известным правильным ответом.

Набор лежит в YAML, а не в коде: его правят по результатам прогонов, и правка
данных не должна быть правкой программы.

Проверка структуры при загрузке строгая. Набор — измерительный прибор, и
опечатка в нём не ломает прогон, а тихо сдвигает метрику: вопрос с пустым
списком уроков, помеченный отвечаемым, навсегда останется «не найденным» и
занизит recall, ничем себя не выдав.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from zerocoder_assistant.errors import AssistantError

#: Вопрос вообще не про курс («какая столица Австралии»). Проверяет грубый отказ.
KIND_OUT_OF_DOMAIN = "out_of_domain"

#: Тема смежная и правдоподобная, но в конспектах её нет. Главный класс набора:
#: именно здесь ассистент склонен придумать ответ, потому что похожие фрагменты
#: находятся и выглядят уместно.
KIND_IN_DOMAIN_ABSENT = "in_domain_absent"

UNANSWERABLE_KINDS = frozenset({KIND_OUT_OF_DOMAIN, KIND_IN_DOMAIN_ABSENT})

DEFAULT_GOLDEN_SET = Path("docs/golden_set.yaml")


class GoldenSetError(AssistantError):
    """Набор вопросов не читается или противоречив."""


@dataclass(frozen=True, slots=True)
class GoldenQuestion:
    """Один вопрос набора вместе с тем, что считается правильным исходом."""

    id: str
    question: str
    lessons: tuple[str, ...]
    answerable: bool
    kind: str | None = None
    note: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], position: int) -> GoldenQuestion:
        where = data.get("id") or f"вопрос №{position}"
        question = str(data.get("question", "")).strip()
        if not question:
            raise GoldenSetError(f"{where}: пустой текст вопроса")

        answerable = bool(data.get("answerable", True))
        lessons = tuple(str(lesson) for lesson in data.get("lessons") or ())
        kind = data.get("kind")

        if answerable and not lessons:
            raise GoldenSetError(
                f"{where}: помечен отвечаемым, но список уроков пуст — "
                "такой вопрос никогда не будет засчитан и занизит recall"
            )
        if not answerable and lessons:
            raise GoldenSetError(f"{where}: помечен неотвечаемым, но перечисляет уроки {lessons}")
        if not answerable and kind not in UNANSWERABLE_KINDS:
            raise GoldenSetError(
                f"{where}: у неотвечаемого вопроса нужен kind из "
                f"{', '.join(sorted(UNANSWERABLE_KINDS))}, получено {kind!r}"
            )

        return cls(
            id=str(data.get("id") or f"q{position:03d}"),
            question=question,
            lessons=lessons,
            answerable=answerable,
            kind=kind,
            note=data.get("note"),
        )


@dataclass(frozen=True, slots=True)
class GoldenSet:
    """Весь набор вопросов."""

    version: int
    questions: tuple[GoldenQuestion, ...]
    path: Path | None = None

    def __len__(self) -> int:
        return len(self.questions)

    @property
    def answerable(self) -> tuple[GoldenQuestion, ...]:
        return tuple(question for question in self.questions if question.answerable)

    @property
    def unanswerable(self) -> tuple[GoldenQuestion, ...]:
        return tuple(question for question in self.questions if not question.answerable)

    def lessons(self) -> set[str]:
        """Все упомянутые уроки — чтобы сверить набор с содержимым индекса."""
        return {lesson for question in self.questions for lesson in question.lessons}

    @classmethod
    def load(cls, path: Path) -> GoldenSet:
        if not path.is_file():
            raise GoldenSetError(f"Набор вопросов не найден: {path}")

        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise GoldenSetError(f"Набор вопросов не читается как YAML: {path}\n{exc}") from exc

        if not isinstance(data, dict) or not isinstance(data.get("questions"), list):
            raise GoldenSetError(f"{path}: ожидается словарь с ключом questions (список)")

        questions = tuple(
            GoldenQuestion.from_dict(item, position)
            for position, item in enumerate(data["questions"], 1)
        )
        if not questions:
            raise GoldenSetError(f"{path}: набор пуст")

        duplicates = _duplicated_ids(questions)
        if duplicates:
            raise GoldenSetError(f"{path}: повторяющиеся id: {', '.join(sorted(duplicates))}")

        return cls(version=int(data.get("version", 1)), questions=questions, path=path)


def _duplicated_ids(questions: tuple[GoldenQuestion, ...]) -> set[str]:
    counts = Counter(question.id for question in questions)
    return {identifier for identifier, count in counts.items() if count > 1}
