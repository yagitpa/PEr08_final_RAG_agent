"""
Слияние мелких соседних секций.

Зачем это нужно. Конспекты состоят не только из развёрнутых разделов: заметная
часть материала оформлена короткими подразделами по 50-150 токенов («Промпты»,
«Чат», «Диалоги» — по два-три предложения каждый). Посекционное разбиение в
чистом виде превращает их в чанки, которые находятся поиском, но не содержат
достаточно контекста, чтобы по ним можно было ответить.

Что делаем. Идущие подряд секции-сёстры (с общим непосредственным родителем)
объединяются в один чанк, пока суммарный размер не достигнет целевого. Заголовок
каждой вошедшей секции сохраняется в теле, поэтому знание «это раздел „Промпты“»
не теряется, а contextual header берётся от общего родителя.

Почему это не противоречит правилу «одна мысль — один чанк». Объединяются только
сёстры под одним заголовком, то есть части одной темы, а не произвольные соседи.
Секция, которая и так дотягивает до целевого размера, не трогается.
"""

from __future__ import annotations

from dataclasses import dataclass

from zerocoder_assistant.config.settings import ChunkingConfig
from zerocoder_assistant.preprocessing.headings import is_summary_section, sanitize_heading
from zerocoder_assistant.preprocessing.models import Block, Section
from zerocoder_assistant.preprocessing.tokenization import TokenCounter

#: Минимальная глубина пути, при которой у секции есть настоящий родитель.
#: Секции верхнего уровня не сливаются: их «родитель» — документ целиком.
MIN_MERGEABLE_DEPTH = 2


@dataclass(frozen=True, slots=True)
class PreparedSection:
    """Секция с уже очищенными блоками и известным размером."""

    section: Section
    blocks: list[Block]
    tokens: int

    @property
    def parent_path(self) -> tuple[str, ...]:
        return self.section.heading_path[:-1]


class SectionMerger:
    """Объединяет идущие подряд мелкие секции-сёстры."""

    def __init__(self, config: ChunkingConfig, counter: TokenCounter) -> None:
        self._config = config
        self._counter = counter

    def merge(self, sections: list[PreparedSection]) -> list[PreparedSection]:
        """Возвращает секции в исходном порядке, часть из них — объединённые."""
        result: list[PreparedSection] = []
        group: list[PreparedSection] = []

        def flush() -> None:
            if not group:
                return
            result.append(group[0] if len(group) == 1 else self._combine(group))
            group.clear()

        for prepared in sections:
            if not self._is_mergeable(prepared):
                flush()
                result.append(prepared)
                continue

            if group and not self._fits(group, prepared):
                flush()
            group.append(prepared)

        flush()
        return result

    def _is_mergeable(self, prepared: PreparedSection) -> bool:
        """Секция достаточно мелкая, имеет настоящего родителя и не является резюме.

        Резюме урока («Результат дня», «Итоги урока») не втягивается в группы:
        объединённая секция берёт заголовок родителя, а тип содержимого
        определяется по заголовку — то есть слияние стирало бы метку
        `content_type=summary`. А именно эти чанки отвечают на вопрос
        «о чём был урок N», и метка нужна им для фильтрации.
        """
        return (
            prepared.tokens < self._config.target_tokens
            and len(prepared.section.heading_path) >= MIN_MERGEABLE_DEPTH
            and not is_summary_section(prepared.section.heading)
        )

    def _fits(self, group: list[PreparedSection], candidate: PreparedSection) -> bool:
        """Можно ли добавить секцию к текущей группе."""
        if candidate.parent_path != group[0].parent_path:
            return False
        total = sum(item.tokens for item in group) + candidate.tokens
        return total <= self._config.target_tokens

    def _combine(self, group: list[PreparedSection]) -> PreparedSection:
        """Склеивает группу в одну синтетическую секцию под общим родителем."""
        parent_path = group[0].parent_path
        first = group[0].section

        blocks: list[Block] = []
        for item in group:
            if item.section.heading:
                # Заголовок обязательно санировать: слияние идёт уже после очистки,
                # и сырой текст с эмодзи-маркерами иначе попал бы прямо в индекс.
                blocks.append(Block(sanitize_heading(item.section.heading), is_code=False))
            blocks.extend(item.blocks)

        combined = Section(
            heading=parent_path[-1] if parent_path else None,
            level=len(parent_path),
            heading_path=parent_path,
            body="",  # тело больше не используется: блоки уже подготовлены
            ordinal=first.ordinal,
        )
        tokens = sum(self._counter.count(block.text) for block in blocks)
        return PreparedSection(section=combined, blocks=blocks, tokens=tokens)
