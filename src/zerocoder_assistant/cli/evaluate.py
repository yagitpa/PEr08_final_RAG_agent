"""
Команды группы `eval` — оценка качества по golden set.

`run` по умолчанию не вызывает языковую модель: recall, MRR, корректность
отказа и латентность поиска считаются без неё. Прогон, который стоит денег,
включается флагом — иначе им перестают пользоваться, а метрика, которую не
запускают, не метрика.

`threshold` отвечает на вопрос «а что было бы при другом пороге», не гоняя
корпус заново на каждое значение: сходства от порога не зависят.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import click

from zerocoder_assistant.config.settings import PROJECT_ROOT, get_settings
from zerocoder_assistant.errors import AssistantError
from zerocoder_assistant.evaluation.golden_set import DEFAULT_GOLDEN_SET
from zerocoder_assistant.reporting import (
    render_cache_usage,
    render_evaluation,
    render_threshold_sweep,
)

if TYPE_CHECKING:  # pragma: no cover - только для аннотаций
    from zerocoder_assistant.cache import SqliteCache
    from zerocoder_assistant.config.settings import Settings
    from zerocoder_assistant.evaluation import EvaluationReport
    from zerocoder_assistant.generation import Answerer

#: Границы перебора по умолчанию. Ниже 0.20 отказов не остаётся вовсе, выше
#: 0.60 не остаётся ответов — крайности показывать незачем.
SWEEP_START = 0.20
SWEEP_STOP = 0.60
SWEEP_STEP = 0.02

golden_option = click.option(
    "--golden",
    "golden_path",
    type=click.Path(path_type=Path),
    default=None,
    metavar="ФАЙЛ",
    help=f"Набор вопросов; по умолчанию {DEFAULT_GOLDEN_SET}.",
)


@click.group(name="eval")
def eval_group() -> None:
    """Оценка качества поиска и ответов."""


@eval_group.command(name="run")
@golden_option
@click.option("--top-k", type=click.IntRange(min=1), default=None, help="Сколько фрагментов брать.")
@click.option(
    "--answers",
    is_flag=True,
    help="Дополнительно вызывать модель: проверка ссылок и время генерации. Стоит денег.",
)
@click.option("--no-cache", is_flag=True, help="Не использовать и не обновлять кэш.")
@click.option(
    "--export",
    type=click.Path(path_type=Path),
    default=None,
    metavar="ФАЙЛ",
    help="Выгрузить исходы по каждому вопросу в JSONL.",
)
def run(
    golden_path: Path | None,
    top_k: int | None,
    answers: bool,
    no_cache: bool,
    export: Path | None,
) -> None:
    """Прогнать golden set и показать метрики."""
    from zerocoder_assistant.cache import SqliteCache
    from zerocoder_assistant.evaluation import Evaluator, GoldenSet

    settings = get_settings()
    cache = None if no_cache else SqliteCache(settings.cache_db)

    try:
        golden = GoldenSet.load(_resolve(golden_path))
        with Evaluator(settings, cache=cache) as evaluator:
            answerer = _build_answerer(settings, cache) if answers else None
            try:
                report = evaluator.run(
                    golden, top_k=top_k, use_cache=not no_cache, answerer=answerer
                )
            finally:
                if answerer is not None:
                    answerer.close()
    except (AssistantError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(render_evaluation(report))
    if cache is not None:
        # Разбивка по уровням: общая доля попаданий в сводке не показывает,
        # какой именно уровень работает. На повторном прогоне это L1 и L2,
        # а L3 включается только с --answers.
        click.echo("")
        click.echo(render_cache_usage(cache.snapshot_usage()))
    if export is not None:
        _export(report, export)
        click.echo(f"\nИсходы выгружены: {export}")


@eval_group.command(name="threshold")
@golden_option
@click.option("--from", "start", type=float, default=SWEEP_START, help="Нижняя граница перебора.")
@click.option("--to", "stop", type=float, default=SWEEP_STOP, help="Верхняя граница перебора.")
@click.option("--step", type=float, default=SWEEP_STEP, help="Шаг перебора.")
@click.option("--top-k", type=click.IntRange(min=1), default=None, help="Сколько фрагментов брать.")
def threshold(
    golden_path: Path | None, start: float, stop: float, step: float, top_k: int | None
) -> None:
    """Подобрать порог релевантности по golden set.

    Корпус проходится один раз: сходства от порога не зависят, зависит только
    отсечение.
    """
    from zerocoder_assistant.cache import SqliteCache
    from zerocoder_assistant.evaluation import Evaluator, GoldenSet, thresholds_range

    settings = get_settings()

    try:
        golden = GoldenSet.load(_resolve(golden_path))
        with Evaluator(settings, cache=SqliteCache(settings.cache_db)) as evaluator:
            points = evaluator.sweep(golden, thresholds_range(start, stop, step), top_k=top_k)
    except (AssistantError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(render_threshold_sweep(points, settings.relevance_threshold))


def _resolve(path: Path | None) -> Path:
    """Путь к набору: относительный раскрывается от корня проекта, а не от cwd."""
    chosen = path or DEFAULT_GOLDEN_SET
    return chosen if chosen.is_absolute() else (PROJECT_ROOT / chosen).resolve()


def _build_answerer(settings: Settings, cache: SqliteCache | None) -> Answerer:
    from zerocoder_assistant.generation import Answerer

    return Answerer(settings, cache=cache)


def _export(report: EvaluationReport, path: Path) -> None:
    """Полный список исходов — по строке JSON на вопрос.

    В терминале перечислять полсотни строк бессмысленно, а разбирать их надо по
    одной, поэтому подробности живут в файле, а не в отчёте.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for outcome in report.outcomes:
            handle.write(
                json.dumps(
                    {
                        "id": outcome.question.id,
                        "question": outcome.question.question,
                        "answerable": outcome.question.answerable,
                        "kind": outcome.question.kind,
                        "expected": list(outcome.question.lessons),
                        "found": list(outcome.lessons),
                        "similarities": [round(value, 4) for value in outcome.similarities],
                        "refused": outcome.refused,
                        "correct": outcome.correct,
                        "answer": outcome.answer,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
