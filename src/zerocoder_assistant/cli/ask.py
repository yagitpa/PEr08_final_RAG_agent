"""
Команда `ask` — ответ по базе знаний.

Два режима одной командой. Разовый вопрос отвечает и завершается; диалог
(`--repl`) держит короткую память сессии, поэтому уточнение «а сколько его
ставить?» понимается по предыдущей реплике. Память живёт в процессе и умирает
вместе с ним — `/clear` очищает её досрочно.

Разделение существенно, а не косметическое: при непустой истории ответ не
кешируется, потому что ключ кэша не знает о предыдущих репликах. Разовый вопрос
и третья реплика диалога — это разные режимы работы кэша.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import click

from zerocoder_assistant.cli.options import retrieval_options
from zerocoder_assistant.cli.repl_commands import (
    COMMAND_CLEAR,
    COMMAND_HELP,
    COMMANDS_EXIT,
    handle_command,
    is_command,
)
from zerocoder_assistant.config.settings import get_settings
from zerocoder_assistant.errors import AssistantError
from zerocoder_assistant.reporting import render_answer

if TYPE_CHECKING:  # pragma: no cover - только для аннотаций
    from zerocoder_assistant.generation import Answerer
    from zerocoder_assistant.memory import SessionHistory

#: Приглашение ввода в диалоговом режиме.
REPL_PROMPT = "вопрос"

REPL_GREETING = (
    "Диалог с ассистентом. Память держит последние {pairs} пар реплик.\n"
    "{clear} — забыть диалог, {exit} — выйти, {help} — справка."
)

SEPARATOR = "-" * 72


@click.command(name="ask")
@click.argument("question", metavar="ВОПРОС", required=False)
@retrieval_options
@click.option("--repl", is_flag=True, help="Диалог с памятью вместо разового вопроса.")
@click.option("--verbose", is_flag=True, help="Показать отбор, размер контекста и тайминги.")
def ask(
    question: str | None,
    lessons: tuple[str, ...],
    modules: tuple[int, ...],
    content_types: tuple[str, ...],
    top_k: int | None,
    no_cache: bool,
    repl: bool,
    verbose: bool,
) -> None:
    """Ответить на ВОПРОС по конспектам либо начать диалог (--repl)."""
    from zerocoder_assistant.cache import SqliteCache
    from zerocoder_assistant.generation import Answerer
    from zerocoder_assistant.memory import SessionHistory
    from zerocoder_assistant.retrieval import build_where

    if not question and not repl:
        raise click.UsageError("Задайте ВОПРОС или запустите диалог с --repl.")

    settings = get_settings()
    where = build_where(lessons=lessons, modules=modules, content_types=content_types)
    cache = None if no_cache else SqliteCache(settings.cache_db)
    history = (
        SessionHistory(settings.history_pairs, lookback=settings.follow_up_lookback)
        if repl
        else None
    )
    options = _Options(top_k=top_k, where=where, use_cache=not no_cache, verbose=verbose)

    try:
        with Answerer(settings, cache=cache) as answerer:
            if question:
                # Разовый вопрос отвечает командой: сбой = ненулевой код возврата.
                # В диалоге тот же сбой не должен уносить память сессии.
                if history is None:
                    _respond(answerer, question, history, options)
                else:
                    _respond_safely(answerer, question, history, options)
            if history is not None:
                _run_repl(answerer, history, options, settings.history_pairs)
    except AssistantError as exc:
        raise click.ClickException(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class _Options:
    """Параметры прогона, одинаковые для всех реплик диалога."""

    top_k: int | None
    where: dict[str, Any] | None
    use_cache: bool
    verbose: bool


def _run_repl(
    answerer: Answerer, history: SessionHistory, options: _Options, history_pairs: int
) -> None:
    """Цикл диалога до `/exit`, конца ввода или Ctrl+C."""
    click.echo("")
    click.echo(
        REPL_GREETING.format(
            pairs=history_pairs,
            clear=COMMAND_CLEAR,
            exit=COMMANDS_EXIT[0],
            help=COMMAND_HELP,
        )
    )

    while True:
        try:
            line = click.prompt(f"\n{REPL_PROMPT}", prompt_suffix="> ").strip()
        except (click.Abort, EOFError):
            click.echo("")
            return

        if not line:
            continue
        if is_command(line):
            outcome = handle_command(line, history)
            if outcome.exit_requested:
                return
            click.echo(outcome.message)
            continue

        _respond_safely(answerer, line, history, options)
        click.echo(SEPARATOR)


def _respond_safely(
    answerer: Answerer, question: str, history: SessionHistory, options: _Options
) -> None:
    """Ответить, не уронив диалог из-за внешнего сбоя.

    Обрыв сети, тайм-аут, исчерпанная квота — обстоятельства, а не поломка
    программы, и стоят они студенту всей памяти сессии: разговор, в котором
    уже был контекст, приходилось начинать заново. Сообщение печатается
    строкой, приглашение возвращается, следующий вопрос задаётся сразу.

    Ловится только предсказуемый отказ. Настоящая ошибка в коде обязана
    прервать работу и показать трассировку — глотать её здесь значило бы
    прятать дефект за приглашением ввода.

    Ошибки, при которых ассистент не работает в принципе (нет промпта, индекс
    собран другой моделью), сюда не попадают: они возникают при создании
    `Answerer`, то есть до входа в цикл.
    """
    try:
        _respond(answerer, question, history, options)
    except AssistantError as exc:
        click.echo("")
        click.echo(f"Не получилось ответить. {exc}")


def _respond(
    answerer: Answerer, question: str, history: SessionHistory | None, options: _Options
) -> None:
    """Задать вопрос, показать ответ и запомнить пару реплик."""
    answer = answerer.answer(
        question,
        history=history,
        top_k=options.top_k,
        where=options.where,
        use_cache=options.use_cache,
    )
    click.echo("")
    click.echo(render_answer(answer, verbose=options.verbose))

    # Отказ «в конспектах этого нет» в историю не кладём: предмета разговора он
    # не добавляет, а следующий уточняющий вопрос утянет его в поисковый запрос
    # и испортит выдачу.
    if history is not None and answer.grounded:
        history.add(question, answer.text)
