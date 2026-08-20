"""
Мост между ядром и оценкой через RAGAS.

RAGAS живёт в отдельном окружении (почему — в докстринге `ragas_runner`), и
ядро общается с ним так же, как пайплайны общаются между собой: через файл на
диске и запуск процесса. Здесь собирается набор данных, вызывается чужой
интерпретатор и разбирается его вывод.

Главное правило этого модуля: **отсутствие RAGAS не является ошибкой**. Если
окружение оценки не настроено, команда честно об этом говорит и завершается
успехом. Оценка, которая роняет сборку, потому что тяжёлая необязательная
зависимость не встала, — худший из возможных вариантов: её просто перестают
запускать вместе со всем остальным.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from zerocoder_assistant.errors import AssistantError

if TYPE_CHECKING:  # pragma: no cover - только для аннотаций
    from collections.abc import Sequence

    from zerocoder_assistant.config.settings import Settings
    from zerocoder_assistant.evaluation.metrics import QuestionOutcome

logger = logging.getLogger(__name__)

#: Путь к скрипту-оценщику относительно этого файла.
RUNNER_PATH = Path(__file__).with_name("ragas_runner.py")

#: Код возврата, которым скрипт сообщает «RAGAS здесь не установлен».
EXIT_RAGAS_MISSING = 3

DEFAULT_DATASET_PATH = Path("./storage/ragas_dataset.jsonl")
DEFAULT_SCORES_PATH = Path("./storage/ragas_scores.json")

#: Сколько ждать оценку. RAGAS делает несколько вызовов модели на вопрос и на
#: метрику, поэтому сорок вопросов идут минутами, а не секундами.
DEFAULT_TIMEOUT_SEC = 1800


class RagasNotConfigured(AssistantError):
    """Окружение для RAGAS не настроено. Это не поломка, а несделанная настройка."""


class RagasFailed(AssistantError):
    """Оценка запустилась, но завершилась ошибкой."""


@dataclass(frozen=True, slots=True)
class MetricScore:
    """Среднее по одной метрике и то, на скольких вопросах его удалось счесть."""

    name: str
    mean: float | None
    counted: int


@dataclass(frozen=True, slots=True)
class RagasReport:
    """Результат прогона RAGAS."""

    questions: int
    model: str
    metrics: tuple[MetricScore, ...]
    per_question: tuple[dict, ...] = field(default_factory=tuple)
    #: Почему не посчиталось НИЧЕГО. Заполняется только в этом случае: отчёт из
    #: одних «не посчитана» выглядит как результат, но не позволяет отличить
    #: неверный ключ от несовместимой версии библиотеки.
    diagnosis: str | None = None

    @classmethod
    def from_payload(cls, payload: dict) -> RagasReport:
        metrics = tuple(
            MetricScore(name=name, mean=values.get("mean"), counted=int(values.get("counted", 0)))
            for name, values in (payload.get("metrics") or {}).items()
        )
        return cls(
            questions=int(payload.get("questions", 0)),
            model=str(payload.get("model", "?")),
            metrics=metrics,
            per_question=tuple(payload.get("per_question") or ()),
            diagnosis=payload.get("diagnosis"),
        )


def build_dataset(outcomes: Sequence[QuestionOutcome]) -> list[dict]:
    """Отобрать вопросы, на которых метрики RAGAS вообще осмысленны.

    Берутся только отвечаемые вопросы, на которых модель действительно
    вызывалась и получила непустой контекст. Отказы сюда не попадают
    сознательно: `faithfulness` спрашивает, следует ли ответ из фрагментов, а
    отказ «в конспектах этого нет» из фрагментов не следует **по замыслу**.
    Посчитанный на отказах, он был бы низким там, где система ведёт себя
    правильно, — то есть наказывал бы за нужное поведение.
    """
    dataset = []
    for outcome in outcomes:
        if not outcome.question.answerable or not outcome.generated or not outcome.answer:
            continue
        if not outcome.contexts:
            continue
        dataset.append(
            {
                "id": outcome.question.id,
                "question": outcome.question.question,
                "answer": outcome.answer,
                "contexts": list(outcome.contexts),
            }
        )
    return dataset


def write_dataset(dataset: Sequence[dict], path: Path) -> Path:
    """Записать набор в JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in dataset:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def resolve_interpreter(settings: Settings) -> Path:
    """Интерпретатор окружения оценки.

    Отдельная проверка существования файла — чтобы опечатка в `RAGAS_PYTHON`
    называлась опечаткой, а не превращалась в `FileNotFoundError` из недр
    subprocess.
    """
    configured = settings.ragas_python
    if configured is None:
        raise RagasNotConfigured(
            "RAGAS_PYTHON не задан: не указано, каким интерпретатором запускать оценку.\n"
            "RAGAS ставится в ОТДЕЛЬНОЕ окружение — в общем он откатывает openai "
            "до 2.x, на котором ядро не работает.\n"
            "См. requirements-eval.txt: там и команды создания окружения, и причина."
        )
    if not configured.is_file():
        raise RagasNotConfigured(f"RAGAS_PYTHON указывает на несуществующий файл: {configured}")
    return configured


def evaluate_with_ragas(
    settings: Settings,
    dataset: Sequence[dict],
    *,
    dataset_path: Path | None = None,
    scores_path: Path | None = None,
    metrics: Sequence[str] | None = None,
    limit: int | None = None,
    timeout: int = DEFAULT_TIMEOUT_SEC,
) -> RagasReport:
    """Прогнать RAGAS отдельным интерпретатором и вернуть разобранный результат."""
    if not dataset:
        raise RagasFailed(
            "Нечего оценивать: ни одного вопроса с ответом модели и непустым контекстом."
        )

    interpreter = resolve_interpreter(settings)
    data_file = write_dataset(dataset, _resolve(dataset_path or DEFAULT_DATASET_PATH))
    out_file = _resolve(scores_path or DEFAULT_SCORES_PATH)

    credentials = settings.credentials(settings.llm_provider)
    command = [
        str(interpreter),
        str(RUNNER_PATH),
        "--dataset",
        str(data_file),
        "--out",
        str(out_file),
        "--model",
        settings.llm_model,
        "--embed-model",
        settings.embed_model,
        "--base-url",
        credentials.base_url,
    ]
    if metrics:
        command += ["--metrics", ",".join(metrics)]
    if limit is not None:
        command += ["--limit", str(limit)]

    # Ключ уезжает через окружение, а не через аргументы: список аргументов
    # виден любому процессу в системе, переменные окружения — нет.
    #
    # PYTHONIOENCODING задаётся здесь же: на Windows дочерний процесс печатает в
    # кодировке консоли (cp1251), и первое же русское сообщение об ошибке от
    # RAGAS роняло мост на разборе его же вывода.
    environment = {
        **os.environ,
        "OPENAI_API_KEY": credentials.api_key,
        "PYTHONIOENCODING": "utf-8",
    }

    logger.info("RAGAS: запускаю %s на %d вопросах", interpreter, len(dataset))
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            # Оценка не должна падать на кодировке чужого вывода: сообщение об
            # ошибке важнее его безупречного декодирования, а чужой процесс
            # может напечатать что угодно — вплоть до сырых байт из библиотеки.
            errors="replace",
            env=environment,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RagasFailed(f"Оценка не уложилась в {timeout} с") from exc

    if completed.returncode == EXIT_RAGAS_MISSING:
        raise RagasNotConfigured(
            f"В окружении {interpreter} нет RAGAS.\n"
            f"{completed.stderr.strip()}\n"
            "Установка: см. requirements-eval.txt"
        )
    if completed.returncode != 0:
        raise RagasFailed(
            f"Оценка завершилась с кодом {completed.returncode}.\n{completed.stderr.strip()}"
        )

    return RagasReport.from_payload(_parse(completed.stdout, out_file))


def _parse(stdout: str, out_file: Path) -> dict:
    """Разобрать вывод скрипта, при неудаче — прочитать записанный им файл.

    Библиотеки RAGAS печатают в stdout прогресс-бары и предупреждения, поэтому
    вывод не обязан быть чистым JSON. Файл надёжнее, но он может отсутствовать,
    если запись не удалась, — поэтому пробуются оба пути.
    """
    start = stdout.find("{")
    if start != -1:
        try:
            return json.loads(stdout[start:])
        except json.JSONDecodeError:
            logger.debug("RAGAS: stdout не разобрался как JSON, читаю %s", out_file)

    if out_file.is_file():
        return json.loads(out_file.read_text(encoding="utf-8"))
    raise RagasFailed("Оценка отработала, но результат не разобрался ни из вывода, ни из файла")


def _resolve(path: Path) -> Path:
    """Относительный путь раскрывается от корня проекта, а не от cwd."""
    from zerocoder_assistant.config.settings import PROJECT_ROOT

    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()
