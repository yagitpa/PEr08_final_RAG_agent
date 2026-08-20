"""
Оценка через RAGAS. Запускается ОТДЕЛЬНЫМ интерпретатором.

Этот файл — единственный в проекте, который не является частью пакета: он не
импортирует ни одного модуля `zerocoder_assistant` и обходится стандартной
библиотекой плюс сам RAGAS. Так сделано не из аккуратности, а по необходимости.

RAGAS 0.2.15 тянет за собой `langchain-openai`, который ограничивает `openai`
версией 2.x. Ядро проекта работает на `openai` 3.x. Поставить RAGAS в то же
окружение — значит молча откатить клиент, которым ядро делает каждый запрос
эмбеддингов и каждый вызов модели. Проверено установкой: 3.3.0 -> 2.54.0.

Поэтому окружения два, а связаны они файлом на диске — ровно как связаны
пайплайны в остальной архитектуре. Ядро пишет набор данных в JSONL, этот
скрипт читает его своим интерпретатором и пишет оценки в JSON. Ни одна
библиотека не пересекает границу.

Запуск (обычно его делает `zassist eval ragas`, но руками тоже можно):

    <окружение-оценки>/python.exe src/zerocoder_assistant/evaluation/ragas_runner.py \
        --dataset storage/ragas_dataset.jsonl --out storage/ragas_scores.json

Формат входа — по строке JSON на вопрос:
`{"id": ..., "question": ..., "answer": ..., "contexts": [...]}`.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

#: Метрики, которые считаются без эталонных ответов.
#:
#: Эталонов у нас нет сознательно. Написать сорок образцовых ответов руками —
#: отдельный проект, а сгенерировать их той же моделью, которую потом ими же и
#: проверяют, — замкнутый круг: она согласится сама с собой. Поэтому
#: `context_recall`, которому эталон обязателен, не считается вовсе, а точность
#: контекста берётся в безэталонном варианте.
METRIC_FAITHFULNESS = "faithfulness"
METRIC_ANSWER_RELEVANCY = "answer_relevancy"
METRIC_CONTEXT_PRECISION = "context_precision"

DEFAULT_METRICS = (METRIC_FAITHFULNESS, METRIC_ANSWER_RELEVANCY, METRIC_CONTEXT_PRECISION)

#: Судья должен быть настолько детерминирован, насколько позволяет модель:
#: метрика, которая скачет от запуска к запуску, не метрика.
JUDGE_TEMPERATURE = 0.0

EXIT_RAGAS_MISSING = 3


class RagasUnavailable(RuntimeError):
    """RAGAS не установлен или не импортируется в этом интерпретаторе."""


def build_metrics(names, llm, embeddings):
    """Собрать метрики по именам: список пар (наше имя, объект RAGAS).

    Пара, а не просто объект, потому что имена не совпадают. В результатах
    RAGAS подписывает точность контекста как `llm_context_precision_without_
    reference` — по имени `context_precision` в них ничего не найдётся, и
    метрика молча окажется «не посчитанной» при полностью исправном прогоне.
    Ровно это и случилось на первом живом запуске.

    Импорты внутри функции, а не наверху файла: без установленного RAGAS модуль
    всё равно должен читаться и печатать `--help`, иначе разобраться, чего не
    хватает, можно будет только по трассировке.
    """
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithoutReference,
        ResponseRelevancy,
    )

    available = {
        METRIC_FAITHFULNESS: lambda: Faithfulness(llm=llm),
        METRIC_ANSWER_RELEVANCY: lambda: ResponseRelevancy(llm=llm, embeddings=embeddings),
        METRIC_CONTEXT_PRECISION: lambda: LLMContextPrecisionWithoutReference(llm=llm),
    }
    unknown = [name for name in names if name not in available]
    if unknown:
        raise SystemExit(
            f"Неизвестные метрики: {', '.join(unknown)}. Доступны: {', '.join(sorted(available))}"
        )
    return [(name, available[name]()) for name in names]


def load_dataset(path: Path) -> list[dict]:
    """Прочитать набор и проверить, что он пригоден для оценки.

    Проверка строгая по той же причине, что и у golden set: пустой контекст или
    пустой ответ не сломают прогон, а тихо утянут среднее вниз, и разбираться
    потом придётся с числом, а не с сообщением.
    """
    if not path.is_file():
        raise SystemExit(f"Набор данных не найден: {path}")

    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}, строка {number}: не разбирается как JSON — {exc}") from exc

        for field in ("id", "question", "answer", "contexts"):
            if not row.get(field):
                raise SystemExit(f"{path}, строка {number}: пустое поле {field!r}")
        if not isinstance(row["contexts"], list):
            raise SystemExit(f"{path}, строка {number}: contexts должен быть списком")
        rows.append(row)

    if not rows:
        raise SystemExit(f"{path}: набор пуст")
    return rows


def build_judge(model: str, embed_model: str, api_key: str, base_url: str):
    """Модель-судья и эмбеддер для метрик."""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=JUDGE_TEMPERATURE,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=embed_model, api_key=api_key, base_url=base_url)
    )
    return llm, embeddings


def run(
    rows: list[dict],
    metric_names: list[str],
    model: str,
    embed_model: str,
    api_key: str,
    base_url: str,
) -> dict:
    """Посчитать метрики и вернуть сводку вместе с разбивкой по вопросам."""
    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
    except ImportError as exc:  # pragma: no cover - зависит от окружения
        raise RagasUnavailable(str(exc)) from exc

    llm, embeddings = build_judge(model, embed_model, api_key, base_url)
    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=row["question"],
                response=row["answer"],
                retrieved_contexts=list(row["contexts"]),
            )
            for row in rows
        ]
    )

    pairs = build_metrics(metric_names, llm, embeddings)
    # Соответствие «наше имя -> имя, которым RAGAS подписывает результат».
    # Без него точность контекста ищется под несуществующим ключом.
    ragas_names = {name: metric.name for name, metric in pairs}
    metrics = [metric for _, metric in pairs]

    # По умолчанию RAGAS глотает исключения и ставит NaN. Для одного сбойного
    # вопроса это правильно — весь прогон из-за него ронять незачем.
    result = evaluate(dataset=dataset, metrics=metrics)

    # `result.scores` — список словарей, по одному на вопрос. Средние считаются
    # здесь, а не берутся из представления RAGAS: пропуски (None/NaN) надо
    # исключать явно, иначе одна несосчитанная метрика обнуляет среднее.
    per_question = []
    for row, scores in zip(rows, list(result.scores), strict=True):
        per_question.append({"id": row["id"], "scores": _clean(scores, ragas_names)})

    averages = _averages(per_question, metric_names)
    report = {
        "questions": len(rows),
        "model": model,
        "metrics": averages,
        "per_question": per_question,
    }

    # «Не посчитано» без причины — бесполезная строка: она выглядит как
    # результат, но не отличает сломанный ключ от несовместимой библиотеки и
    # от опечатки в имени метрики. Диагностируется КАЖДАЯ метрика, у которой
    # ноль засчитанных вопросов, а не только случай «не посчиталось вообще
    # ничего»: именно частичный отказ и оказался настоящим на первом прогоне.
    failed = [name for name, item in averages.items() if item["counted"] == 0]
    if failed:
        report["diagnosis"] = _diagnose(dataset, dict(pairs), failed)
    return report


def _diagnose(dataset, by_name: dict, failed: list[str]) -> str:
    """Повторить один вопрос по каждой несосчитанной метрике и назвать причину."""
    from ragas import EvaluationDataset, evaluate

    reasons = []
    sample = EvaluationDataset(samples=dataset.samples[:1])
    for name in failed:
        try:
            evaluate(dataset=sample, metrics=[by_name[name]], raise_exceptions=True)
        except Exception as exc:  # причина важнее её типа
            reasons.append(f"{name}: {type(exc).__name__}: {exc}")
        else:
            reasons.append(f"{name}: повторный прогон прошёл — причина не воспроизвелась")
    return "; ".join(reasons)


def _clean(scores, ragas_names: dict[str, str]) -> dict:
    """Оценки одного вопроса под НАШИМИ именами и без NaN.

    Ключи переводятся здесь, потому что дальше по коду метрика всюду зовётся
    так, как её назвал пользователь в `--metrics`.
    """
    raw = dict(scores)
    return {name: _as_number(raw.get(ragas_name)) for name, ragas_name in ragas_names.items()}


def _as_number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    # NaN не равен сам себе — единственный надёжный способ его опознать.
    return None if number != number else number


def _averages(per_question: list[dict], metric_names: list[str]) -> dict:
    """Среднее по каждой метрике и число вопросов, на которых её удалось счесть."""
    averages = {}
    for name in metric_names:
        values = [
            row["scores"][name] for row in per_question if row["scores"].get(name) is not None
        ]
        averages[name] = {
            "mean": round(statistics.fmean(values), 4) if values else None,
            "counted": len(values),
        }
    return averages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Оценка RAG через RAGAS в изолированном окружении.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Входной JSONL.")
    parser.add_argument("--out", type=Path, default=None, help="Куда записать оценки (JSON).")
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "gpt-4o-mini"))
    parser.add_argument(
        "--embed-model", default=os.environ.get("EMBED_MODEL", "text-embedding-3-small")
    )
    parser.add_argument(
        "--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Список метрик через запятую.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Взять только первые N вопросов.")
    args = parser.parse_args(argv)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("Нет OPENAI_API_KEY в окружении.", file=sys.stderr)
        return 2

    rows = load_dataset(args.dataset)
    if args.limit is not None:
        rows = rows[: args.limit]
    metric_names = [name.strip() for name in args.metrics.split(",") if name.strip()]

    try:
        report = run(rows, metric_names, args.model, args.embed_model, api_key, args.base_url)
    except RagasUnavailable as exc:
        print(f"RAGAS недоступен в этом интерпретаторе: {exc}", file=sys.stderr)
        return EXIT_RAGAS_MISSING

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")

    # Отчёт уходит в stdout всегда: вызывающая сторона читает именно его и не
    # обязана знать, куда лёг файл.
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
