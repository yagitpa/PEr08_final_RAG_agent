"""Оценка качества: golden set, встроенные метрики, подбор порога."""

from zerocoder_assistant.evaluation.golden_set import (
    DEFAULT_GOLDEN_SET,
    KIND_IN_DOMAIN_ABSENT,
    KIND_OUT_OF_DOMAIN,
    GoldenQuestion,
    GoldenSet,
    GoldenSetError,
)
from zerocoder_assistant.evaluation.metrics import (
    EvaluationReport,
    QuestionOutcome,
    RefusalStats,
    build_report,
    mean_reciprocal_rank,
    recall_at,
)
from zerocoder_assistant.evaluation.ragas_bridge import (
    RagasFailed,
    RagasNotConfigured,
    RagasReport,
    build_dataset,
    evaluate_with_ragas,
)
from zerocoder_assistant.evaluation.runner import (
    Candidate,
    Evaluator,
    ThresholdPoint,
    thresholds_range,
)

__all__ = [
    "DEFAULT_GOLDEN_SET",
    "KIND_IN_DOMAIN_ABSENT",
    "KIND_OUT_OF_DOMAIN",
    "Candidate",
    "EvaluationReport",
    "Evaluator",
    "GoldenQuestion",
    "GoldenSet",
    "GoldenSetError",
    "QuestionOutcome",
    "RagasFailed",
    "RagasNotConfigured",
    "RagasReport",
    "RefusalStats",
    "ThresholdPoint",
    "build_dataset",
    "build_report",
    "evaluate_with_ragas",
    "mean_reciprocal_rank",
    "recall_at",
    "thresholds_range",
]
