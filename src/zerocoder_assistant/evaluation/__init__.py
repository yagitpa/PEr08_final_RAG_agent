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
    "RefusalStats",
    "ThresholdPoint",
    "build_report",
    "mean_reciprocal_rank",
    "recall_at",
    "thresholds_range",
]
