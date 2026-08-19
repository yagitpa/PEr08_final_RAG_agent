"""
Манифест индекса: чем и как он был собран.

Зачем. Главная тихая поломка RAG — индекс собран одной моделью эмбеддингов, а
запрос векторизуется другой. Поиск при этом не падает: он честно находит
ближайшие векторы, только в чужом пространстве, и возвращает правдоподобный
мусор. Заметить это по выдаче почти невозможно.

Манифест лежит рядом с базой и сверяется при каждом запросе. Расхождения
разделены на два класса, и это разделение принципиально:

* **Несовместимость** (модель или размерность) — отказ. Работать нельзя.
* **Устаревание** (параметры чанкинга, версия очистки) — предупреждение.
  Индекс валиден, просто собран по другим правилам; пересборка желательна,
  но данные не испорчены.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from zerocoder_assistant.config.settings import ChunkingConfig

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "index_manifest.json"

#: Версия правил препроцессинга. Поднимается вручную, когда меняются очистка,
#: отбор секций или чанкинг настолько, что старый индекс стоит пересобрать.
PREPROCESSING_VERSION = 1


@dataclass(frozen=True, slots=True)
class IndexManifest:
    """Паспорт собранного индекса."""

    collection: str
    embed_provider: str
    embed_model: str
    dimension: int
    chunk_target_tokens: int
    chunk_max_tokens: int
    chunk_min_tokens: int
    chunk_overlap_pct: int
    tokenizer_encoding: str
    preprocessing_version: int
    chunks: int
    notes: int
    built_at: str

    @classmethod
    def create(
        cls,
        *,
        collection: str,
        embed_provider: str,
        embed_model: str,
        dimension: int,
        chunking: ChunkingConfig,
        chunks: int,
        notes: int,
    ) -> IndexManifest:
        return cls(
            collection=collection,
            embed_provider=embed_provider,
            embed_model=embed_model,
            dimension=dimension,
            chunk_target_tokens=chunking.target_tokens,
            chunk_max_tokens=chunking.max_tokens,
            chunk_min_tokens=chunking.min_tokens,
            chunk_overlap_pct=chunking.overlap_pct,
            tokenizer_encoding=chunking.encoding,
            preprocessing_version=PREPROCESSING_VERSION,
            chunks=chunks,
            notes=notes,
            built_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )

    # -- хранение ----------------------------------------------------------

    def save(self, directory: Path) -> Path:
        path = directory / MANIFEST_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, directory: Path) -> IndexManifest | None:
        """Манифест из каталога, либо None, если его нет или он нечитаем.

        Повреждённый манифест не роняет запуск: он равносилен отсутствующему,
        а отсутствующий означает «индекс ещё не собран».
        """
        path = directory / MANIFEST_FILENAME
        if not path.is_file():
            return None

        try:
            data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Манифест %s нечитаем (%s) — считаем, что индекса нет", path, exc)
            return None

        known = {field.name for field in fields(cls)}
        missing = known - data.keys()
        if missing:
            logger.warning("В манифесте %s нет полей %s — считаем, что индекса нет", path, missing)
            return None

        return cls(**{key: value for key, value in data.items() if key in known})

    # -- сверка ------------------------------------------------------------

    def incompatibility(self, embed_provider: str, embed_model: str) -> str | None:
        """Причина, по которой этим индексом пользоваться нельзя, либо None.

        Сверяется только модель эмбеддингов: именно она задаёт пространство
        векторов. Провайдер входит в сообщение для внятности — один и тот же
        `text-embedding-3-small` у OpenAI и ProxyAPI даёт совместимые векторы,
        поэтому расхождение по провайдеру само по себе не блокирует.
        """
        if self.embed_model != embed_model:
            return (
                f"Индекс собран моделью {self.embed_model!r} "
                f"(провайдер {self.embed_provider}), а сейчас настроена "
                f"{embed_model!r} (провайдер {embed_provider}). "
                "Векторы разных моделей несравнимы: поиск вернул бы правдоподобный мусор."
            )
        return None

    def staleness(self, chunking: ChunkingConfig) -> list[str]:
        """Расхождения, при которых индекс устарел, но остаётся работоспособным."""
        reasons: list[str] = []
        if self.preprocessing_version != PREPROCESSING_VERSION:
            reasons.append(
                f"правила препроцессинга обновились "
                f"(в индексе v{self.preprocessing_version}, сейчас v{PREPROCESSING_VERSION})"
            )

        changed = [
            f"{name}: {was} -> {now}"
            for name, was, now in (
                ("target_tokens", self.chunk_target_tokens, chunking.target_tokens),
                ("max_tokens", self.chunk_max_tokens, chunking.max_tokens),
                ("min_tokens", self.chunk_min_tokens, chunking.min_tokens),
                ("overlap_pct", self.chunk_overlap_pct, chunking.overlap_pct),
                ("encoding", self.tokenizer_encoding, chunking.encoding),
            )
            if was != now
        ]
        if changed:
            reasons.append("изменились параметры чанкинга (" + "; ".join(changed) + ")")
        return reasons
