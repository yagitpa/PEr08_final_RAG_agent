"""
Настройки приложения, читаемые из окружения (.env).

Разделены на две сущности намеренно:

* `Settings` — полная конфигурация процесса: пути, провайдеры, поиск, генерация.
* `ChunkingConfig` — узкий срез, нужный препроцессингу.

Благодаря этому чанкер зависит не от «всего приложения», а от четырёх чисел,
и его можно вызвать в тесте без .env, ключей и путей.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Final

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from zerocoder_assistant.errors import MissingCredentialsError

#: Корень проекта: <root>/src/zerocoder_assistant/config/settings.py -> parents[3].
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]

#: Какие поля настроек описывают каждого провайдера: (ключ, адрес).
#: Добавить Сбер или Яндекс — значит дописать сюда строку и положить рядом
#: реализацию клиента; ядро при этом не меняется.
PROVIDER_FIELDS: Final[dict[str, tuple[str, str]]] = {
    "openai": ("openai_api_key", "openai_base_url"),
    "proxyapi": ("proxyapi_api_key", "proxyapi_base_url"),
}


class ProviderCredentials(BaseModel):
    """Разрешённые доступы к одному провайдеру."""

    model_config = {"frozen": True}

    api_key: str
    base_url: str


class ChunkingConfig(BaseModel):
    """Параметры разбиения на чанки. Размеры — в токенах, не в символах."""

    model_config = {"frozen": True}

    target_tokens: int = Field(default=400, ge=50, le=4000)
    max_tokens: int = Field(default=500, ge=50, le=8000)
    min_tokens: int = Field(default=80, ge=0, le=2000)
    overlap_pct: int = Field(default=15, ge=0, le=50)
    encoding: str = Field(default="o200k_base")

    @model_validator(mode="after")
    def _check_ordering(self) -> ChunkingConfig:
        if self.min_tokens >= self.target_tokens:
            raise ValueError(
                f"min_tokens ({self.min_tokens}) должен быть меньше "
                f"target_tokens ({self.target_tokens})"
            )
        if self.target_tokens > self.max_tokens:
            raise ValueError(
                f"target_tokens ({self.target_tokens}) не может превышать "
                f"max_tokens ({self.max_tokens})"
            )
        return self

    @property
    def overlap_tokens(self) -> int:
        """Размер перекрытия в токенах, посчитанный от целевого размера чанка."""
        return round(self.target_tokens * self.overlap_pct / 100)


class Settings(BaseSettings):
    """Конфигурация процесса. Все значения переопределяются через .env."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Провайдеры (этап 2) ------------------------------------------------
    llm_provider: str = "openai"
    embed_provider: str = "openai"

    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    proxyapi_api_key: str | None = None
    proxyapi_base_url: str = "https://api.proxyapi.ru/openai/v1"

    llm_model: str = "gpt-4o-mini"
    embed_model: str = "text-embedding-3-small"
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    # --- Пути ---------------------------------------------------------------
    notes_dir: Path = Path("./data/notes")
    raw_dir: Path = Path("./data/raw")
    chroma_dir: Path = Path("./storage/chroma")
    cache_db: Path = Path("./storage/cache.db")

    embed_batch_size: int = Field(default=64, ge=1, le=2048)
    request_timeout: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)

    # --- Чанкинг (этап 1) ---------------------------------------------------
    chunk_target_tokens: int = 400
    chunk_max_tokens: int = 500
    chunk_min_tokens: int = 80
    chunk_overlap_pct: int = 15
    tokenizer_encoding: str = "o200k_base"

    # --- Поиск и генерация (этапы 4-5) --------------------------------------
    top_k: int = Field(default=5, ge=1, le=50)
    overfetch_factor: int = Field(default=3, ge=1, le=10)
    relevance_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    max_context_tokens: int = Field(default=3000, ge=200)
    history_pairs: int = Field(default=5, ge=0, le=50)

    log_level: str = "INFO"

    @model_validator(mode="after")
    def _resolve_paths(self) -> Settings:
        """Относительные пути раскрываются от корня проекта, а не от cwd.

        Иначе поведение команд зависит от того, из какой папки их запустили.
        """
        for field in ("notes_dir", "raw_dir", "chroma_dir", "cache_db"):
            value: Path = getattr(self, field)
            if not value.is_absolute():
                object.__setattr__(self, field, (PROJECT_ROOT / value).resolve())
        return self

    @property
    def chunking(self) -> ChunkingConfig:
        """Срез настроек, который нужен препроцессингу."""
        return ChunkingConfig(
            target_tokens=self.chunk_target_tokens,
            max_tokens=self.chunk_max_tokens,
            min_tokens=self.chunk_min_tokens,
            overlap_pct=self.chunk_overlap_pct,
            encoding=self.tokenizer_encoding,
        )

    def credentials(self, provider: str) -> ProviderCredentials:
        """Ключ и адрес для названного провайдера.

        ProxyAPI отличается от OpenAI только адресом, поэтому оба обслуживаются
        одним адаптером, а различие живёт здесь, в конфигурации.
        """
        fields = PROVIDER_FIELDS.get(provider.lower())
        if fields is None:
            supported = ", ".join(sorted(PROVIDER_FIELDS))
            raise ValueError(f"Неизвестный провайдер {provider!r}. Поддерживаются: {supported}")

        key_field, url_field = fields
        api_key: str | None = getattr(self, key_field)
        if not api_key:
            raise MissingCredentialsError(provider=provider, env_var=key_field.upper())

        return ProviderCredentials(api_key=api_key, base_url=getattr(self, url_field))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Настройки процесса (создаются один раз)."""
    return Settings()
