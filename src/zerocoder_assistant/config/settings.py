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

from pydantic import BaseModel, Field, SecretStr, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from zerocoder_assistant.errors import (
    ConfigurationError,
    MissingCredentialsError,
    UnknownProviderError,
)

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
    """Разрешённые доступы к одному провайдеру.

    Ключ — `SecretStr`, а не строка. Разница видна ровно тогда, когда что-то
    пошло не так: настройки и учётные данные попадают в отладочный вывод, в
    текст исключения, в лог по `%r`, — и обычная строка утекает туда целиком.
    `SecretStr` печатается как `**********`, а настоящее значение достаётся
    единственным явным вызовом `get_secret_value()`, который легко найти
    поиском.
    """

    model_config = {"frozen": True}

    api_key: SecretStr
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

    openai_api_key: SecretStr | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    proxyapi_api_key: SecretStr | None = None
    proxyapi_base_url: str = "https://api.proxyapi.ru/openai/v1"

    llm_model: str = "gpt-4o-mini"
    embed_model: str = "text-embedding-3-small"
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    # --- Пути ---------------------------------------------------------------
    notes_dir: Path = Path("./data/notes")
    # RAW_DIR здесь нет намеренно. Конвейер авторства (P1) спроектирован, но не
    # реализован, и сырьё складывать некуда. Настройка, которая ничего не
    # меняет, хуже её отсутствия: её выставляют и ждут последствий. Вернётся
    # вместе с командой `notes`.
    chroma_dir: Path = Path("./storage/chroma")
    cache_db: Path = Path("./storage/cache.db")
    prompts_dir: Path = Path("./prompts")

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
    # Подобран прогоном golden set (51 вопрос, `zassist eval threshold`).
    #
    # Первая калибровка на 16 вопросах давала зазор +0.055 и обещала, что порог
    # умеет отказывать. Это оказалось верно только для вопросов, далёких от
    # корпуса целиком, — других в той калибровке и не было. На смежных темах,
    # которых в конспектах нет («поднять Redis для кэша», «раздать роли агентам
    # в CrewAI»), фрагменты находятся со сходством до 0.588 — ВЫШЕ, чем минимум
    # у вопросов, ответ на которые есть (0.325). Классы перекрываются, зазор
    # отрицательный (−0.263), и никакой порог их не разделит: перебор
    # показывает, что довести отказы до 10 из 11 можно только уронив recall@5
    # с 88% до 40%.
    #
    # Отсюда роль порога: он чистит контекст и отсекает заведомо чужие вопросы
    # (3 из 3 `out_of_domain`), но отказ на смежной теме — работа модели (ветка
    # C системного промпта, 8 из 8 вопросов, которые до неё дошли; на остальных
    # трёх модель не вызывалась вовсе). Поэтому значение выбрано у нижней
    # границы: 0.30 даёт recall@5 88% и ни одного ложного отказа НА УРОВНЕ
    # ПОРОГА. Сама модель при этом отказывается на 7 отвечаемых вопросах из
    # 40 — но это её решение, а не отсечение: фрагменты порог пропустил.
    relevance_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    # Строго больше нуля: при нуле похожими считаются ЛЮБЫЕ два фрагмента
    # (`jaccard >= 0` истинно всегда) и от выдачи остаётся один. Значение
    # 0.8 отсекает дословный повтор и заведомо не трогает перекрытие
    # чанков — по замерам на корпусе соседние чанки одной секции дают
    # сходство около 0.09, а максимум по всем парам выдачи — 0.200.
    dedup_threshold: float = Field(default=0.8, gt=0.0, le=1.0)
    max_context_tokens: int = Field(default=3000, ge=200)
    max_answer_tokens: int = Field(default=1000, ge=100, le=16000)
    history_pairs: int = Field(default=5, ge=0, le=50)
    #: Сколько предыдущих вопросов подмешивать в поиск, когда текущий выглядит
    #: уточняющим. 0 — не подмешивать: уточнения перестанут находиться в базе,
    #: зато поисковый запрос всегда будет ровно тем, что спросил студент.
    follow_up_lookback: int = Field(default=1, ge=0, le=5)
    #: Имя файла системного промпта в `prompts_dir`. Вынесено в настройки, чтобы
    #: сравнить две редакции промпта можно было запуском, а не правкой кода.
    #: Содержимое файла хешируется и входит в ключ кэша L3, поэтому подмена
    #: промпта не отдаёт ответы, посчитанные по прежней редакции.
    answer_prompt_file: str = "rag_answer_v2.md"

    # --- Оценка через RAGAS (этап 8) ----------------------------------------
    #: Интерпретатор ОТДЕЛЬНОГО окружения, в котором стоит RAGAS.
    #:
    #: Не «путь к пакету» и не флаг «включить»: RAGAS невозможно поставить рядом
    #: с ядром. Он тянет langchain-openai, который держит openai на 2.x, а ядро
    #: работает на 3.x — проверено установкой, клиент откатывается с 3.3.0 до
    #: 2.54.0. Поэтому окружения два, а связаны они файлом на диске.
    #:
    #: None означает «оценка через RAGAS не настроена». Это не ошибка: команда
    #: скажет, чего не хватает, и завершится успехом.
    ragas_python: Path | None = None

    log_level: str = "INFO"

    @model_validator(mode="after")
    def _resolve_paths(self) -> Settings:
        """Относительные пути раскрываются от корня проекта, а не от cwd.

        Иначе поведение команд зависит от того, из какой папки их запустили.
        """
        for field in ("notes_dir", "chroma_dir", "cache_db", "prompts_dir"):
            value: Path = getattr(self, field)
            if not value.is_absolute():
                object.__setattr__(self, field, (PROJECT_ROOT / value).resolve())
        # Необязательный путь раскрывается по тем же правилам, но только если
        # задан: None здесь — законное значение «оценка не настроена».
        if self.ragas_python is not None and not self.ragas_python.is_absolute():
            object.__setattr__(self, "ragas_python", (PROJECT_ROOT / self.ragas_python).resolve())
        return self

    @property
    def answer_prompt_path(self) -> Path:
        """Полный путь к файлу системного промпта генерации."""
        return self.prompts_dir / self.answer_prompt_file

    @property
    def chunking(self) -> ChunkingConfig:
        """Срез настроек, который нужен препроцессингу.

        Противоречие между CHUNK_* всплывает только здесь: сами по себе числа
        допустимы, несовместимы они попарно. Раньше это выходило наружу
        трассировкой pydantic прямо из `index build`.
        """
        try:
            return ChunkingConfig(
                target_tokens=self.chunk_target_tokens,
                max_tokens=self.chunk_max_tokens,
                min_tokens=self.chunk_min_tokens,
                overlap_pct=self.chunk_overlap_pct,
                encoding=self.tokenizer_encoding,
            )
        except ValidationError as exc:
            raise ConfigurationError.from_validation(
                "Параметры чанкинга несовместимы", exc
            ) from exc

    def credentials(self, provider: str) -> ProviderCredentials:
        """Ключ и адрес для названного провайдера.

        ProxyAPI отличается от OpenAI только адресом, поэтому оба обслуживаются
        одним адаптером, а различие живёт здесь, в конфигурации.
        """
        fields = PROVIDER_FIELDS.get(provider.lower())
        if fields is None:
            raise UnknownProviderError(provider, tuple(sorted(PROVIDER_FIELDS)))

        key_field, url_field = fields
        api_key: SecretStr | None = getattr(self, key_field)
        # Пустая строка в .env — самая частая форма «ключ забыли вписать»,
        # и для SecretStr она непустой объект: проверять надо содержимое.
        if api_key is None or not api_key.get_secret_value():
            raise MissingCredentialsError(provider=provider, env_var=key_field.upper())

        return ProviderCredentials(api_key=api_key, base_url=getattr(self, url_field))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Настройки процесса (создаются один раз).

    Единственная точка, где настройки читаются из окружения, — и потому
    единственное место, где ошибку в .env ещё можно объяснить человеку.
    Прямой вызов `Settings()` (в тестах) исключение pydantic не прячет.
    """
    try:
        return Settings()
    except ValidationError as exc:
        raise ConfigurationError.from_validation("Настройки в .env недопустимы", exc) from exc
