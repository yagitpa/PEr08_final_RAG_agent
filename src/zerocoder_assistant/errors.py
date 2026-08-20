"""
Исключения приложения.

Собраны в одном месте, чтобы CLI мог отличить предсказуемый отказ («не задан
ключ», «индекс собран другой моделью») от настоящей поломки и показать первое
внятным сообщением, а не трассировкой стека.
"""

from __future__ import annotations


class AssistantError(Exception):
    """Базовая ошибка приложения: ожидаемая ситуация, а не сбой."""


class MissingCredentialsError(AssistantError):
    """Для выбранного провайдера не задан ключ доступа."""

    def __init__(self, provider: str, env_var: str) -> None:
        self.provider = provider
        self.env_var = env_var
        super().__init__(
            f"Не задан ключ для провайдера {provider!r}. "
            f"Укажите {env_var} в .env (образец — в .env.example)."
        )


class ConfigurationError(AssistantError):
    """Значения в .env противоречат друг другу или выходят из допустимых границ.

    Проверку делает pydantic, и делает хорошо, но его `ValidationError` — это
    трассировка на пол-экрана с внутренними путями модели. Человеку, который
    поправил одну строку в .env, нужно знать, какую именно строку он поправил
    неудачно, поэтому текст пересобирается в список «поле: что не так».
    """

    def __init__(self, what: str, problems: tuple[str, ...]) -> None:
        self.problems = problems
        listed = "\n".join(f"  {problem}" for problem in problems)
        super().__init__(f"{what}:\n{listed}\nПравьте .env (образец — в .env.example).")

    @classmethod
    def from_validation(cls, what: str, error: object) -> ConfigurationError:
        """Собрать из pydantic-овского ValidationError.

        Имя поля печатается как есть, без приведения к виду переменной
        окружения. Соблазн был: `top_k` в .env пишется `TOP_K`. Но у чанкинга
        имена не совпадают — поле `min_tokens` задаётся через
        `CHUNK_MIN_TOKENS`, — и «исправленное» имя отправило бы искать
        несуществующую строку. Проверки уровня модели `loc` не имеют вовсе;
        их сообщение и так называет оба поля, поэтому пустой префикс опущен.
        """
        problems = []
        for item in getattr(error, "errors", lambda: [])():
            where = ".".join(str(part) for part in item.get("loc", ()))
            message = str(item.get("msg", "недопустимое значение"))
            problems.append(f"{where}: {message}" if where else message)
        return cls(what, tuple(problems) or (str(error),))


class UnknownProviderError(AssistantError):
    """Провайдера с таким именем в проекте нет.

    Опечатка в `LLM_PROVIDER` — самая частая ошибка настройки, и раньше она
    прилетала голым `ValueError` мимо разбора ошибок в CLI: команда падала
    трассировкой вместо строчки «поддерживаются openai, proxyapi».
    """

    def __init__(self, provider: str, supported: tuple[str, ...]) -> None:
        self.provider = provider
        self.supported = supported
        super().__init__(
            f"Неизвестный провайдер {provider!r}. Поддерживаются: {', '.join(supported)}."
        )


class ProviderRequestError(AssistantError):
    """Провайдер не ответил: сеть, тайм-аут, отказ API, исчерпанная квота.

    Отдельный тип нужен ради диалога. Повторы при сетевых сбоях делает сам SDK,
    и если он всё-таки сдался, то это внешнее обстоятельство, а не поломка
    программы. Пропущенное наружу, оно роняло `ask --repl` вместе со всей
    памятью сессии: студент терял разговор из-за одного разорванного
    соединения. Обёрнутое — печатается строкой, и следующий вопрос можно
    задать сразу.
    """

    def __init__(self, what: str, reason: object) -> None:
        self.reason = reason
        super().__init__(f"{what}: {reason}")


class IndexMismatchError(AssistantError):
    """Индекс собран с несовместимыми настройками.

    Главный случай — смена модели эмбеддингов. Поиск при этом не падает,
    а молча возвращает мусор: векторы из разных пространств сравнивать нельзя.
    Поэтому расхождение должно быть именно отказом, а не предупреждением.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"{reason}\nПересоберите индекс: zassist index build --rebuild")


class IndexBuildError(AssistantError):
    """Сборку индекса просят сделать так, что она разрушит базу знаний.

    Отдельный отказ нужен там, где разрушение выглядит успехом: сочетание
    полной пересборки с фильтром по урокам очищает коллекцию и наполняет её
    только отобранным, а отчёт показывает `removed=0` — плановых удалений
    действительно не было, содержимое исчезло раньше. Такое дешевле запретить,
    чем объяснять потом по отчёту.
    """
