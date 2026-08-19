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


class IndexMismatchError(AssistantError):
    """Индекс собран с несовместимыми настройками.

    Главный случай — смена модели эмбеддингов. Поиск при этом не падает,
    а молча возвращает мусор: векторы из разных пространств сравнивать нельзя.
    Поэтому расхождение должно быть именно отказом, а не предупреждением.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"{reason}\nПересоберите индекс: zassist index build --rebuild")
