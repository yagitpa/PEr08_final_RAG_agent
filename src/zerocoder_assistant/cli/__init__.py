"""Пакет команд терминального интерфейса.

Наружу отдаётся только корневая группа: на неё ссылается entry point `zassist`
в pyproject.toml, и она же — единственный поддерживаемый способ звать команды.
"""

from zerocoder_assistant.cli.main import main

__all__ = ["main"]
