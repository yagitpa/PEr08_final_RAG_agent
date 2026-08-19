"""Запуск пакета как модуля: `python -m zerocoder_assistant`.

Дублирует console-script `zassist` — нужен, чтобы CLI работал и без установки
пакета в окружение, прямо из исходников.
"""

from zerocoder_assistant.cli import main

if __name__ == "__main__":
    main()
