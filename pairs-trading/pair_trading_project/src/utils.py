from typing import Dict

from yaml import safe_load


def load_config(config_path: str) -> Dict:
    """Загружает yaml конфиг в виде python словаря.

    Args:
        config_path: Путь до конфига.

    Returns:
        Словарь с параметрами конфига.
    """
    with open(config_path) as file:
        config = safe_load(file)
    return config
