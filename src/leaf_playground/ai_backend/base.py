from abc import abstractmethod
from typing import List

from pydantic import BaseModel

from .._config import _Config, _Configurable


class AIBackendConfig(_Config):
    pass


class AIBackend(_Configurable):
    config_obj = AIBackendConfig
    config: config_obj

    @classmethod
    def from_config(cls, config: config_obj) -> "AIBackend":
        return cls(config=config)


__all__ = [
    "AIBackendConfig",
    "AIBackend"
]
