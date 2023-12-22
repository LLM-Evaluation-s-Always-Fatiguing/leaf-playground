from abc import abstractmethod
from typing import List

from pydantic import BaseModel

from .._config import _Config, _Configurable


class AIBackendConfig(_Config):
    pass


class AIBackend(_Configurable):
    config_cls = AIBackendConfig
    config: config_cls

    @classmethod
    def from_config(cls, config: config_cls) -> "AIBackend":
        return cls(config=config)


__all__ = [
    "AIBackendConfig",
    "AIBackend"
]
