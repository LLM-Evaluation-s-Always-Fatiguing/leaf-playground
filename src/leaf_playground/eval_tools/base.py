from typing import List

from .._config import _Config, _Configurable


class EvalToolConfig(_Config):
    pass


class EvalTool(_Configurable):
    config_cls = EvalToolConfig
    config: config_cls
    activated_metrics: List[str]

    def set_activated_metrics(self, metrics: List[str]):
        self.activated_metrics = metrics

    @classmethod
    def from_config(cls, config: config_cls) -> "EvalTool":
        return cls(config=config)


__all__ = [
    "EvalToolConfig",
    "EvalTool"
]
