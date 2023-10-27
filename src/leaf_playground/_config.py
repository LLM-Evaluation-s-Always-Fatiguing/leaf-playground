import json
from abc import abstractmethod

from pydantic import BaseModel


class _Config(BaseModel):

    def save(self, file_path: str):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(by_alias=True), f, indent=2)

    @classmethod
    def load(cls, file_path: str):
        kwargs = json.load(open(file_path, "r", encoding="utf-8"))
        return cls(**kwargs)


class _Configurable:
    _config_type = _Config

    def __init__(self, config: "_Configurable._config_type"):
        self.config = config

    def __init_subclass__(cls, **kwargs):
        if cls._config_type == _Config or not issubclass(cls._config_type, _Config):
            raise ValueError(
                "'_config_type' must be _Config's sub-class and can't be _Config itself, "
                "which means, when design a _Configurable class, you must also design a "
                "_Config class and override '_config_type' of the new _Configurable class "
                "with the new _Config class."
            )

    def to_dict(self, **kwargs) -> dict:
        return self.config.model_dump(**kwargs)

    def to_json(self, **kwargs) -> str:
        return self.config.model_dump_json(**kwargs)

    def save_config(self, file_path: str):
        self.config.save(file_path)

    @classmethod
    @abstractmethod
    def from_config(cls, config: _Config) -> "_Configurable":
        raise NotImplementedError()

    @classmethod
    def from_file(cls, file_path: str) -> "_Configurable":
        config = cls._config_type.load(file_path)
        return cls.from_config(config)
