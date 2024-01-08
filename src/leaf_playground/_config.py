import inspect
import json
from typing import Literal, Type

from pydantic import create_model, BaseModel


class _Config(BaseModel):
    def save(self, file_path: str):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(mode="json", by_alias=True), f, indent=2)

    @classmethod
    def load(cls, file_path: str):
        kwargs = json.load(open(file_path, "r", encoding="utf-8"))
        return cls(**kwargs)

    @classmethod
    def get_json_schema(cls, by_alias: bool = False, mode: Literal["validation", "serialization"] = "validation"):
        def _create_new_annotation(annotation: Type):
            if not inspect.isclass(annotation) or not issubclass(annotation, BaseModel):
                return annotation
            anno_fields = {f_name: f_info for f_name, f_info in annotation.model_fields.items() if not f_info.exclude}
            for f_name, f_info in anno_fields.items():
                f_info.annotation = _create_new_annotation(f_info.annotation)
                anno_fields[f_name] = (f_info.annotation, f_info)
            return create_model(
                __model_name=f"{annotation.__name__}", __module__="pydantic_temp", __base__=BaseModel, **anno_fields
            )

        return _create_new_annotation(cls).model_json_schema(by_alias=by_alias, mode=mode)


class _Configurable:
    config_cls = _Config

    def __init__(self, config: config_cls):
        if not isinstance(config, self.config_cls):
            raise TypeError(
                f"required '{self.config_cls.__name__}' type config, but get '{config.__class__.__name__}'"
            )
        self.config = config

    def __init_subclass__(cls, **kwargs):
        if cls.config_cls == _Config or not issubclass(cls.config_cls, _Config):
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
    def from_config(cls, config: _Config) -> "_Configurable":
        raise NotImplementedError(f"{cls.__name__} not implements from_config method")

    @classmethod
    def from_config_file(cls, file_path: str) -> "_Configurable":
        config = cls.config_cls.load(file_path)
        return cls.from_config(config)
