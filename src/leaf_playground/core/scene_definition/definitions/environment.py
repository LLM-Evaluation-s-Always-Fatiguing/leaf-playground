from sys import _getframe
from typing import Any, Type

from pydantic import create_model, field_serializer, BaseModel, Field, PrivateAttr

from ...._config import _Config
from ....data.environment import EnvironmentVariable


class EnvVarDefinition(BaseModel):
    name: str = Field(default=...)
    description: str = Field(default=...)
    env_var_cls: Type[EnvironmentVariable] = Field(default=EnvironmentVariable)

    def model_post_init(self, __context: Any) -> None:
        if not issubclass(self.env_var_cls, EnvironmentVariable):
            raise ValueError(f"env_var_cls [{self.env_var_cls.__name__}] should be a subclass of EnvironmentVariable")

    @field_serializer("env_var_cls")
    def serialize_type(self, env_var_cls: Type[EnvironmentVariable], _info) -> str:
        return env_var_cls.__name__


class EnvVarConfig(_Config):
    _env_var_definition: EnvVarDefinition = PrivateAttr(default=None)

    def __init_subclass__(cls, _env_var_definition: EnvVarDefinition, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._env_var_definition = _env_var_definition

    @property
    def env_var_definition(self) -> EnvVarDefinition:
        return self._env_var_definition

    @classmethod
    def create_config_model(cls, env_var_definition: EnvVarDefinition) -> Type["EnvironmentVariable"]:
        model_name = "".join([each.capitalize() for each in env_var_definition.name.split("_")]) + cls.__name__
        module = _getframe(1).f_globals["__name__"]
        cls_kwargs = {"_env_var_definition": env_var_definition}
        fields = {
            f_name: (f_info.annotation, f_info)
            for f_name, f_info in env_var_definition.env_var_cls.model_fields.items()
            if f_name not in ["name", "description"] and not f_info.exclude
        }
        return create_model(
            __model_name=model_name, __module__=module, __base__=cls, __cls_kwargs__=cls_kwargs, **fields
        )

    def initiate_env_var(self) -> EnvironmentVariable:
        return self.env_var_definition.env_var_cls(
            name=self.env_var_definition.name,
            description=self.env_var_definition.description,
            **self.model_dump(mode="json", by_alias=True),
        )


__all__ = ["EnvVarDefinition", "EnvVarConfig"]
