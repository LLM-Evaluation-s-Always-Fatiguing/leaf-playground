from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Type

from pydantic import create_model, BaseModel, Field
from pydantic.fields import FieldInfo

from .._config import _Config, _Configurable
from ..data.base import Data
from ..data.environment import EnvironmentVariable
from ..data.profile import Role
from ..utils.import_util import dynamically_import_obj, DynamicObject


def _get_model_fields(cls: Type[BaseModel]):
    return {f_name: (f_info.annotation, f_info) for f_name, f_info in cls.model_fields.items()}


def _partially_instantiate_model(cls: Type[BaseModel], __model_name: str, **fields_value):
    fields = _get_model_fields(cls)

    for f_name, f_value in fields_value.items():
        f_info = fields[f_name][1]
        if isinstance(f_value, (str, int, float, bool, list, dict)):
            f_annotation = Literal[f_value]
        else:
            f_annotation = type(f_value)
        f_info.annotation = f_annotation
        f_info.default = f_value
        f_info.frozen = True
        fields[f_name] = (f_annotation, f_info)

    return create_model(
        __model_name=__model_name,
        __base__=cls,
        **fields
    )


class RoleDefinition(BaseModel):
    name: str = Field(default=...)
    description: str = Field(default=...)
    num_agents: int = Field(default=-1, ge=-1)
    is_static: bool = Field(default=False)
    type: DynamicObject = Field(default=DynamicObject(obj="Role", module="leaf_playground.data.profile"))
    agent_type: Optional[DynamicObject] = Field(default=None)

    def model_post_init(self, __context: Any) -> None:
        if self.is_static and self.num_agents != 1:
            raise ValueError(f"num_agents should be 1 when the role is a static role, got {self.num_agents}")
        if self.is_static and not self.agent_type:
            raise ValueError(f"agent_type should be specified when the role is a static role")
        if not issubclass(dynamically_import_obj(self.type), Role):
            raise TypeError(f"type of {self.type.obj} should be a subclass of Role")
        if self.num_agents < -1 or self.num_agents == 0:
            raise ValueError(f"num_agents should be -1 or positive, got {self.num_agents}")


class EnvVarDefinition(BaseModel):
    name: str = Field(default=...)
    description: str = Field(default=...)
    type: DynamicObject = Field(
        default=DynamicObject(obj="EnvironmentVariable", module="leaf_playground.data.environment")
    )

    def model_post_init(self, __context: Any) -> None:
        if not issubclass(dynamically_import_obj(self.type), EnvironmentVariable):
            raise TypeError(f"type of {self.type.obj} should be a subclass of EnvironmentVariable")


class RolesConfigBase(BaseModel):
    def model_post_init(self, __context: Any) -> None:
        fields = _get_model_fields(self.__class__)
        for k, v in self.roles_agent_num.items():
            if k not in fields:
                raise ValueError(f"role of {k} is not defined in RolesConfig")
            if getattr(self, k).is_static and v != 1:
                raise ValueError(f"num_agents of {k} should be 1 when the role is a static role, got {v}")
            if v < -1 or v == 0:
                raise ValueError(f"num_agents of {k} should be -1 or positive, got {v}")

    @classmethod
    def create_subclass(cls, definitions: List[RoleDefinition], **kwargs):
        fields = _get_model_fields(cls)
        name2cls = {}
        for definition in definitions:
            f_name = definition.name
            f_annotation = _partially_instantiate_model(
                cls=dynamically_import_obj(definition.type),
                __model_name=definition.name,
                name=definition.name,
                description=definition.description,
                is_static=definition.is_static,
                agent_type=definition.agent_type
            )
            f_default = f_annotation()
            fields[f_name] = (f_annotation, FieldInfo(default=f_default, annotation=f_annotation))
            name2cls[f_name] = f_annotation

        roles_agent_num = {definition.name: definition.num_agents for definition in definitions}
        fields["roles_agent_num"] = (
            Dict[str, int], FieldInfo(default=roles_agent_num, annotation=Dict[str, int])
        )
        return create_model(__model_name="RolesConfig", __base__=cls, **fields), name2cls


class EnvVarsConfigBase(BaseModel):

    @classmethod
    def create_subclass(cls, definitions: List[EnvVarDefinition], **kwargs):
        fields = _get_model_fields(cls)
        name2cls = {}
        for definition in definitions:
            f_name = definition.name
            f_annotation = _partially_instantiate_model(
                cls=dynamically_import_obj(definition.type),
                __model_name=definition.name,
                name=definition.name,
                description=definition.description
            )
            fields[f_name] = (f_annotation, FieldInfo(annotation=f_annotation))
            name2cls[f_name] = f_annotation
        return create_model(__model_name="EnvVarsConfig", __base__=cls, **fields), name2cls


class SceneMetaData(Data):
    name: str = Field(default=...)
    description: str = Field(default=...)
    role_definitions: List[RoleDefinition] = Field(default=...)
    env_definitions: List[EnvVarDefinition] = Field(default=...)

    def get_roles_agent_num(self) -> Dict[str, int]:
        return {
            role.name: role.num_agents for role in self.role_definitions
        }


class SceneInfoConfigBase(_Config):
    name: str = Field(default=...)
    description: str = Field(default=...)
    roles: RolesConfigBase = Field(default=...)
    environments: EnvVarsConfigBase = Field(default=...)

    @classmethod
    def create_subclass(
        cls,
        metadata: SceneMetaData,
        **kwargs
    ):
        name = metadata.name
        description = metadata.description
        role_definitions = metadata.role_definitions
        env_definitions = metadata.env_definitions

        fields = _get_model_fields(cls)

        roles_config_model, roles_cls = RolesConfigBase.create_subclass(role_definitions)
        envs_config_model, envs_cls = EnvVarsConfigBase.create_subclass(env_definitions)

        fields.update(
            **{
                "name": (Literal[name], FieldInfo(default=name, annotation=Literal[name])),
                "description": (Literal[description], FieldInfo(default=description, annotation=Literal[description])),
                "roles": (roles_config_model, FieldInfo(default=roles_config_model(), annotation=roles_config_model)),
                "environments": (envs_config_model, FieldInfo(annotation=envs_config_model)),
            }
        )
        return (
            create_model(__model_name="SceneInfoConfig", __base__=cls, **fields),
            roles_config_model,
            envs_config_model,
            roles_cls,
            envs_cls
        )


class SceneInfo(_Configurable):
    config_obj = SceneInfoConfigBase
    config: config_obj

    def __init__(self, config: config_obj):
        super().__init__(config=config)

        self.name: str = self.config.name
        self.description: str = self.config.description
        self.environments: Dict[str, EnvironmentVariable] = {
            env_name: getattr(self.config.environments, env_name)
            for env_name in self.config.environments.model_fields.keys()
        }
        self.roles: Dict[str, Role] = {
            role_name: getattr(self.config.roles, role_name)
            for role_name in self.config.roles.model_fields.keys() if role_name != "roles_agent_num"
        }
        self.roles_agent_num: Dict[str, int] = self.config.roles.roles_agent_num

    @classmethod
    def from_config(cls, config: config_obj) -> "SceneInfo":
        return cls(config=config)

    def get_role(self, name: str) -> Role:
        return self.roles[name]

    def get_env_var(self, name: str) -> EnvironmentVariable:
        return self.environments[name]


class SceneState(Enum):
    PENDING = 0
    RUNNING = 1
    FINISHED = 2


__all__ = [
    "RoleDefinition",
    "EnvVarDefinition",
    "RolesConfigBase",
    "EnvVarsConfigBase",
    "SceneMetaData",
    "SceneInfoConfigBase",
    "SceneInfo",
    "SceneState"
]
