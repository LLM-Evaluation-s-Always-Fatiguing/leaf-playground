import json
from datetime import datetime
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Set

from pydantic import BaseModel, Field, FilePath

from .._schema import _Schema
from .._config import _Config, _Configurable
from ..data.profile import Role
from ..data.environment import EnvironmentVariable
from ..utils.import_util import dynamically_import_obj, DynamicObject


class RoleDefinition(BaseModel):
    name: str = Field(default=...)
    description: str = Field(default=...)
    type: DynamicObject = Field(default=DynamicObject(obj="Role", module="leaf_playground.data.profile"))
    role_num: int = Field(default=-1, ge=-1)
    template: str = Field(default="RoleDefinition :: name={name}; description={description}")
    template_fields: Set[str] = Field(default={"name", "description"})

    def model_post_init(self, __context: Any) -> None:
        if self.role_num == 0:
            raise ValueError("role_num can't be 0")


class RoleSchema(_Schema):
    num_participants: int = Field(default=-1, ge=-1)
    definitions: List[RoleDefinition] = Field(default=...)

    def model_post_init(self, __context: Any) -> None:
        if self.num_participants == 0:
            raise ValueError("num_participants can't be 0")
        if self.num_participants != -1:
            if any(definition.role_num == -1 for definition in self.definitions):
                raise ValueError(
                    "we can't infer the total number of participants when there are some role_definition's "
                    "'role_num' is -1 but 'num_participants' is a certain positive number"
                )
            if self.num_participants != sum([definition.role_num for definition in self.definitions]):
                raise ValueError("'num_participants' must equal to the sum of all roles' 'role_num'")

    def valid(self, roles: List[Role]):
        assert len(self.definitions) == len(roles)

        name2definitions = {definition.name: definition for definition in self.definitions}
        name2roles = {role.name: role for role in roles}

        assert set(name2definitions.keys()) == set(name2roles.keys())
        for name, role in name2roles.items():
            definition = name2definitions[name]
            assert definition.description == role.description
            assert isinstance(role, dynamically_import_obj(definition.type))

    def get_definition(self, name: str) -> RoleDefinition:
        name2definitions = {definition.name: definition for definition in self.definitions}
        return name2definitions[name]


class EnvironmentVariableDefinition(BaseModel):
    name: str = Field(default=...)
    description: str = Field(default=...)
    type: DynamicObject = Field(
        default=DynamicObject(
            obj="EnvironmentVariable",
            module="leaf_playground.data.environment"
        )
    )
    template: str = Field(default="EnvironmentVariableDefinition :: name={name}; description={description}")
    template_fields: Set[str] = Field(default={"name", "description"})


class EnvironmentSchema(_Schema):
    definitions: List[EnvironmentVariableDefinition] = Field(default=...)

    def valid(self, env_vars: List[EnvironmentVariable]):
        assert len(self.definitions) == len(env_vars)

        name2definitions = {definition.name: definition for definition in self.definitions}
        name2variables = {variable.name: variable for variable in env_vars}

        assert set(name2definitions.keys()) == set(name2variables.keys())
        for name, variable in name2variables.items():
            definition = name2definitions[name]
            assert definition.description == variable.description
            assert isinstance(variable, dynamically_import_obj(definition.type))

    def get_definition(self, name: str) -> EnvironmentVariableDefinition:
        name2definitions = {definition.name: definition for definition in self.definitions}
        return name2definitions[name]


class SceneSchema(_Schema):
    name: str = Field(default=...)
    description: str = Field(default=...)
    role_schema: RoleSchema = Field(default=...)
    environment_schema: EnvironmentSchema = Field(default=...)

    def valid(self, roles: List[Role], env_vars: List[EnvironmentVariable]):
        self.role_schema.valid(roles)
        self.environment_schema.valid(env_vars)

    @property
    def num_participants(self):
        return self.role_schema.num_participants


class RoleConfig(_Config):
    role_data: Optional[dict] = Field(default=None)
    role_file: Optional[FilePath] = Field(default=None, pattern=r".*.json")
    role_obj: DynamicObject = Field(
        default=DynamicObject(obj="Role", module="leaf_playground.data.profile")
    )

    def model_post_init(self, __context: Any) -> None:
        if self.role_data is None and self.role_file is None:
            raise ValueError("at least on of role_data and role_file is specified")

    @property
    def role(self) -> Role:
        if self.role_data is None:
            with open(self.role_file, "r", encoding="utf-8") as f:
                self.role_data = json.load(f)
        return dynamically_import_obj(self.role_obj)(**self.role_data)


class EnvironmentVarConfig(_Config):
    env_var_data: Optional[dict] = Field(default=None)
    env_var_file: Optional[FilePath] = Field(default=None, pattern=r".*.json")
    env_var_obj: DynamicObject = Field(
        default=DynamicObject(obj="EnvironmentVariable", module="leaf_playground.data.environment")
    )

    def model_post_init(self, __context: Any) -> None:
        if self.env_var_data is None and self.env_var_file is None:
            raise ValueError("at least one of env_var_data and env_var_file is specified")

    @property
    def env_var(self) -> EnvironmentVariable:
        if self.env_var_data is None:
            with open(self.env_var_file, "r", encoding="utf-8") as f:
                self.env_var_data = json.load(f)
        return dynamically_import_obj(self.env_var_obj)(**self.env_var_data)


class SceneConfig(_Config):
    role_configs: List[RoleConfig] = Field(default=...)
    env_var_configs: List[EnvironmentVarConfig] = Field(default=...)

    @property
    def roles(self) -> List[Role]:
        return [role_config.role for role_config in self.role_configs]

    @property
    def env_vars(self) -> List[EnvironmentVariable]:
        return [env_var_config.env_var for env_var_config in self.env_var_configs]


class Scene(_Configurable):
    schema: SceneSchema
    config_obj = SceneConfig

    def __init__(self, config: config_obj):
        super().__init__(config=config)

        roles = self.config.roles
        env_vars = self.config.env_vars
        self.schema.valid(roles=roles, env_vars=env_vars)

        self.roles = {role.name: role for role in roles}
        self.environments = {env_var.name: env_var for env_var in env_vars}

        self._init()

    @classmethod
    def from_config(cls, config: SceneConfig) -> "Scene":
        return cls(config=config)

    @property
    def name(self):
        return self.schema.name

    @property
    def description(self):
        return self.schema.description

    def get_role(self, name: str) -> Role:
        return self.roles[name]

    def get_env_var(self, name: str) -> EnvironmentVariable:
        return self.environments[name]

    def display_role(
        self,
        name: str,
        displayer: Callable[[str], None] = print,
        return_obj: bool = True
    ) -> Optional[Role]:
        role = self.get_role(name)
        template = self.schema.role_schema.get_definition(name).template
        fields = self.schema.role_schema.get_definition(name).template_fields
        displayer(role.format(template, fields))
        if return_obj:
            return role

    def display_env_var(
        self,
        name: str,
        displayer: Callable[[str], None] = print,
        return_obj: bool = True
    ) -> Optional[EnvironmentVariable]:
        env_var = self.get_env_var(name)
        template = self.schema.environment_schema.get_definition(name).template
        fields = self.schema.environment_schema.get_definition(name).template_fields
        displayer(env_var.format(template, fields))
        if return_obj:
            return env_var

    @classmethod
    def display_schema(cls, displayer: Callable[[str], None] = print) -> None:
        displayer(cls.schema.model_dump_json(indent=2))

    @abstractmethod
    def is_terminal(self) -> bool:
        return True

    def _init(self) -> None:
        pass

    @classmethod
    def implement_sub_cls(cls, config: "SceneSubClsImplConfig"):
        kwds = {
            "schema": config.scene_schema,
            "config_obj": dynamically_import_obj(config.config_obj),
            "is_terminal": config.is_terminal_impl,
        }
        if config.init_impl is not None:
            kwds["_init"] = config.init_impl
        if config.new_attrs is not None:
            kwds.update(config.new_attrs)
        if config.new_methods is not None:
            kwds.update(config.new_methods)
        return type(config.cls_name, (cls,), kwds)


class SceneSubClsImplConfig(BaseModel):
    cls_name: str = Field(default=...)
    scene_schema: SceneSchema = Field(default=...)
    is_terminal_impl: Callable[[Scene], bool] = Field(default=...)
    config_obj: DynamicObject = Field(
        default=DynamicObject(obj="SceneConfig", module="leaf_playground.scene.base")
    )
    init_impl: Optional[Callable[[Scene], None]] = Field(default=None)
    new_attrs: Optional[Dict[str, Any]] = Field(default=None)
    new_methods: Optional[Dict[str, Callable]] = Field(default=None)
