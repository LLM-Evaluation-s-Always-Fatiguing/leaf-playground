from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..data.profile import Role, ROLE_TYPES
from ..data.environment import EnvironmentVariable, ENVIRONMENT_VARIABLE_TYPES


@dataclass
class RoleDefinition:
    name: str
    description: str
    type: str
    role_num: int
    template: str
    template_fields: List[str]

    def __post_init__(self):
        if self.type not in ROLE_TYPES:
            types = list(ROLE_TYPES.keys())
            raise ValueError(f"Types of Role are {types}, bug get {self.type}")


@dataclass
class RoleSchema:
    num_participants: int
    definitions: List[RoleDefinition]

    def __post_init__(self):
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
            assert ROLE_TYPES[definition.type].__name__ == role.__class__.__name__

    def get_definition(self, name: str) -> RoleDefinition:
        name2definitions = {definition.name: definition for definition in self.definitions}
        return name2definitions[name]


@dataclass
class EnvironmentVariableDefinition:
    name: str
    description: str
    type: str
    template: str
    template_fields: List[str]

    def __post_init__(self):
        if self.type not in ENVIRONMENT_VARIABLE_TYPES:
            types = list(ENVIRONMENT_VARIABLE_TYPES.keys())
            raise ValueError(f"Types of EnvironmentVariable are {types}, bug get {self.type}")


@dataclass
class EnvironmentSchema:
    definitions: List[EnvironmentVariableDefinition]

    def valid(self, env_vars: List[EnvironmentVariable]):
        assert len(self.definitions) == len(env_vars)

        name2definitions = {definition.name: definition for definition in self.definitions}
        name2variables = {variable.name: variable for variable in env_vars}

        assert set(name2definitions.keys()) == set(name2variables.keys())
        for name, variable in name2variables.items():
            definition = name2definitions[name]
            assert definition.description == variable.description
            assert ENVIRONMENT_VARIABLE_TYPES[definition.type].__name__ == variable.__class__.__name__

    def get_definition(self, name: str) -> EnvironmentVariableDefinition:
        name2definitions = {definition.name: definition for definition in self.definitions}
        return name2definitions[name]


@dataclass
class SceneSchema:
    name: str
    description: str
    role_schema: RoleSchema
    environment_schema: EnvironmentSchema

    def valid(self, roles: List[Role], env_vars: List[EnvironmentVariable]):
        self.role_schema.valid(roles)
        self.environment_schema.valid(env_vars)


@dataclass
class SceneLogBody:
    turn_id: int
    turn_step_id: int
    global_step_id: int
    event: Optional[dict] = field(default=None)
    message: Optional[dict] = field(default=None)

    def format(
        self,
        template: str,
        fields: List[str]
    ) -> str:
        data = {f: self.__getattribute__(f) for f in fields}
        return template.format(**data)


class Scene:
    schema: SceneSchema

    def __int__(self, roles: List[Role], env_vars: List[EnvironmentVariable]):
        self.schema.valid(roles=roles, env_vars=env_vars)

        self.roles = {role.name: role for role in roles}
        self.environment = {env_var.name: env_var for env_var in env_vars}
        self._logs: List[SceneLogBody] = []

    @property
    def name(self):
        return self.schema.name

    @property
    def description(self):
        return self.schema.description

    def get_role(self, name: str) -> Role:
        return self.roles[name]

    def get_env_var(self, name: str) -> EnvironmentVariable:
        return self.environment[name]

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
        fields = self.schema.role_schema.get_definition(name).template_fields
        displayer(env_var.format(template, fields))
        if return_obj:
            return env_var

    def display_logs(
        self,
        template: str,
        fields: List[str],
        displayer: Callable[[str], None] = print,
        return_obj: bool = False
    ) -> Optional[List[SceneLogBody]]:
        for log in self._logs:
            displayer(log.format(template, fields))
        if return_obj:
            return self._logs

    def append_log(self, log: SceneLogBody):
        self._logs.append(log)

    @classmethod
    def display_schema(cls, displayer: Callable[[str], None] = print) -> None:
        displayer(repr(cls.schema))

    @abstractmethod
    def is_terminal(self) -> bool:
        return True

    @classmethod
    def build_sub_cls(
        cls,
        config: "SceneSubClsBuildingConfig"
    ):
        kwds = {"schema": config.schema, "is_terminal": config.is_terminal_impl}
        if config.init_impl is not None:
            kwds["__init__"] = config.init_impl
        if config.new_attrs is not None:
            kwds.update(config.new_attrs)
        if config.new_methods is not None:
            kwds.update(config.new_methods)
        return type(config.cls_name, (Scene,), kwds)


@dataclass
class SceneSubClsBuildingConfig:
    cls_name: str
    schema: SceneSchema
    is_terminal_impl: Callable[[], bool]
    init_impl: Optional[Callable[[*Any, List[Role], List[EnvironmentVariable]], None]] = None
    new_attrs: Optional[Dict[str, Any]] = None
    new_methods: Optional[Dict[str, Callable]] = None
