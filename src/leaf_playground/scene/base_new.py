import asyncio
import json
import random
from abc import abstractmethod
from enum import Enum
from functools import partial
from itertools import chain
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Type, Union

from pydantic import create_model, BaseModel, Field, FilePath
from pydantic.fields import FieldInfo

from .._config import _Config, _Configurable
from ..agent_new.base import Agent
from ..data.environment import EnvironmentVariable
from ..data.log_body import LogBody
from ..data.message import MessagePool
from ..data.profile import Role
from ..utils.import_util import dynamically_import_obj, DynamicObject


def _get_model_fields(cls: Type[BaseModel]):
    return {f_name: (f_info.annotation, f_info) for f_name, f_info in cls.model_fields.items()}


def _partially_instantiate_model(cls: Type[BaseModel], __model_name: str, **fields_value):
    fields = _get_model_fields(cls)

    for f_name, f_value in fields_value.items():
        f_info = fields[f_name][1]
        f_annotation = Literal[f_value]
        f_info.annotation = f_annotation
        f_info.default = f_value
        fields[f_name] = (f_annotation, f_info)

    return create_model(
        __model_name=__model_name,
        __base__=cls,
        **fields
    )


class RoleDefinition(BaseModel):
    name: str = Field(default=...)
    description: str = Field(default=...)
    num_agents: int = Field(default=-1, gt=-1)
    type: DynamicObject = Field(default=DynamicObject(obj="Role", module="leaf_playground.data.profile"))

    def model_post_init(self, __context: Any) -> None:
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


class RolesConfig(BaseModel):
    roles_agent_num: Dict[str, int] = Field(default=..., frozen=True)

    def model_post_init(self, __context: Any) -> None:
        for k, v in self.roles_agent_num.items():
            if v < -1 or v == 0:
                raise ValueError(f"num_agents of {k} should be -1 or positive, got {v}")

    @classmethod
    def create_subclass(cls, definitions: List[RoleDefinition], **kwargs):
        fields = _get_model_fields(cls)
        for definition in definitions:
            f_name = definition.name
            f_annotation = _partially_instantiate_model(
                cls=dynamically_import_obj(definition.type),
                __model_name=definition.name,
                name=definition.name,
                description=definition.description
            )
            f_default = f_annotation()
            fields[f_name] = (f_annotation, FieldInfo(default=f_default, annotation=f_annotation))

        roles_agent_num = {definition.name: definition.num_agents for definition in definitions}
        fields["roles_agent_num"][1].default = roles_agent_num
        return create_model(__model_name="Roles", __base__=cls, **fields)


class EnvVarsConfig(BaseModel):

    @classmethod
    def create_subclass(cls, definitions: List[EnvVarDefinition], **kwargs):
        fields = _get_model_fields(cls)
        for definition in definitions:
            f_name = definition.name
            f_annotation = _partially_instantiate_model(
                cls=dynamically_import_obj(definition.type),
                __model_name=definition.name,
                name=definition.name,
                description=definition.description
            )
            fields[f_name] = (f_annotation, FieldInfo(annotation=f_annotation))
        return create_model(__model_name="Environments", __base__=cls, **fields)


class SceneInfoConfig(_Config):
    name: str = Field(default=...)
    description: str = Field(default=...)
    roles: RolesConfig = Field(default=...)
    environments: EnvVarsConfig = Field(default=...)

    @classmethod
    def create_subclass(
        cls,
        name: str,
        description: str,
        role_definitions: List[RoleDefinition],
        env_definitions: List[EnvVarDefinition],
        **kwargs
    ):
        fields = _get_model_fields(cls)

        roles_cls = RolesConfig.create_subclass(role_definitions)
        envs_cls = EnvVarsConfig.create_subclass(env_definitions)

        fields.update(
            **{
                "name": (Literal[name], FieldInfo(default=name, annotation=Literal[name])),
                "description": (Literal[description], FieldInfo(default=description, annotation=Literal[description])),
                "roles": (roles_cls, FieldInfo(default=roles_cls(), annotation=roles_cls)),
                "environments": (envs_cls, FieldInfo(annotation=envs_cls)),
            }
        )
        return create_model(__model_name="SceneInfo", __base__=cls, **fields)


class SceneInfo(_Configurable):
    config_obj = SceneInfoConfig
    config: SceneInfoConfig

    def __init__(self, config: config_obj):
        super().__init__(config=config)

        self.name = self.config.name
        self.description = self.config.description
        self.environments = {
            env_name: getattr(self.config.environments, env_name)
            for env_name in self.config.environments.model_fields.keys()
        }
        self.roles = {
            role_name: getattr(self.config.roles, role_name)
            for role_name in self.config.roles.model_fields.keys()
        }
        self.roles_agent_num = self.config.roles.roles_agent_num

    @classmethod
    def from_config(cls, config: config_obj) -> "SceneInfo":
        return cls(config=config)

    def get_role(self, name: str) -> Role:
        return self.roles[name]

    def get_env_var(self, name: str) -> EnvironmentVariable:
        return self.environments[name]


class AgentObjConfig(_Config):
    agent_config_data: Optional[dict] = Field(default=None)
    agent_config_file: Optional[FilePath] = Field(default=None)
    agent_obj: DynamicObject = Field(default=...)

    def model_post_init(self, __context: Any) -> None:
        if not self.agent_config_data and not self.agent_config_file:
            raise ValueError("at least agent_config_data or agent_config_file should be specified")

    def create_instance(self) -> "Agent":
        if not self.agent_config_data:
            with open(self.agent_config_file.as_posix(), "r", encoding="utf-8") as f:
                self.agent_config_data = json.load(f)
        obj = dynamically_import_obj(self.agent_obj)
        config = obj.config_obj(**self.agent_config_data)
        return obj.from_config(config=config)


class AgentsObjConfig(_Config):
    agents: List[AgentObjConfig] = Field(default=...)

    def create_instances(self) -> List["Agent"]:
        return [agent.create_instance() for agent in self.agents]


class SceneInfoObjConfig(_Config):
    scene_info_config_data: Optional[dict] = Field(default=None)
    scene_info_config_file: Optional[FilePath] = Field(default=None)
    scene_info_obj: DynamicObject = Field(default=...)

    def model_post_init(self, __context: Any) -> None:
        if not self.scene_info_config_data and not self.scene_info_config_file:
            raise ValueError("at least scene_info_config_data or scene_info_config_file should be specified")

    def create_instance(self) -> "SceneInfo":
        if not self.scene_info_config_data:
            with open(self.scene_info_config_file.as_posix(), "r", encoding="utf-8") as f:
                self.scene_info_config_data = json.load(f)
        obj = dynamically_import_obj(self.scene_info_obj)
        config = obj.config_obj(**self.scene_info_config_data)
        return obj.from_config(config=config)


class SceneState(Enum):
    PENDING = 0
    RUNNING = 1
    FINISHED = 2


class SceneConfig(_Config):
    scene_info: SceneInfoObjConfig = Field(default=...)
    agents: AgentsObjConfig = Field(default=...)


class Scene(_Configurable):
    config_obj = SceneConfig
    config: SceneConfig

    def __init__(self, config: config_obj):
        super().__init__(config=config)

        self.scene_info: SceneInfo = self.config.scene_info.create_instance()
        self.agents: List[Agent] = self.config.agents.create_instances()
        self._valid_agent_num()
        self._assign_roles()
        assert all(bool(agent.profile.role) for agent in self.agents), "Not all agents are assigned roles."

        self._logs: List[LogBody] = []
        self._message_pool: MessagePool = MessagePool()
        self._state = SceneState.PENDING

    def _valid_agent_num(self):
        roles_agent_num = self.scene_info.roles_agent_num
        if any(v == -1 for v in roles_agent_num.values()):
            num_min_agents = (
                sum(v for v in roles_agent_num.values() if v != -1) +
                len([v for v in roles_agent_num.values() if v == -1])
            )
            if num_min_agents > len(self.agents):
                raise ValueError(f"required at least {num_min_agents} agents, got {len(self.agents)}")
        else:
            num_total_agents = sum(roles_agent_num.values())
            if num_total_agents != len(self.agents):
                raise ValueError(f"required total {num_total_agents} agents, got {len(self.agents)}")

    @property
    def state(self):
        return self._state

    def _assign_roles(self) -> None:
        agent_indices = list(range(len(self.agents)))
        random.shuffle(agent_indices)
        static_roles = list(
            chain(*[[role] * num for role, num in self.scene_info.roles_agent_num.items() if num != -1])
        )
        for idx, agent_idx in enumerate(agent_indices[:len(static_roles)]):
            self.agents[agent_idx].profile.role = self.scene_info.get_role(static_roles[idx])
        dynamic_roles = [role for role, num in self.scene_info.roles_agent_num.items() if num == -1]
        for agent_idx in agent_indices[len(static_roles):]:
            self.agents[agent_idx].profile.role = self.scene_info.get_role(random.choice(dynamic_roles))

    @abstractmethod
    async def _run(self):
        raise NotImplementedError()

    async def _stream_logs(
        self,
        template: str = "[{time}] :: EVENT - {event} :: MESSAGE - {message}",
        fields: Set[str] = {"time", "event", "message"},
        displayer: Callable[[str], None] = partial(print, flush=True)
    ):
        cur = 0
        while True:
            if cur >= len(self._logs):
                await asyncio.sleep(0.001)
            else:
                displayer(self._logs[cur].format(template, fields))
                cur += 1
                await asyncio.sleep(0.1)
            if self._state == SceneState.FINISHED:
                for log in self._logs[cur:]:
                    displayer(log.format(template, fields))
                    await asyncio.sleep(0.1)
                break

    def export_logs(self, file: str):
        with open(file, "w", encoding="utf-8") as f:
            for log in self._logs:
                f.write(log.model_dump_json(by_alias=True) + "\n")

    def start(self):
        async def _run_wrapper():
            self._state = SceneState.RUNNING
            await self._run()
            self._state = SceneState.FINISHED

        async def _start():
            await asyncio.gather(_run_wrapper(), self._stream_logs())

        asyncio.get_event_loop().run_until_complete(_start())

    @classmethod
    def from_config(cls, config: config_obj) -> "Scene":
        return cls(config=config)
