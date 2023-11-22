import asyncio
import inspect
import random
from abc import abstractmethod
from itertools import chain
from os.path import dirname, join
from typing import Any, Callable, List, Type

from fastapi import WebSocket
from pydantic import Field, ValidationError

from .scene_agent import SceneAgent, SceneAgentConfig
from .scene_info import SceneInfo, SceneInfoConfigBase, SceneMetaData, SceneState
from .._config import _Config, _Configurable
from ..data.log_body import LogBody
from ..data.message import MessagePool
from ..data.socket_data import SocketData, SocketDataType
from ..utils.import_util import dynamically_import_obj, find_subclasses, DynamicObject


class SceneAgentObjConfig(_Config):
    agent_config_data: dict = Field(default=...)
    agent_obj: DynamicObject = Field(default=...)

    def model_post_init(self, __context: Any) -> None:
        self.valid(self.agent_config, self.agent_obj)

    def create_instance(self) -> "SceneAgent":
        obj = dynamically_import_obj(self.agent_obj)
        return obj.from_config(config=self.agent_config)

    @staticmethod
    def valid(agent_config: SceneAgentConfig, agent_obj: DynamicObject):
        agent_cls = dynamically_import_obj(agent_obj)
        if not issubclass(agent_cls, SceneAgent):
            raise TypeError(
                f"agent_obj {agent_obj.obj} should be a subclass of SceneAgent, "
                f"but get {agent_cls.__name__}"
            )
        if not isinstance(agent_config, agent_cls.config_obj):
            raise TypeError(
                f"agent_config should be an instance of {agent_cls.config_obj.__name__}, "
                f"but get {agent_config.__class__.__name__}"
            )

    @property
    def agent_config(self) -> SceneAgentConfig:
        agent_cls = dynamically_import_obj(self.agent_obj)
        agent_config = agent_cls.config_obj(**self.agent_config_data)
        return agent_config


class SceneAgentsObjConfig(_Config):
    agents: List[SceneAgentObjConfig] = Field(default=...)

    def create_instances(self) -> List["SceneAgent"]:
        return [agent.create_instance() for agent in self.agents]


class SceneInfoObjConfig(_Config):
    scene_info_config_data: dict = Field(default=...)
    scene_info_obj: DynamicObject = Field(default=...)

    def model_post_init(self, __context: Any) -> None:
        self.valid(self.scene_info_config, self.scene_info_obj)

    def create_instance(self) -> "SceneInfo":
        obj = dynamically_import_obj(self.scene_info_obj)
        return obj.from_config(config=self.scene_info_config)

    @staticmethod
    def valid(scene_info_config: SceneInfoConfigBase, scene_info_obj: DynamicObject):
        scene_info_cls = dynamically_import_obj(scene_info_obj)
        if not issubclass(scene_info_cls, SceneInfo):
            raise TypeError(
                f"scene_info_obj {scene_info_obj.obj} should be a subclass of SceneInfo, "
                f"but get {scene_info_cls.__name__}"
            )
        if not isinstance(scene_info_config, scene_info_cls.config_obj):
            raise TypeError(
                f"scene_info_config should be an instance of {scene_info_cls.config_obj.__name__}, "
                f"but get {scene_info_config.__class__.__name__}"
            )

    @property
    def scene_info_config(self) -> SceneInfoConfigBase:
        scene_info_cls = dynamically_import_obj(self.scene_info_obj)
        return scene_info_cls.config_obj(**self.scene_info_config_data)


class SceneConfig(_Config):
    scene_info: SceneInfoObjConfig = Field(default=...)
    # agents specified here are dynamic agents whose roles remain unknown before assignment (Scene._post_init_agents)
    scene_agents: SceneAgentsObjConfig = Field(default=...)


class Scene(_Configurable):
    config_obj = SceneConfig
    config: SceneConfig

    metadata: SceneMetaData
    dynamic_agent_base_classes: List[Type[SceneAgent]]
    scene_info_class: Type[SceneInfo]

    def __init__(self, config: config_obj):
        super().__init__(config=config)

        self.scene_info: SceneInfo = self.config.scene_info.create_instance()
        self.agents: List[SceneAgent] = self.config.scene_agents.create_instances()
        self.static_agents: List[SceneAgent] = self._init_static_agents()
        self._valid_agent_num()
        self._post_init_agents()

        self.socket_cache: List[SocketData] = []
        self.message_pool: MessagePool = MessagePool()
        self.state = SceneState.PENDING

    def _init_static_agents(self) -> List[SceneAgent]:
        agents = []
        for role in self.scene_info.roles.values():
            if role.is_static:
                agent_obj: Type[SceneAgent] = dynamically_import_obj(role.agent_type)
                # by this way, all fields in static agent config should have default value
                # TODO: is there a better solution so that we can remove above constraint?
                try:
                    agent_config = agent_obj.config_obj()
                except:
                    raise ValidationError(f"all fields of {agent_obj.config_obj.__name__} must have default value")
                agent_config.profile.role = role
                agents.append(agent_obj.from_config(config=agent_config))
        return agents

    def _valid_agent_num(self):
        roles_agent_num = {
            n: v for n, v in self.scene_info.roles_agent_num.items() if not self.scene_info.get_role(n).is_static
        }  # only focus on dynamic roles
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

    def _post_init_agents(self) -> None:
        # static agents already specified roles at initialization
        for agent in self.static_agents:
            agent.post_init(None, self.scene_info)

        # assign roles that are not static to the remained agents
        agent_indices = list(range(len(self.agents)))
        random.shuffle(agent_indices)
        fix_num_roles = list(
            chain(
                *[
                    [role] * num for role, num in self.scene_info.roles_agent_num.items()
                    if num != -1 and not self.scene_info.get_role(role).is_static
                ]
            )
        )
        for idx, agent_idx in enumerate(agent_indices[:len(fix_num_roles)]):
            role = self.scene_info.get_role(fix_num_roles[idx])
            self.agents[agent_idx].post_init(role, self.scene_info)
        dynamic_num_roles = [
            role for role, num in self.scene_info.roles_agent_num.items()
            if num == -1 and not self.scene_info.get_role(role).is_static
        ]
        for agent_idx in agent_indices[len(fix_num_roles):]:
            role = self.scene_info.get_role(random.choice(dynamic_num_roles))
            self.agents[agent_idx].post_init(role, self.scene_info)

    @abstractmethod
    async def _run(self):
        pass

    async def stream_sockets(self, websocket: WebSocket):
        cur = 0
        while self.state != SceneState.FINISHED:
            if cur >= len(self.socket_cache):
                await asyncio.sleep(0.001)
            else:
                await websocket.send_json(self.socket_cache[cur].model_dump_json())
                await asyncio.sleep(0.001)
                cur += 1
        for socket in self.socket_cache[cur:]:
            await websocket.send_json(socket.model_dump_json())

    async def stream_logs(self, log_handler: Callable[[LogBody], Any] = print):
        cur = 0
        while self.state != SceneState.FINISHED:
            if cur >= len(self.socket_cache):
                await asyncio.sleep(0.001)
            else:
                socket = self.socket_cache[cur]
                if socket.type == SocketDataType.LOG:
                    if asyncio.iscoroutinefunction(log_handler):
                        await log_handler(socket.data)
                    else:
                        log_handler(socket.data)
                await asyncio.sleep(0.001)
                cur += 1
        for socket in self.socket_cache[cur:]:
            if socket.type == SocketDataType.LOG:
                if asyncio.iscoroutinefunction(log_handler):
                    await log_handler(socket.data)
                else:
                    log_handler(socket.data)

    def export_logs(self, file: str):
        with open(file, "w", encoding="utf-8") as f:
            for socket in self.socket_cache:
                if socket.type == SocketDataType.LOG:
                    f.write(socket.data.model_dump_json(by_alias=True) + "\n")

    def start(self):
        async def _run_wrapper():
            self.state = SceneState.RUNNING
            await self._run()
            self.state = SceneState.FINISHED

        asyncio.new_event_loop().run_until_complete(_run_wrapper())

    async def a_start(self):
        async def _run_wrapper():
            self.state = SceneState.RUNNING
            await self._run()
            self.state = SceneState.FINISHED

        await asyncio.gather(_run_wrapper())

    @classmethod
    def from_config(cls, config: config_obj) -> "Scene":
        return cls(config=config)

    @classmethod
    def get_metadata(cls) -> SceneMetaData:
        try:
            return cls.metadata
        except:
            raise ValueError("metadata not found, please specify metadata in your scene class")

    @classmethod
    def get_dynamic_agent_classes(cls) -> List[DynamicObject]:
        try:
            base_classes = cls.dynamic_agent_base_classes
        except:
            raise ValueError(
                "dynamic_agent_base_classes not found, please specify dynamic_agent_base_classes in your scene class"
            )

        search_dir = join(dirname(inspect.getfile(cls)), "agents")

        classes = []
        for base_cls in base_classes:
            classes.extend(find_subclasses(search_dir, base_cls))

        return classes

    @classmethod
    def get_scene_info_class(cls) -> Type[SceneInfo]:
        try:
            return cls.scene_info_class
        except:
            raise ValueError(
                "scene_info_class not found, please specify scene_info_class in your scene class"
            )


__all__ = [
    "SceneAgentObjConfig",
    "SceneAgentsObjConfig",
    "SceneInfoObjConfig",
    "SceneConfig",
    "Scene"
]
