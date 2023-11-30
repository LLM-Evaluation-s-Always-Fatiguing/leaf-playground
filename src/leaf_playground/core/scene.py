import asyncio
import inspect
import json
import random
from abc import abstractmethod
from datetime import datetime
from itertools import chain
from os import getcwd, makedirs
from os.path import dirname, join
from typing import Any, Callable, List, Optional, Type
from uuid import uuid4

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import Field, ValidationError

from .scene_agent import SceneAgent, SceneAgentConfig
from .scene_evaluator import SceneEvaluatorConfig, SceneEvaluator
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

    def create_instance(self) -> SceneAgent:
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

    def create_instances(self) -> List[SceneAgent]:
        return [agent.create_instance() for agent in self.agents]


class SceneEvaluatorObjConfig(_Config):
    evaluator_config_data: dict = Field(default=...)
    evaluator_obj: DynamicObject = Field(default=...)

    def model_post_init(self, __context: Any) -> None:
        self.valid(self.evaluator_config, self.evaluator_obj)

    def create_instance(self, agents: List[SceneAgent]) -> SceneEvaluator:
        obj: Type[SceneEvaluator] = dynamically_import_obj(self.evaluator_obj)
        return obj(config=self.evaluator_config, agents=agents)

    @staticmethod
    def valid(evaluator_config: SceneEvaluatorConfig, evaluator_obj: DynamicObject):
        evaluator_cls = dynamically_import_obj(evaluator_obj)
        if not issubclass(evaluator_cls, SceneEvaluator):
            raise TypeError(
                f"evaluator_obj {evaluator_obj.obj} should be a subclass of SceneEvaluator, "
                f"but get {evaluator_cls.__name__}"
            )
        if not isinstance(evaluator_config, evaluator_cls.config_obj):
            raise TypeError(
                f"evaluator_config should be an instance of {evaluator_cls.config_obj.__name__}, "
                f"but get {evaluator_config.__class__.__name__}"
            )

    @property
    def evaluator_config(self) -> SceneEvaluatorConfig:
        evaluator_cls = dynamically_import_obj(self.evaluator_obj)
        evaluator_config = evaluator_cls.config_obj(**self.evaluator_config_data)
        return evaluator_config


class SceneEvaluatorsObjConfig(_Config):
    evaluators: List[SceneEvaluatorObjConfig] = Field(default=[])

    def create_instances(self, agents: List[SceneAgent]) -> List["SceneEvaluator"]:
        return [evaluator.create_instance(agents) for evaluator in self.evaluators]


class SceneInfoObjConfig(_Config):
    scene_info_config_data: dict = Field(default=...)
    scene_info_obj: DynamicObject = Field(default=...)

    def model_post_init(self, __context: Any) -> None:
        self.valid(self.scene_info_config, self.scene_info_obj)

    def create_instance(self) -> SceneInfo:
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
    debug_mode: bool = Field(default=False, exclude=True)
    scene_info: SceneInfoObjConfig = Field(default=...)
    # agents specified here are dynamic agents whose roles remain unknown before assignment (Scene._post_init_agents)
    scene_agents: SceneAgentsObjConfig = Field(default=...)
    scene_evaluators: SceneEvaluatorsObjConfig = Field(default=...)


class Scene(_Configurable):
    config_obj = SceneConfig
    config: SceneConfig

    metadata: SceneMetaData
    dynamic_agent_base_classes: List[Type[SceneAgent]]
    evaluator_classes: Optional[List[Type[SceneEvaluator]]] = None
    scene_info_class: Type[SceneInfo]
    log_body_class: Type[LogBody]

    def __init__(self, config: config_obj):
        if hasattr(self, f"_{self.__class__.__name__}__valid_class_attributes"):
            getattr(self, f"_{self.__class__.__name__}__valid_class_attributes")()
        self.__valid_class_attributes()
        super().__init__(config=config)

        self.scene_info: SceneInfo = self.config.scene_info.create_instance()
        self.agents: List[SceneAgent] = self.config.scene_agents.create_instances()
        self.static_agents: List[SceneAgent] = self._init_static_agents()
        self.evaluators: List[SceneEvaluator] = self.config.scene_evaluators.create_instances(self.agents)
        self._valid_agent_num()
        self._post_init_agents()

        self.socket_cache: List[SocketData] = []
        self.message_pool: MessagePool = MessagePool()
        self.state = SceneState.PENDING

    def __valid_class_attributes(self):
        if not hasattr(self, "metadata"):
            raise ValueError("metadata not found, please specify in your scene class")
        if not hasattr(self, "dynamic_agent_base_classes"):
            raise ValueError(
                "dynamic_agent_base_classes not found, please specify in your scene class"
            )
        if not hasattr(self, "scene_info_class"):
            raise ValueError(
                "scene_info_class not found, please specify in your scene class"
            )
        if not hasattr(self, "log_body_class"):
            raise ValueError(
                "log_body_class not found, please specify in your scene class"
            )

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
        try:
            while self.state in [SceneState.PENDING, SceneState.RUNNING, SceneState.PAUSED]:
                if cur >= len(self.socket_cache):
                    await asyncio.sleep(0.001)
                else:
                    await websocket.send_json(self.socket_cache[cur].model_dump_json())
                    await asyncio.sleep(0.001)
                    cur += 1
            for socket in self.socket_cache[cur:]:
                await websocket.send_json(socket.model_dump_json())
        except WebSocketDisconnect:
            pass

    async def stream_logs(self, log_handler: Callable[[LogBody], Any] = print):
        cur = 0
        while self.state != SceneState.FINISHED:
            if cur >= len(self.socket_cache):
                await asyncio.sleep(0.001)
            else:
                socket = self.socket_cache[cur]
                if socket.type == SocketDataType.LOG:
                    if asyncio.iscoroutinefunction(log_handler):
                        await log_handler(self.log_body_class(**socket.data))
                    else:
                        log_handler(self.log_body_class(**socket.data))
                await asyncio.sleep(0.001)
                cur += 1
        for socket in self.socket_cache[cur:]:
            if socket.type == SocketDataType.LOG:
                if asyncio.iscoroutinefunction(log_handler):
                    await log_handler(self.log_body_class(**socket.data))
                else:
                    log_handler(self.log_body_class(**socket.data))

    def start(self):
        async def _run_wrapper():
            self.state = SceneState.RUNNING
            await self._run()
            self.save()
            self.state = SceneState.FINISHED

        asyncio.new_event_loop().run_until_complete(_run_wrapper())

    async def a_start(self):
        async def _run_wrapper():
            self.state = SceneState.RUNNING
            await self._run()
            self.save()
            self.state = SceneState.FINISHED

        await asyncio.gather(_run_wrapper())

    @classmethod
    def obj_for_import(cls) -> DynamicObject:
        return DynamicObject(obj=cls.__name__, source_file=inspect.getfile(cls))

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
    def get_evaluator_classes(cls) -> Optional[List[DynamicObject]]:
        return cls.evaluator_classes if not cls.evaluator_classes else [
            evaluator_cls.obj_for_import for evaluator_cls in cls.evaluator_classes
        ]

    @classmethod
    def get_scene_info_class(cls) -> Type[SceneInfo]:
        try:
            return cls.scene_info_class
        except:
            raise ValueError(
                "scene_info_class not found, please specify scene_info_class in your scene class"
            )

    def _save_scene(self, save_dir: str):
        with open(join(save_dir, "scene.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": self.config.model_dump(mode="json"),
                    "metadata": self.metadata.model_dump(mode="json"),
                    "type": self.obj_for_import().model_dump(mode="json")
                },
                f,
                ensure_ascii=False,
                indent=2
            )

    def _save_agents(self, save_dir: str):
        agents_info = {}
        for agent in self.agents:
            config = agent.config.model_dump(mode="json")
            agents_info[config["profile"]["id"]] = {
                "config": config,
                "metadata": agent.get_metadata().model_dump(mode="json"),
                "type": agent.obj_for_import.model_dump(mode="json")
            }
        with open(join(save_dir, "agents.json"), "w", encoding="utf-8") as f:
            json.dump(agents_info, f, ensure_ascii=False, indent=2)

    def _save_logs(self, save_dir: str):
        with open(join(save_dir, "logs.jsonl"), "w", encoding="utf-8") as f:
            for socket in self.socket_cache:
                if socket.type == SocketDataType.LOG:
                    f.write(json.dumps(socket.data, ensure_ascii=False) + "\n")

    def _save_metrics(self, save_dir: str):
        with open(join(save_dir, "metrics.jsonl"), "w", encoding="utf-8") as f:
            for socket in self.socket_cache:
                if socket.type == SocketDataType.METRIC:
                    f.write(json.dumps(socket.data, ensure_ascii=False) + "\n")

    def _save_charts(self, save_dir: str):
        charts_dir = join(save_dir, "charts")
        makedirs(charts_dir, exist_ok=True)
        if not self.evaluators:
            return
        for evaluator in self.evaluators:
            for chart in evaluator.paint_charts():
                chart.render_chart(join(charts_dir, f"{chart.name}.html"))
                chart.dump_chart_options(join(charts_dir, f"{chart.name}.json"))

    def save(self, save_dir: Optional[str] = None):
        if not save_dir:
            save_dir = join(getcwd(), f"tmp/{datetime.utcnow().timestamp().hex() + uuid4().hex}")

        makedirs(save_dir, exist_ok=True)

        self._save_scene(save_dir)
        self._save_agents(save_dir)
        self._save_logs(save_dir)
        self._save_metrics(save_dir)
        self._save_charts(save_dir)

        self.socket_cache.append(
            SocketData(
                type=SocketDataType.ENDING,
                data={"save_dir": save_dir}
            )
        )


__all__ = [
    "SceneAgentObjConfig",
    "SceneAgentsObjConfig",
    "SceneInfoObjConfig",
    "SceneConfig",
    "Scene"
]
