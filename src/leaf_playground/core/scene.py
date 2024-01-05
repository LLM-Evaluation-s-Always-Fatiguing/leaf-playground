import asyncio
import time
from abc import abstractmethod, ABC, ABCMeta
from sys import _getframe
from typing import Dict, List, Optional, Type

from pydantic import BaseModel, Field

from .scene_agent import SceneAgent, SceneDynamicAgent, SceneHumanAgent
from .scene_definition import SceneDefinition, SceneConfig
from .workers import MetricEvaluator, Logger
from .._config import _Configurable
from .._type import Immutable
from ..data.environment import EnvironmentVariable
from ..data.log_body import ActionLogBody
from ..data.message import MessagePool
from ..utils.import_util import DynamicObject
from ..utils.type_util import validate_type


class SceneMetaClass(ABCMeta):
    def __new__(
        cls,
        name,
        bases,
        attrs,
        *,
        scene_definition: SceneDefinition = None,
        log_body_class: Type[ActionLogBody] = ActionLogBody,
    ):
        attrs["scene_definition"] = Immutable(scene_definition or getattr(bases[0], "scene_definition", None))
        attrs["log_body_class"] = Immutable(log_body_class)
        attrs["obj_for_import"] = Immutable(DynamicObject(obj=name, module=_getframe(1).f_globals["__name__"]))

        new_cls = super().__new__(cls, name, bases, attrs)

        DynamicObject.bind_dynamic_obj(attrs["obj_for_import"], new_cls)

        if not validate_type(attrs["scene_definition"], Immutable[Optional[SceneDefinition]]):
            raise TypeError(
                f"class [{name}]'s class attribute [scene_definition] should be a [SceneDefinition] instance, "
                f"got [{type(attrs['scene_definition']).__name__}] type"
            )
        if not validate_type(attrs["log_body_class"], Immutable[Type[ActionLogBody]]):
            raise TypeError(
                f"class [{name}]'s class attribute [log_body_class] should be subclass of [ActionLogBody]"
            )

        if ABC not in bases:
            # check if those class attrs are empty when the class is not abstract
            if not new_cls.scene_definition:
                raise AttributeError(
                    f"class [{name}] missing class attribute [scene_definition], please specify it by "
                    f"doing like: `class {name}(scene_definition=your_scene_def)`, or you can also "
                    f"specify in the super class [{bases[0].__name__}]"
                )

        return new_cls

    def __init__(
        cls,
        name,
        bases,
        attrs,
        *,
        scene_definition: SceneDefinition = None,
        log_body_class: Type[ActionLogBody] = ActionLogBody
    ):
        super().__init__(name, bases, attrs)

    def __setattr__(self, key, value):
        # make sure those class attributes immutable in class-wise
        if key in ["scene_definition", "log_body_class", "obj_for_import"] and hasattr(self, key):
            raise AttributeError(f"class attribute {key} is immutable")
        return super().__setattr__(key, value)

    def __delattr__(self, item):
        # make sure those class attributes can't be deleted
        if item in ["scene_definition", "log_body_class", "obj_for_import"] and hasattr(self, item):
            raise AttributeError(f"class attribute [{item}] can't be deleted")
        return super().__delattr__(item)


class SceneMetadata(BaseModel):
    scene_definition: SceneDefinition = Field(default=...)
    config_schema: dict = Field(default=...)
    obj_for_import: DynamicObject = Field(default=...)


class Scene(_Configurable, ABC, metaclass=SceneMetaClass):
    config_cls = SceneConfig
    config: SceneConfig

    # class attributes initialized in metaclass
    scene_definition: SceneDefinition
    log_body_class: Type[ActionLogBody]
    obj_for_import: DynamicObject

    def __init__(self, config: config_cls, logger: Logger):
        super().__init__(config=config)

        static_roles = [role_def.name for role_def in self.scene_definition.roles if role_def.is_static]
        dynamic_roles = [role_def.name for role_def in self.scene_definition.roles if not role_def.is_static]
        agents: Dict[str, List[SceneAgent]] = self.config.init_agents()
        self.static_agents = {role_name: agents[role_name] for role_name in static_roles}
        self.agents = {role_name: agents[role_name] for role_name in dynamic_roles}
        self.env_vars: Dict[str, EnvironmentVariable] = self.config.init_env_vars()
        self._bind_env_vars_to_agents()
        self.evaluators: List[MetricEvaluator] = []

        self.logger = logger
        self.message_pool: MessagePool = MessagePool()

        self._dynamic_agents: Dict[str, SceneDynamicAgent] = {}
        self._agent_list: List[SceneAgent] = []
        self._human_agents: List[SceneHumanAgent] = []
        for agents_ in self.agents.values():
            self._human_agents += [agent for agent in agents_ if isinstance(agent, SceneHumanAgent)]
            self._agent_list += agents_
            self._dynamic_agents.update(**{agent.id: agent for agent in agents_})
        for agents_ in self.static_agents.values():
            self._agent_list += agents_

        self._run_task: Optional[asyncio.Task] = None

    @property
    def dynamic_agents(self) -> Dict[str, SceneDynamicAgent]:
        return self._dynamic_agents

    @property
    def human_agents(self) -> List[SceneHumanAgent]:
        return self._human_agents

    def get_dynamic_agent(self, agent_id: str) -> SceneDynamicAgent:
        return self._dynamic_agents[agent_id]

    def _bind_env_vars_to_agents(self):
        for agents in self.static_agents.values():
            for agent in agents:
                agent.bind_env_vars(self.env_vars)
        for agents in self.agents.values():
            for agent in agents:
                agent.bind_env_vars(self.env_vars)

    def wait_agents_ready(self):
        while True:
            all_agents_ready = True
            for agents in self.agents.values():
                agents: List[SceneDynamicAgent]
                if not all(agent.connected for agent in agents):
                    all_agents_ready = False
            if all_agents_ready:
                break
            time.sleep(0.1)

    def registry_metric_evaluator(self, evaluator: MetricEvaluator):
        self.evaluators.append(evaluator)

    def notify_evaluators_record(self, log: ActionLogBody):
        for evaluator in self.evaluators:
            evaluator.notify_to_record(log)

    def notify_evaluators_compare(self, log: ActionLogBody):
        for evaluator in self.evaluators:
            evaluator.notify_to_compare(log)

    def notify_evaluators_can_stop(self):
        for evaluator in self.evaluators:
            evaluator.notify_can_stop()

    @abstractmethod
    async def _run(self):
        pass

    async def run(self):
        self._run_task = asyncio.ensure_future(self._run())
        try:
            await self._run_task
            self.notify_evaluators_can_stop()
        except:
            self.notify_evaluators_can_stop()
            raise

    def pause(self):
        for agent in self._agent_list:
            agent.pause()

    def resume(self):
        for agent in self._agent_list:
            agent.resume()

    def interrupt(self):
        if self._run_task and not self._run_task.done():
            self._run_task.cancel()

    @classmethod
    def get_metadata(cls) -> SceneMetadata:
        return SceneMetadata(
            scene_definition=cls.scene_definition,
            config_schema=cls.config_cls.get_json_schema(by_alias=True),
            obj_for_import=cls.obj_for_import
        )

    @classmethod
    def from_config(cls, config: config_cls) -> "Scene":
        raise NotImplementedError()

    @classmethod
    def from_config_file(cls, file_path: str) -> "Scene":
        raise NotImplementedError()


__all__ = [
    "Scene",
    "SceneMetadata"
]
