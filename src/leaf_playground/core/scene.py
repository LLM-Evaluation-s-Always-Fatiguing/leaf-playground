import asyncio
from abc import abstractmethod, ABC, ABCMeta
from sys import _getframe
from typing import Dict, List, Optional, Type

from pydantic import BaseModel, Field

from .scene_agent import SceneAgent
from .scene_definition import SceneDefinition, SceneConfig
from .scene_observer import MetricEvaluator
from .._config import _Configurable
from .._type import Immutable
from ..data.environment import EnvironmentVariable
from ..data.log_body import LogBody
from ..data.message import MessagePool
from ..data.socket_data import SocketData
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
        log_body_class: Type[LogBody] = LogBody,
    ):
        attrs["scene_definition"] = Immutable(scene_definition or getattr(bases[0], "scene_definition", None))
        attrs["log_body_class"] = Immutable(log_body_class)
        attrs["obj_for_import"] = Immutable(DynamicObject(obj=name, source_file=_getframe(1).f_code.co_filename))

        new_cls = super().__new__(cls, name, bases, attrs)

        DynamicObject.bind_dynamic_obj(attrs["obj_for_import"], new_cls)

        if not validate_type(attrs["scene_definition"], Immutable[Optional[SceneDefinition]]):
            raise TypeError(
                f"class [{name}]'s class attribute [scene_definition] should be a [SceneDefinition] instance, "
                f"got [{type(attrs['scene_definition']).__name__}] type"
            )
        if not validate_type(attrs["log_body_class"], Immutable[Type[LogBody]]):
            raise TypeError(
                f"class [{name}]'s class attribute [log_body_class] should be subclass of [LogBody]"
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
        log_body_class: Type[LogBody] = LogBody
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
    log_body_class: Type[LogBody]
    obj_for_import: DynamicObject

    def __init__(self, config: config_cls):
        super().__init__(config=config)

        static_roles = [role_def.name for role_def in self.scene_definition.roles if role_def.is_static]
        dynamic_roles = [role_def.name for role_def in self.scene_definition.roles if not role_def.is_static]
        agents: Dict[str, List[SceneAgent]] = self.config.init_agents()
        self.static_agents = {role_name: agents[role_name] for role_name in static_roles}
        self.agents = {role_name: agents[role_name] for role_name in dynamic_roles}
        self.env_vars: Dict[str, EnvironmentVariable] = self.config.init_env_vars()
        self.evaluators: List[MetricEvaluator] = []

        self.socket_cache: List[SocketData] = []
        self.message_pool: MessagePool = MessagePool()

    def registry_metric_evaluator(self, evaluator: MetricEvaluator):
        self.evaluators.append(evaluator)

    def notify_evaluators_record(self, log: LogBody):
        for evaluator in self.evaluators:
            evaluator.notify_to_record(log)

    def notify_evaluators_compare(self, logs: List[LogBody]):
        for evaluator in self.evaluators:
            evaluator.notify_to_compare(logs)

    def notify_evaluators_can_stop(self):
        for evaluator in self.evaluators:
            evaluator.notify_can_stop()

    @abstractmethod
    async def _run(self):
        pass

    def start(self):
        async def _run_wrapper():
            await self._run()
            self.notify_evaluators_can_stop()

        asyncio.new_event_loop().run_until_complete(_run_wrapper())

    async def a_start(self):
        async def _run_wrapper():
            await self._run()
            self.notify_evaluators_can_stop()

        await _run_wrapper()

    @classmethod
    def get_metadata(cls) -> SceneMetadata:
        return SceneMetadata(
            scene_definition=cls.scene_definition,
            config_schema=cls.config_cls.get_json_schema(by_alias=True),
            obj_for_import=cls.obj_for_import
        )

    @classmethod
    def from_config(cls, config: config_cls) -> "Scene":
        return cls(config=config)

    # def _save_scene(self):
    #     with open(join(self.save_dir, "scene.json"), "w", encoding="utf-8") as f:
    #         json.dump(
    #             {
    #                 "config": self.config.model_dump(mode="json"),
    #                 "metadata": self.metadata.model_dump(mode="json"),
    #                 "type": self.obj_for_import().model_dump(mode="json")
    #             },
    #             f,
    #             ensure_ascii=False,
    #             indent=2
    #         )
    #
    # def _save_agents(self):
    #     agents_info = {}
    #     for agent in self.agents:
    #         config = agent.config.model_dump(mode="json")
    #         agents_info[config["profile"]["id"]] = {
    #             "config": config,
    #             "metadata": agent.get_metadata().model_dump(mode="json"),
    #             "type": agent.obj_for_import.model_dump(mode="json")
    #         }
    #     with open(join(self.save_dir, "agents.json"), "w", encoding="utf-8") as f:
    #         json.dump(agents_info, f, ensure_ascii=False, indent=2)
    #
    # def _save_logs(self):
    #     with open(join(self.save_dir, "logs.jsonl"), "w", encoding="utf-8") as f:
    #         for socket in self.socket_cache:
    #             if socket.type == SocketDataType.LOG:
    #                 f.write(json.dumps(socket.data, ensure_ascii=False) + "\n")

    # def _save_metrics(self):
    #     with open(join(self.save_dir, "metrics.jsonl"), "w", encoding="utf-8") as f:
    #         for socket in self.socket_cache:
    #             if socket.type == SocketDataType.METRIC:
    #                 f.write(json.dumps(socket.data, ensure_ascii=False) + "\n")
    #
    # def _save_charts(self):
    #     charts_dir = join(self.save_dir, "charts")
    #     makedirs(charts_dir, exist_ok=True)
    #     if not self.evaluators:
    #         return
    #     for evaluator in self.evaluators:
    #         for chart in evaluator.paint_charts():
    #             chart.render_chart(join(charts_dir, f"{chart.name}.html"))
    #             chart.dump_chart_options(join(charts_dir, f"{chart.name}.json"))

    # def save(self):
    #     if not self.save_dir:
    #         self.save_dir = join(getcwd(), f"tmp/{datetime.utcnow().timestamp().hex() + uuid4().hex}")
    #
    #     makedirs(self.save_dir, exist_ok=True)
    #
    #     self._save_scene()
    #     self._save_agents()
    #     self._save_logs()
    #     self._save_metrics()
    #     self._save_charts()
    #
    #     self.socket_cache.append(
    #         SocketData(
    #             type=SocketDataType.ENDING,
    #             data={"save_dir": self.save_dir}
    #         )
    #     )


__all__ = [
    "Scene",
    "SceneMetadata"
]
