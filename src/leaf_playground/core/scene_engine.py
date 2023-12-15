import asyncio
import json
from enum import Enum
from os import makedirs
from os.path import join
from typing import Any, Callable, List, Literal, Type, Union
from uuid import uuid4, UUID

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import Field

from .scene import Scene
from .scene_definition import SceneConfig
from .scene_observer import MetricEvaluator, MetricEvaluatorConfig, MetricReporter
from .._config import _Config
from ..data.log_body import LogBody
from ..data.socket_data import SocketData, SocketDataType
from ..utils.import_util import dynamically_import_obj, DynamicObject


class SceneObjConfig(_Config):
    scene_config_data: dict = Field(default=...)
    scene_obj: DynamicObject = Field(default=...)

    def initialize_scene(self) -> Scene:
        scene_cls: Type[Scene] = dynamically_import_obj(self.scene_obj)
        scene_config_cls: Type[SceneConfig] = scene_cls.config_cls
        return scene_cls(config=scene_config_cls(**self.scene_config_data))


class MetricEvaluatorObjConfig(_Config):
    evaluator_config_data: dict = Field(default=...)
    evaluator_obj: DynamicObject = Field(default=...)

    def initialize_evaluator(
        self,
        scene_config: SceneConfig,
        socket_cache: List[SocketData],
        reporter: MetricReporter
    ) -> MetricEvaluator:
        evaluator_cls: Type[MetricEvaluator] = dynamically_import_obj(self.evaluator_obj)
        evaluator_config_cls: Type[MetricEvaluatorConfig] = evaluator_cls.config_cls
        return evaluator_cls(
            config=evaluator_config_cls(**self.evaluator_config_data),
            scene_config=scene_config,
            socket_cache=socket_cache,
            reporter=reporter
        )


class MetricEvaluatorObjsConfig(_Config):
    evaluators: List[MetricEvaluatorObjConfig] = Field(default=[])

    def initialize_evaluators(
        self,
        scene_config: SceneConfig,
        socket_cache: List[SocketData],
        reporter: MetricReporter
    ) -> List[MetricEvaluator]:
        return [
            evaluator_obj_config.initialize_evaluator(
                scene_config=scene_config, socket_cache=socket_cache, reporter=reporter
            )
            for evaluator_obj_config in self.evaluators
        ]


class SceneEngineState(Enum):
    PENDING = 0
    RUNNING = 1
    FINISHED = 2
    INTERRUPTED = 3
    PAUSED = 4
    FAILED = 5


class SceneEngine:
    def __init__(self, scene_config: SceneObjConfig, evaluators_config: MetricEvaluatorObjsConfig):
        self.scene = scene_config.initialize_scene()
        self.reporter = MetricReporter(scene_definition=self.scene.scene_definition)
        self.evaluators = evaluators_config.initialize_evaluators(
            scene_config=self.scene.config,
            socket_cache=self.scene.socket_cache,
            reporter=self.reporter
        )

        for evaluator in self.evaluators:
            self.scene.registry_metric_evaluator(evaluator)

        self.socket_cache = self.scene.socket_cache
        self._state = SceneEngineState.PENDING
        self._id = uuid4()

        self.save_dir = None

    @property
    def state(self) -> SceneEngineState:
        return self._state

    @property
    def id(self) -> UUID:
        return self._id

    def run(self):
        self._state = SceneEngineState.RUNNING

        for evaluator in self.evaluators:
            evaluator.start()

        self.scene.start()

        for evaluator in self.evaluators:
            evaluator.join()

        self._state = SceneEngineState.FINISHED

    async def a_run(self):
        self._state = SceneEngineState.RUNNING

        for evaluator in self.evaluators:
            evaluator.start()

        await self.scene.a_start()

        for evaluator in self.evaluators:
            evaluator.join()

        self._state = SceneEngineState.FINISHED

    async def stream_sockets(self, websocket: WebSocket):
        cur = 0
        try:
            while self.state in [SceneEngineState.PENDING, SceneEngineState.RUNNING, SceneEngineState.PAUSED]:
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
        while self.state != SceneEngineState.FINISHED:
            if cur >= len(self.socket_cache):
                await asyncio.sleep(0.001)
            else:
                socket = self.socket_cache[cur]
                if socket.type == SocketDataType.LOG:
                    if asyncio.iscoroutinefunction(log_handler):
                        await log_handler(self.scene.log_body_class(**socket.data))
                    else:
                        log_handler(self.scene.log_body_class(**socket.data))
                await asyncio.sleep(0.001)
                cur += 1
        for socket in self.socket_cache[cur:]:
            if socket.type == SocketDataType.LOG:
                if asyncio.iscoroutinefunction(log_handler):
                    await log_handler(self.scene.log_body_class(**socket.data))
                else:
                    log_handler(self.scene.log_body_class(**socket.data))

    def get_scene_config(
        self,
        mode: Literal["pydantic", "dict", "json"] = "dict"
    ) -> Union[SceneConfig, dict, str]:
        if mode == "pydantic":
            return self.scene.config
        elif mode == "dict":
            return self.scene.config.model_dump(mode="json", by_alias=True)
        elif mode == "json":
            return self.scene.config.model_dump_json()
        else:
            raise ValueError(f"invalid mode {mode}")

    def get_evaluator_configs(
        self,
        mode: Literal["pydantic", "dict", "json"] = "dict"
    ) -> Union[List[MetricEvaluatorConfig], List[dict], str]:
        if mode == "pydantic":
            return [evaluator.config for evaluator in self.evaluators]
        elif mode == "dict":
            return [evaluator.config.model_dump(mode="json") for evaluator in self.evaluators]
        elif mode == "json":
            return json.dumps([evaluator.config.model_dump(mode="json") for evaluator in self.evaluators])
        else:
            raise ValueError(f"invalid mode {mode}")

    def save(self):
        makedirs(self.save_dir, exist_ok=True)

        scene_config = self.get_scene_config(mode="dict")
        evaluator_configs = self.get_evaluator_configs(mode="dict")

        with open(join(self.save_dir, "scene_config.json"), "w", encoding="utf-8") as f:
            json.dump(scene_config, f, indent=4, ensure_ascii=False)

        with open(join(self.save_dir, "evaluator_configs.json"), "w", encoding="utf-8") as f:
            json.dump(evaluator_configs, f, indent=4, ensure_ascii=False)

        with open(join(self.save_dir, "sockets.jsonl"), "w", encoding="utf-8") as f:
            for socket_data in self.socket_cache:
                f.write(json.dumps(socket_data.model_dump(mode="json"), ensure_ascii=False) + "\n")

        # TODO: save charts


__all__ = [
    "SceneObjConfig",
    "MetricEvaluatorObjConfig",
    "MetricEvaluatorObjsConfig",
    "SceneEngineState",
    "SceneEngine"
]
