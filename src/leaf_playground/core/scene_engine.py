import asyncio
import json
import traceback
from enum import Enum
from os import makedirs
from os.path import join
from typing import Callable, List, Literal, Type, Union
from uuid import uuid4

from pydantic import Field

from .scene import Scene
from .scene_definition import SceneConfig, SceneDefinition
from .workers import MetricEvaluator, MetricEvaluatorConfig, MetricReporter, Logger, SocketHandler
from .workers.chart import Chart
from .._config import _Config
from ..data.log_body import SystemLogBody, SystemEvent
from ..utils.import_util import dynamically_import_obj, DynamicObject


class SceneObjConfig(_Config):
    scene_config_data: dict = Field(default=...)
    scene_obj: DynamicObject = Field(default=...)

    def initialize_scene(self, logger: Logger) -> Scene:
        scene_cls: Type[Scene] = dynamically_import_obj(self.scene_obj)
        scene_config_cls: Type[SceneConfig] = scene_cls.config_cls
        return scene_cls(config=scene_config_cls(**self.scene_config_data), logger=logger)


class ReporterObjConfig(_Config):
    charts: List[DynamicObject] = Field(default=...)

    def initialize_reporter(self, scene_definition: SceneDefinition) -> MetricReporter:
        chart_classes = []
        for chart_dynamic_obj in self.charts:
            chart_cls: Type[Chart] = dynamically_import_obj(chart_dynamic_obj)
            chart_classes.append(chart_cls)
        return MetricReporter(scene_definition=scene_definition, chart_classes=chart_classes)


class MetricEvaluatorObjConfig(_Config):
    evaluator_config_data: dict = Field(default=...)
    evaluator_obj: DynamicObject = Field(default=...)

    def initialize_evaluator(
            self,
            scene_config: SceneConfig,
            logger: Logger,
            reporter: MetricReporter
    ) -> MetricEvaluator:
        evaluator_cls: Type[MetricEvaluator] = dynamically_import_obj(self.evaluator_obj)
        evaluator_config_cls: Type[MetricEvaluatorConfig] = evaluator_cls.config_cls
        return evaluator_cls(
            config=evaluator_config_cls(**self.evaluator_config_data),
            scene_config=scene_config,
            logger=logger,
            reporter=reporter
        )


class MetricEvaluatorObjsConfig(_Config):
    evaluators: List[MetricEvaluatorObjConfig] = Field(default=[])

    def initialize_evaluators(
            self,
            scene_config: SceneConfig,
            logger: Logger,
            reporter: MetricReporter
    ) -> List[MetricEvaluator]:
        return [
            evaluator_obj_config.initialize_evaluator(scene_config=scene_config, logger=logger, reporter=reporter)
            for evaluator_obj_config in self.evaluators
        ]


class SceneEngineState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    FINISHED = "finished"
    INTERRUPTED = "interrupted"
    PAUSED = "paused"
    FAILED = "failed"


class SceneEngineStateProxy:
    def __init__(
            self,
            bound_name: str = "_state",
            callbacks_attr_name: str = "_state_change_callbacks"
    ):
        self._bound_name = bound_name
        self._callbacks_attr_name = callbacks_attr_name

    def __get__(self, instance, owner):
        return getattr(instance, self._bound_name)

    def __set__(self, instance, value):
        setattr(instance, self._bound_name, value)
        for cb in getattr(instance, self._callbacks_attr_name):
            cb()


class SceneEngine:
    state = SceneEngineStateProxy()

    def __init__(
            self,
            scene_config: SceneObjConfig,
            evaluators_config: MetricEvaluatorObjsConfig,
            reporter_config: ReporterObjConfig,
            state_change_callbacks: List[Callable] = []
    ):
        self._state_change_callbacks = state_change_callbacks
        self.state = SceneEngineState.PENDING
        self._id = "engine_" + uuid4().hex[:8]

        self.logger = Logger()
        self.socket_handler = SocketHandler()

        self.logger.registry_handler(self.socket_handler)

        self.scene = scene_config.initialize_scene(logger=self.logger)
        self.reporter = reporter_config.initialize_reporter(scene_definition=self.scene.scene_definition)
        self.evaluators = evaluators_config.initialize_evaluators(
            scene_config=self.scene.config,
            logger=self.logger,
            reporter=self.reporter
        )

        for evaluator in self.evaluators:
            self.scene.registry_metric_evaluator(evaluator)

        self.save_dir = None

    @property
    def id(self) -> str:
        return self._id

    def _wait_agents_ready(self):
        self.scene.wait_agents_ready()

    async def run(self):
        self._wait_agents_ready()

        self.state = SceneEngineState.RUNNING

        for evaluator in self.evaluators:
            evaluator.start()

        self.logger.add_log(
            SystemLogBody(system_event=SystemEvent.SIMULATION_START)
        )
        try:
            await self.scene.run()
        except asyncio.CancelledError:
            for evaluator in self.evaluators:
                evaluator.terminate()
        except:
            self.state = SceneEngineState.FAILED
            self.logger.add_log(
                SystemLogBody(system_event=SystemEvent.SIMULATION_FAILED, log_msg=traceback.format_exc())
            )
            for evaluator in self.evaluators:
                evaluator.terminate()
        else:
            self.logger.add_log(
                SystemLogBody(system_event=SystemEvent.SIMULATION_FINISHED)
            )
            for evaluator in self.evaluators:
                evaluator.join()
            self.state = SceneEngineState.FINISHED
            self.logger.add_log(
                SystemLogBody(system_event=SystemEvent.EVALUATION_FINISHED)
            )
            self.logger.add_log(
                SystemLogBody(system_event=SystemEvent.EVERYTHING_DONE)
            )

    def pause(self):
        if self.state not in [SceneEngineState.FINISHED, SceneEngineState.INTERRUPTED, SceneEngineState.FAILED]:
            self.logger.add_log(
                SystemLogBody(system_event=SystemEvent.SIMULATION_PAUSED)
            )
            self.state = SceneEngineState.PAUSED
            self.scene.pause()

    def resume(self):
        if self.state not in [SceneEngineState.FINISHED, SceneEngineState.INTERRUPTED, SceneEngineState.FAILED]:
            self.logger.add_log(
                SystemLogBody(system_event=SystemEvent.SIMULATION_RESUME)
            )
            self.state = SceneEngineState.RUNNING
            self.scene.resume()

    def interrupt(self):
        if self.state not in [SceneEngineState.FINISHED, SceneEngineState.INTERRUPTED, SceneEngineState.FAILED]:
            self.logger.add_log(
                SystemLogBody(system_event=SystemEvent.SIMULATION_INTERRUPTED)
            )
            self.state = SceneEngineState.INTERRUPTED
            self.scene.interrupt()

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

        with open(join(self.save_dir, "logs.jsonl"), "w", encoding="utf-8") as f:
            for log in self.logger.logs:
                f.write(json.dumps(log.model_dump(mode="json"), ensure_ascii=False) + "\n")

        metrics_data, charts = self.reporter.generate_reports(
            scene_config=self.get_scene_config(mode="pydantic"),
            evaluator_configs=self.get_evaluator_configs(mode="pydantic"),
            logs=self.logger.logs
        )
        metrics_data = {
            "metrics": {
                name: [each.model_dump(mode="json") for each in data]
                if isinstance(data, list) else data.model_dump(mode="json")
                for name, data in metrics_data["metrics"].items()
            },
            "human_metrics": {
                name: [each.model_dump(mode="json") for each in data]
                if isinstance(data, list) else data.model_dump(mode="json")
                for name, data in metrics_data["human_metrics"].items()
            }
        }
        with open(join(self.save_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=4, ensure_ascii=False)

        with open(join(self.save_dir, "charts.json"), "w", encoding="utf-8") as f:
            json.dump(charts, f, indent=4, ensure_ascii=False)


__all__ = [
    "SceneObjConfig",
    "MetricEvaluatorObjConfig",
    "MetricEvaluatorObjsConfig",
    "ReporterObjConfig",
    "SceneEngineState",
    "SceneEngine",
]
