import asyncio
import importlib
import pickle
import time
from abc import abstractmethod, ABC, ABCMeta
from collections import defaultdict
from enum import Enum
from multiprocessing import Manager, Process
from queue import Empty
from sys import _getframe
from threading import Thread
from types import new_class
from typing import Any, Dict, List, Optional, Type, Union
from uuid import uuid4

from pydantic import BaseModel, Field

from .logger import Logger
from .reporter import MetricReporter
from ..scene_definition import CompareMetricDefinition, MetricDefinition, VALUE_DETYPE_2_DEFAULT_VALUE, SceneConfig
from ..._config import _Config, _Configurable
from ..._type import Immutable
from ...data.log_body import ActionLogBody
from ...utils.import_util import DynamicObject
from ...utils.type_util import validate_type


class RecordOutput(BaseModel):
    record_value: Any = Field(default=...)
    reason: Optional[str] = Field(default=None)
    misc: Optional[dict] = Field(default=None)


class CompareOutput(BaseModel):
    compare_result: Any = Field(default=...)
    reason: Optional[str] = Field(default=None)
    misc: Optional[dict] = Field(default=None)


_MetricName = str
_MetricRecordValue = Any


class MetricEvaluatorProxy(Process):
    def __init__(
        self,
        config_cls: Type["MetricEvaluatorConfig"],
        config_data: dict,
        record_metrics: List[_MetricName],
        compare_metrics: List[_MetricName],
        manager: Manager
    ):
        super().__init__(daemon=True)

        self._queue = manager.Queue()
        self._result_cache = manager.dict()
        self._can_stop = manager.Value("b", False)
        self._state = manager.Value("c", MetricEvaluatorState.PENDING.value.encode("utf-8"))

        self._config_cls = config_cls
        self._config_data = config_data
        self._record_metrics = record_metrics
        self._compare_metrics = compare_metrics

    def __init_subclass__(cls, _init_evaluator, _record, _compare, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._init_evaluator = _init_evaluator
        cls._record = _record
        cls._compare = _compare

    @property
    def can_stop(self):
        return self._can_stop.value

    @can_stop.setter
    def can_stop(self, can_stop: bool):
        self._can_stop.value = can_stop

    @property
    def result_cache(self):
        return self._result_cache

    @property
    def queue(self):
        return self._queue

    @property
    def state(self) -> "MetricEvaluatorState":
        return MetricEvaluatorState(self._state.value.decode("utf-8"))

    def run(self):
        self._state.value = MetricEvaluatorState.INITIALIZING.value.encode("utf-8")
        try:
            evaluator = self._init_evaluator(
                self._config_cls(**self._config_data), self._record_metrics, self._compare_metrics
            )
        except Exception as e:
            self._state.value = MetricEvaluatorState.INIT_FAILED.value.encode("utf-8")
            return
        loop = asyncio.new_event_loop()
        self._state.value = MetricEvaluatorState.RUNNING.value.encode("utf-8")

        while True:
            try:
                log_data, is_compare, id_ = self._queue.get_nowait()
            except Empty:
                time.sleep(0.001)
                continue
            try:
                logs = pickle.loads(log_data)
                if not is_compare:
                    output = loop.run_until_complete(self._record(logs, evaluator))
                else:
                    output = loop.run_until_complete(self._compare(logs, evaluator))
            except Exception as e:
                output = {}
            self._result_cache[id_] = {k: v.model_dump(mode="json", by_alias=True) for k, v in output.items()}

            if self._queue.empty() and self._can_stop.value:
                break

        self._state.value = MetricEvaluatorState.FINISHED.value.encode("utf-8")

    def terminate(self):
        super().terminate()
        self._state.value = MetricEvaluatorState.TERMINATED.value.encode("utf-8")


# TODO: We hope that the MetricEvaluator class can be inherited through composition
#  (or at least subcls can extend defs), however, using meta class makes this tough

class MetricEvaluatorMetaClass(ABCMeta):
    def __new__(
        cls,
        name,
        bases,
        attrs,
        *,
        metric_definitions: List[Union[CompareMetricDefinition, MetricDefinition]] = None,
        cls_description: str = None
    ):
        # create proxy class on the fly
        evaluator_proxy_class = new_class(
            name=f"{name}Proxy",
            bases=(MetricEvaluatorProxy,),
            kwds={
                "_init_evaluator": attrs["_init_evaluator"],
                "_record": attrs["_record"],
                "_compare": attrs["_compare"]
            }
        )
        evaluator_proxy_class.__module__ = _getframe(1).f_globals["__name__"]
        setattr(
            importlib.import_module(evaluator_proxy_class.__module__),
            evaluator_proxy_class.__name__,
            evaluator_proxy_class
        )

        attrs["metric_definitions"] = Immutable(metric_definitions or getattr(bases[0], "metric_definitions", None))
        attrs["cls_description"] = Immutable(cls_description or "")
        attrs["evaluator_proxy_class"] = Immutable(evaluator_proxy_class)
        attrs["obj_for_import"] = Immutable(DynamicObject(obj=name, module=_getframe(1).f_globals["__name__"]))

        new_cls = super().__new__(cls, name, bases, attrs)

        DynamicObject.bind_dynamic_obj(attrs["obj_for_import"], new_cls)

        if not validate_type(attrs["metric_definitions"], Immutable[Optional[List[MetricDefinition]]]):
            raise TypeError(
                f"class [{name}]'s class attribute [metric_definitions] should be a List[MetricDefinition]"
            )
        if not validate_type(attrs["cls_description"], Immutable[Optional[str]]):
            raise TypeError(
                f"class [{name}]'s class attribute [cls_description] should be a [str] instance, "
                f"got [{type(attrs['cls_description']).__name__}] type"
            )
        if not validate_type(attrs["evaluator_proxy_class"], Immutable[Optional[MetricEvaluatorProxy]]):
            raise TypeError(
                f"class [{name}]'s class attribute [evaluator_proxy_class] should be a subclass of "
                f"[MetricEvaluatorProxy]"
            )

        if ABC not in bases:
            # check if those class attrs are empty when the class is not abstract
            if not new_cls.metric_definitions:
                raise AttributeError(
                    f"class [{name}] missing class attribute [metric_definitions], please specify it by "
                    f"doing like: `Class {name}(metric_definitions=your_metric_defs)`, or you can also "
                    f"specify in the super class [{bases[0].__name__}]"
                )
            if not new_cls.cls_description:
                raise AttributeError(
                    f"class [{name}] missing class attribute [cls_description], please specify it by "
                    f"doing like: `class {name}(cls_description=your_cls_desc)`, where 'your_cls_desc' "
                    f"is a string that introduces your evaluator class"
                )
            if not new_cls.evaluator_proxy_class:
                raise AttributeError(
                    f"class [{name}] missing class attribute [cls_description], please specify it by "
                    f"doing like: `class {name}(evaluator_proxy_class=your_evaluator_proxy_class)`, where "
                    f"'your_evaluator_proxy_class' is the proxy class(a Process) of your evaluator class"
                )

        return new_cls

    def __init__(
        cls,
        name,
        bases,
        attrs,
        *,
        metric_definitions: List[Union[CompareMetricDefinition, MetricDefinition]] = None,
        cls_description: str = None
    ):
        super().__init__(name, bases, attrs)

    def __setattr__(self, key, value):
        # make sure those class attributes immutable in class-wise
        if key in ["metric_definitions", "cls_description", "obj_for_import"] and hasattr(self, key):
            raise AttributeError(f"class attribute {key} is immutable")
        return super().__setattr__(key, value)

    def __delattr__(self, item):
        # make sure those class attributes can't be deleted
        if item in ["metric_definitions", "cls_description", "obj_for_import"] and hasattr(self, item):
            raise AttributeError(f"class attribute [{item}] can't be deleted")
        return super().__delattr__(item)


class MetricEvaluatorMetadata(BaseModel):
    cls_name: str = Field(default=...)
    description: str = Field(default=...)
    config_schema: dict = Field(default=...)
    obj_for_import: DynamicObject = Field(default=...)
    metrics: List[str] = Field(default=...)


class MetricEvaluatorState(Enum):
    PENDING = "pending"
    INITIALIZING = "initializing"
    INIT_FAILED = "init failed"
    RUNNING = "running"
    FINISHED = "finished"
    TERMINATED = "terminated"


class MetricEvaluatorConfig(_Config):
    pass


class MetricEvaluator(_Configurable, ABC, metaclass=MetricEvaluatorMetaClass):
    config_cls: Type[MetricEvaluatorConfig] = MetricEvaluatorConfig
    config: config_cls

    # class attrs that are immutable
    metric_definitions: List[Union[CompareMetricDefinition, MetricDefinition]]
    cls_description: str
    evaluator_proxy_class: Type[MetricEvaluatorProxy]
    obj_for_import: DynamicObject

    def __init__(
        self,
        config: config_cls,
        scene_config: SceneConfig,
        logger: Logger,
        reporter: MetricReporter
    ):
        super().__init__(config=config)

        self.config_data = self.config.model_dump(mode="json", by_alias=True)

        metric_name2def = {metric_def.belonged_chain: metric_def for metric_def in self.metric_definitions}
        metric_name2conf = {
            metric_def.belonged_chain: scene_config.get_metric_config(metric_def.belonged_chain)
            for metric_def in self.metric_definitions
        }
        self.metric_name2metric_defs = {
            metric_name: metric_def for metric_name, metric_def in metric_name2def.items()
            if metric_name2conf[metric_name].enable
        }
        self.resp_msg_type2metric_defs = defaultdict(list)
        for metric in self.metric_name2metric_defs.values():
            self.resp_msg_type2metric_defs[metric.expect_resp_msg_type].append(metric)

        self.metrics_for_record = [
            metric_def.belonged_chain for metric_def in self.metric_name2metric_defs.values()
            if not metric_def.is_comparison
        ]
        self.metrics_for_compare = [
            metric_def.belonged_chain for metric_def in self.metric_name2metric_defs.values()
            if metric_def.is_comparison
        ]

        self.logger = logger
        self.reporter = reporter

        self.proxy = self.evaluator_proxy_class(
            config_cls=self.config_cls,
            config_data=self.config_data,
            record_metrics=self.metrics_for_record,
            compare_metrics=self.metrics_for_compare,
            manager=Manager()
        )

    @staticmethod
    @abstractmethod
    def _init_evaluator(
        config: MetricEvaluatorConfig,
        record_metrics: List[_MetricName],
        compare_metrics: List[_MetricName]
    ) -> Any:
        pass

    @staticmethod
    @abstractmethod
    async def _record(log: ActionLogBody, evaluator: Any) -> Dict[_MetricName, RecordOutput]:
        pass

    @staticmethod
    @abstractmethod
    async def _compare(log: ActionLogBody, evaluator: Any) -> Dict[_MetricName, CompareOutput]:
        pass

    def notify_to_record(self, log: ActionLogBody):
        Thread(target=self.record, args=(log,), daemon=True).start()

    def notify_to_compare(self, log: ActionLogBody):
        Thread(target=self.compare, args=(log,), daemon=True).start()

    def notify_can_stop(self):
        self.proxy.can_stop = True

    def start(self):
        self.proxy.start()

    def terminate(self):
        self.proxy.terminate()

    def join(self):
        self.proxy.join()

    def _wait_result(self, log: ActionLogBody, is_compare: bool = False):
        id_ = uuid4()
        self.proxy.queue.put_nowait((pickle.dumps(log), is_compare, id_))
        while id_ not in self.proxy.result_cache:
            time.sleep(0.1)  # sleep longer to let scene's main process have more CPU time slice
        return {
            k: (CompareOutput if is_compare else RecordOutput)(**v) for k, v in self.proxy.result_cache.pop(id_).items()
        }

    def record(self, log: ActionLogBody) -> None:
        resp_type = type(log.response)
        if resp_type not in self.resp_msg_type2metric_defs:
            return
        target_agent = log.response.sender_id
        # this may very slow
        record_results = self._wait_result(log)
        records = {}
        for metric_name, record_output in record_results.items():
            if metric_name not in self.metrics_for_record:
                continue

            record_value = record_output.record_value
            reason = record_output.reason
            misc = record_output.misc

            metric_def = self.metric_name2metric_defs[metric_name]
            expect_dtype = metric_def.record_value_dtype

            # validate value dtype
            if not validate_type(record_value, VALUE_DETYPE_2_DEFAULT_VALUE[expect_dtype]):
                continue
            # save record data
            _, record_data_model = metric_def.create_data_models()
            record_data = record_data_model(
                value=record_value,
                reason=reason,
                misc=misc,
                target_agent=target_agent,
                evaluator=self.__class__.__name__
            )
            self.reporter.put_record(record_data, metric_def.belonged_chain)
            records[metric_name] = record_data.model_dump(mode="json")
        self.logger.add_action_log_record(log_id=log.id, records=records, field_name="eval_records")

    def compare(self, log: ActionLogBody) -> None:
        resp_type = type(log.response)
        if resp_type not in self.resp_msg_type2metric_defs:
            return

        # this may very, very slow
        compare_results = self._wait_result(log, is_compare=True)
        records = {}
        for metric_name, compare_output in compare_results.items():
            if metric_name not in self.metrics_for_compare:
                continue

            compare_result = compare_output.compare_result
            reason = compare_output.reason
            misc = compare_output.misc

            metric_def = self.metric_name2metric_defs[metric_name]

            # validate value dtype
            if not validate_type(compare_result, List[List[str]]):
                continue
            # save record data
            _, record_data_model = metric_def.create_data_models()
            record_data = record_data_model(
                value=compare_result,
                reason=reason,
                misc=misc,
                evaluator=self.__class__.__name__
            )
            self.reporter.put_record(record_data, metric_def.belonged_chain)
            records[metric_name] = record_data.model_dump(mode="json")
        self.logger.add_action_log_record(log_id=log.id, records=records, field_name="compare_records")

    @classmethod
    def get_metadata(cls) -> MetricEvaluatorMetadata:
        return MetricEvaluatorMetadata(
            cls_name=cls.__name__,
            description=cls.cls_description,
            config_schema=cls.config_cls.get_json_schema(by_alias=True),
            obj_for_import=cls.obj_for_import,
            metrics=[metric_def.belonged_chain for metric_def in cls.metric_definitions]
        )

    @classmethod
    def from_config(cls, config: config_cls) -> "MetricEvaluator":
        raise NotImplementedError()

    @classmethod
    def from_config_file(cls, file_path: str) -> "MetricEvaluator":
        raise NotImplementedError()

    def save_config(self, file_path: str):
        raise NotImplementedError()


__all__ = [
    "RecordOutput",
    "CompareOutput",
    "MetricEvaluatorConfig",
    "MetricEvaluatorProxy",
    "MetricEvaluator",
    "MetricEvaluatorMetadata"
]
