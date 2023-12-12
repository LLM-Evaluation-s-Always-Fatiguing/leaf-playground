import asyncio
import pickle
import time
from abc import abstractmethod, ABC, ABCMeta
from collections import defaultdict
from enum import Enum
from multiprocessing import Manager, Process
from queue import Empty
from sys import _getframe
from threading import Thread
from typing import Any, Dict, List, Optional, Type, Union
from uuid import uuid4

from pydantic import BaseModel, Field

from ..scene_definition import MetricDefinition, MetricType2Annotation, SceneConfig
from ..._config import _Config, _Configurable
from ..._type import Immutable
from ...data.log_body import LogBody
from ...data.socket_data import SocketData, SocketDataType, SocketOperation
from ...utils.import_util import DynamicObject
from ...utils.thread_util import run_asynchronously
from ...utils.type_util import validate_type


_MetricName = str
_MetricRecordValue = Any


class MetricEvaluatorProxy(Process):
    def __init__(
        self,
        config_cls: Type["MetricEvaluatorConfig"],
        config_data: dict,
        manager: Manager
    ):
        super().__init__(daemon=True)

        self._queue = manager.Queue()
        self._result_cache = manager.dict()
        self._can_stop = manager.Value("b", False)
        self._state = manager.Value("c", MetricEvaluatorState.PENDING.value.encode("utf-8"))

        self._config_cls = config_cls
        self._config_data = config_data

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

    @abstractmethod
    def _init_evaluator(self, config) -> Any:
        pass

    def run(self):
        self._state.value = MetricEvaluatorState.INITIALIZING.value.encode("utf-8")
        try:
            evaluator = self._init_evaluator(self._config_cls(**self._config_data))
        except Exception as e:
            self._state.value = MetricEvaluatorState.INIT_FAILED.value.encode("utf-8")
            return
        loop = asyncio.new_event_loop()
        self._state.value = MetricEvaluatorState.RUNNING.value.encode("utf-8")

        async def gather(data, methods):
            return await asyncio.gather(*[run_asynchronously(method, data, evaluator) for method in methods])

        while True:
            try:
                log_data, is_compare, id_ = self._queue.get_nowait()
            except Empty:
                time.sleep(0.001)
                continue
            try:
                if not is_compare:
                    log = LogBody(**log_data)
                    results = loop.run_until_complete(
                        gather(log, [self._cal_record_value, self._comment_record, self._collect_record_misc])
                    )
                else:
                    logs = [LogBody(**each) for each in log_data]
                    results = loop.run_until_complete(
                        gather(logs, [self._cal_compare_value, self._comment_compare, self._collect_compare_misc])
                    )
            except:
                results = ({}, None, None)
            self._result_cache[id_] = pickle.dumps(results)

            if self._queue.empty() and self._can_stop.value:
                break

        self._state.value = MetricEvaluatorState.FINISHED.value.encode("utf-8")

    def terminate(self):
        super().terminate()
        self._state.value = MetricEvaluatorState.TERMINATED.value.encode("utf-8")

    @abstractmethod
    async def _cal_record_value(self, log: LogBody, evaluator: Any) -> Dict[_MetricName, _MetricRecordValue]:
        pass

    @abstractmethod
    async def _comment_record(self, log: LogBody, evaluator: Any) -> Optional[Dict[_MetricName, str]]:
        pass

    @abstractmethod
    async def _collect_record_misc(self, log: LogBody, evaluator: Any) -> Optional[Dict[_MetricName, dict]]:
        pass

    @abstractmethod
    async def _cal_compare_value(self, logs: List[LogBody], evaluator: Any) -> Dict[_MetricName, _MetricRecordValue]:
        pass

    @abstractmethod
    async def _comment_compare(self, logs: List[LogBody], evaluator: Any) -> Optional[Dict[_MetricName, str]]:
        pass

    @abstractmethod
    async def _collect_compare_misc(self, logs: List[LogBody], evaluator: Any) -> Optional[Dict[_MetricName, dict]]:
        pass


# TODO: We hope that the MetricEvaluator class can be inherited through composition
#  (or at least subcls can extend defs), however, using meta class makes this tough

class MetricEvaluatorMetaClass(ABCMeta):
    def __new__(
        cls,
        name,
        bases,
        attrs,
        *,
        metric_definitions: List[MetricDefinition] = None,
        cls_description: str = None,
        evaluator_proxy_class: Type["MetricEvaluatorProxy"] = None
    ):
        attrs["metric_definitions"] = Immutable(metric_definitions or getattr(bases[0], "metric_definitions", None))
        attrs["cls_description"] = Immutable(cls_description or "")
        attrs["evaluator_proxy_class"] = Immutable(evaluator_proxy_class)
        attrs["obj_for_import"] = Immutable(DynamicObject(obj=name, source_file=_getframe(1).f_code.co_filename))

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
        metric_definitions: List[MetricDefinition] = None,
        cls_description: str = None,
        evaluator_proxy_class: "MetricEvaluatorProxy" = None
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
    metric_definitions: List[MetricDefinition]
    cls_description: str
    evaluator_proxy_class: Type[MetricEvaluatorProxy]
    obj_for_import: DynamicObject

    def __init__(self, config: config_cls, scene_config: SceneConfig, socket_cache: List[SocketData]):
        super().__init__(config=config)

        self.config_data = self.config.model_dump(mode="json", by_alias=True)
        self.socket_cache = socket_cache

        metric_name2def = {metric_def.name: metric_def for metric_def in self.metric_definitions}
        metric_name2conf = {
            metric_def.name: scene_config.get_metric_config(metric_def.belonged_chain)
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
            metric_def.name for metric_def in self.metric_name2metric_defs.values() if not metric_def.is_comparison
        ]
        self.metrics_for_compare = [
            metric_def.name for metric_def in self.metric_name2metric_defs.values() if metric_def.is_comparison
        ]

        self.proxy = self.evaluator_proxy_class(
            config_cls=self.config_cls,
            config_data=self.config_data,
            manager=Manager()
        )

    def notify_to_record(self, log: LogBody):
        Thread(target=self.record, args=(log,), daemon=True).start()

    def notify_to_compare(self, logs: List[LogBody]):
        Thread(target=self.compare, args=(logs,), daemon=True).start()

    def notify_can_stop(self):
        self.proxy.can_stop = True

    def start(self):
        self.proxy.start()

    def terminate(self):
        self.proxy.terminate()

    def join(self):
        self.proxy.join()

    def _wait_result(self, logs: Union[LogBody, List[LogBody]]):
        id_ = uuid4()
        if isinstance(logs, LogBody):
            log_data = logs.model_dump(mode="json", by_alias=True)
            is_compare = False
        else:
            log_data = [log.model_dump(mode="json", by_alias=True) for log in logs]
            is_compare = True
        self.proxy.queue.put_nowait((log_data, is_compare, id_))
        while id_ not in self.proxy.result_cache:
            time.sleep(0.1)  # sleep longer to let scene's main process have more CPU time slice
        return pickle.loads(self.proxy.result_cache.pop(id_))

    def record(self, log: LogBody) -> None:
        resp_type = type(log.response)
        if resp_type not in self.resp_msg_type2metric_defs:
            return
        target_agent = log.response.sender_id
        # this may very slow
        metric_name2record_value, metric_name2comments, metric_name2misc = self._wait_result(log)
        for metric_name, record_value in metric_name2record_value.items():
            if metric_name not in self.metrics_for_record:
                raise TypeError(f"metric [{metric_name}] can't be used in record relevant methods")

            comment = metric_name2comments.get(metric_name, None)
            misc = metric_name2misc.get(metric_name, None)

            metric_def = self.metric_name2metric_defs[metric_name]
            expect_dtype = metric_def.metric_dtype

            # validate value dtype
            if not validate_type(record_value, MetricType2Annotation[expect_dtype]):
                raise TypeError(
                    f"metric [{metric_name}]'s dtype is [{expect_dtype}], got record_value: {record_value}"
                )
            # save record data
            _, record_data_model = metric_def.create_data_models()
            log.eval_records[metric_name].append(
                record_data_model(
                    value=record_value,
                    comment=comment,
                    misc=misc,
                    target_agent=target_agent,
                    evaluator=self.__class__.__name__
                ).model_dump(mode="json")
            )
            self.socket_cache.append(
                SocketData(
                    type=SocketDataType.LOG,
                    data=log.model_dump(mode="json"),
                    operation=SocketOperation.UPDATE
                )
            )

    def compare(self, logs: List[LogBody]) -> None:
        resp_type2logs = defaultdict(list)
        for log in logs:
            resp_type2logs[type(log.response)].append(log)
        for resp_type, logs_ in resp_type2logs.items():
            if resp_type not in self.resp_msg_type2metric_defs:
                continue

            # this may very, very slow
            metric_name2compare_value, metric_name2compare_comment, metric_name2compare_misc = self._wait_result(logs)
            for metric_name, compare_value in metric_name2compare_value.items():
                if metric_name not in self.metrics_for_compare:
                    raise TypeError(f"metric [{metric_name}] can't be used in compare relevant methods")

                compare_comment = metric_name2compare_comment.get(metric_name, None)
                compare_misc = metric_name2compare_misc.get(metric_name, None)

                metric_def = self.metric_name2metric_defs[metric_name]
                expect_dtype = metric_def.metric_dtype

                # validate value dtype
                if not validate_type(compare_value, MetricType2Annotation[expect_dtype]):
                    raise TypeError(
                        f"metric [{metric_name}]'s dtype is [{expect_dtype}], got record_value: {compare_value}"
                    )
                # save record data
                _, record_data_model = metric_def.create_data_models()
                record_data = record_data_model(
                    value=compare_value,
                    comment=compare_comment,
                    misc=compare_misc,
                    evaluator=self.__class__.__name__
                )
                for log in logs_:
                    log.eval_records[metric_name].append(record_data)
                    self.socket_cache.append(
                        SocketData(
                            type=SocketDataType.LOG,
                            data=log.model_dump(mode="json"),
                            operation=SocketOperation.UPDATE
                        )
                    )

    @classmethod
    def get_metadata(cls) -> MetricEvaluatorMetadata:
        return MetricEvaluatorMetadata(
            cls_name=cls.__name__,
            description=cls.cls_description,
            config_schema=cls.config_cls.get_json_schema(by_alias=True),
            obj_for_import=cls.obj_for_import
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
    "MetricEvaluatorConfig",
    "MetricEvaluatorProxy",
    "MetricEvaluator",
    "MetricEvaluatorMetadata"
]
