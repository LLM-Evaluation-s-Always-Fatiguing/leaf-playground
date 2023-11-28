import asyncio
import json
from uuid import UUID
from collections import defaultdict
from typing import Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field

from .._config import _Config, _Configurable
from ..data.base import Data
from ..data.message import MessageType
from ..data.metric import Comparison, ComparisonMetric, ComparisonConfig, MetricRecord, Metric, MetricConfig
from ..utils.import_util import DynamicObject
from ..utils.thread_util import run_asynchronously


MetricName = str
ComparisonName = str


class SceneEvaluatorRecord(Data):
    evaluator: str = Field(default=...)
    records: Dict[MetricName, MetricRecord] = Field(default=...)


class SceneEvaluatorCompare(Data):
    evaluator: str = Field(default=...)
    comparisons: Dict[ComparisonName, Comparison] = Field(default=...)


class SceneEvaluatorReport(Data):
    evaluator: str = Field(default=...)
    report: Dict[UUID, Dict[Union[ComparisonName, MetricName], Union[Metric, ComparisonMetric]]] = Field(default=...)


class SceneEvaluatorMetadata(BaseModel):
    cls_name: str = Field(default=...)
    description: str = Field(default=...)
    metrics: List[dict] = Field(default=...)
    comparisons: List[dict] = Field(default=...)


class SceneEvaluatorConfig(_Config):
    pass


class SceneEvaluator(_Configurable):
    config_obj = SceneEvaluatorConfig
    config: config_obj

    description: str
    obj_for_import: DynamicObject

    _metric_configs: Optional[List[MetricConfig]]
    _comparison_configs: Optional[List[ComparisonConfig]]
    _target_type: Type[MessageType]
    _record_type: Type[SceneEvaluatorRecord] = SceneEvaluatorRecord
    _compare_type: Type[SceneEvaluatorCompare] = SceneEvaluatorCompare
    _report_type: Type[SceneEvaluatorReport] = SceneEvaluatorReport

    def __init__(self, config: config_obj):
        if hasattr(self, f"_{self.__class__.__name__}__valid_class_attributes"):
            getattr(self, f"_{self.__class__.__name__}__valid_class_attributes")()
        self.__valid_class_attributes()
        super().__init__(config=config)
        self._name2records: Dict[str, List[MetricRecord]] = defaultdict(list)
        self._name2comparisons: Dict[str, List[Comparison]] = defaultdict(list)

    def post_init(self, *args, **kwargs):
        pass

    def __valid_class_attributes(self):
        if not hasattr(self, "description"):
            raise AttributeError(f"class attribute description not found, must specify in your evaluator class")
        if not hasattr(self, "_metric_configs"):
            raise AttributeError(f"class attribute _metrics not found, must specify in your evaluator class")
        if not hasattr(self, "_comparison_configs"):
            raise AttributeError(f"class attribute _comparisons not found, must specify in your evaluator class")
        metric_names = []
        if self._metric_configs:
            metric_names += [m.metric_name for m in self._metric_configs]
        if self._comparison_configs:
            metric_names += [m.comparison_name for m in self._comparison_configs]
        if len(set(metric_names)) != len(metric_names):
            raise ValueError(f"name of each metric should be unique")
        if not hasattr(self, "_target_type"):
            raise AttributeError(f"class attribute _target_type not found, must specify in your evaluator class")
        if self.__class__.__name__ != self.obj_for_import.obj:
            raise ValueError(
                f"obj_for_import isn't correct, should be {self.__class__.__name__}, got {self.obj_for_import.obj}"
            )

    async def record(self, target: MessageType) -> Optional[_record_type]:
        async def _record():
            records = {}

            async def calculate(metric_config: MetricConfig):
                record = await run_asynchronously(metric_config.metric_record_type.calculate, target, self)
                records[metric_config.metric_name] = record
                self._name2records[metric_config.metric_name].append(record)

            await asyncio.gather(*[calculate(metric_config) for metric_config in self._metric_configs])

            return self._record_type(evaluator=self.__class__.__name__, records=records)

        if not self._metric_configs or target.__class__ != self._target_type:  # exact match, not allow subclass
            return None
        return await _record()

    async def compare(self, candidates: List[MessageType]) -> Optional[_compare_type]:
        async def _compare():
            comparisons = {}

            async def calculate(comparison_config: ComparisonConfig):
                comparison = await run_asynchronously(
                    comparison_config.comparison_type.compare, candidates, comparison_config.guidance, self
                )
                comparisons[comparison_config.comparison_name] = comparison
                self._name2comparisons[comparison_config.comparison_name].append(comparison)

            await asyncio.gather(*[calculate(comparison_config) for comparison_config in self._comparison_configs])

            return self._compare_type(evaluator=self.__class__.__name__, comparisons=comparisons)

        if not self._comparison_configs or not all(cand.__class__ == self._target_type for cand in candidates):
            return None
        return await _compare()

    async def report(self) -> Optional[_report_type]:

        async def _report():
            agent2metrics = defaultdict(dict)

            async def calculate(metric_name: str):
                all_records = self._name2records.get(metric_name, self._name2comparisons.get(metric_name, []))
                if not all_records:
                    return
                agent2records = defaultdict(list)
                for record in all_records:
                    agent2records[record.target_agent].append(record)

                async def _calculate(agent: UUID):
                    agent2metrics[agent][metric_name] = await run_asynchronously(
                        name2metric_type[metric_name].calculate, agent2records[agent], self
                    )

                await asyncio.gather(
                    *[_calculate(agent) for agent in agent2records]
                )

            name2metric_type = {}
            if self._metric_configs:
                for config in self._metric_configs:
                    name2metric_type[config.metric_name] = config.metric_type
            if self._comparison_configs:
                for config in self._comparison_configs:
                    name2metric_type[config.comparison_name] = config.metric_type

            await asyncio.gather(
                *[calculate(metric_name) for metric_name in name2metric_type]
            )

            return self._report_type(evaluator=self.__class__.__name__, report=agent2metrics)

        if not self._name2records and not self._name2comparisons:
            return None
        return await _report()

    @classmethod
    def from_config(cls, config: config_obj) -> "SceneEvaluator":
        return cls(config=config)

    @classmethod
    def get_metadata(cls) -> SceneEvaluatorMetadata:
        return SceneEvaluatorMetadata(
            cls_name=cls.__name__,
            description=cls.description,
            metrics=[
                json.loads(metric.model_dump_json()) for metric in (cls._metric_configs or [])
            ],
            comparisons=[
                json.loads(comparison.model_dump_json()) for comparison in (cls._comparison_configs or [])
            ],
        )


__all__ = [
    "SceneEvaluatorRecord",
    "SceneEvaluatorCompare",
    "SceneEvaluatorReport",
    "SceneEvaluatorMetadata",
    "SceneEvaluatorConfig",
    "SceneEvaluator",
]