from abc import abstractmethod
from enum import Enum
from typing import Callable, Dict, List, Optional, Type
from uuid import UUID

from pydantic import field_serializer, Field

from ....data.base import Data
from ....data.message import MessageType
from ...._config import _Config


class MetricRecord(Data):
    value: float = Field(default=...)
    comment: Optional[str] = Field(default=None)
    misc: Optional[dict] = Field(default=None)
    target_agent: UUID = Field(default=...)

    @classmethod
    @abstractmethod
    async def calculate(
        cls,
        target: MessageType,
        evaluator: "leaf_playground.core.scene_evaluator.SceneEvaluator"
    ) -> "MetricRecord":
        pass


class Metric(Data):
    value: float = Field(default=...)
    records: List[MetricRecord] = Field(default=...)
    misc: Optional[dict] = Field(default=None)

    @classmethod
    @abstractmethod
    async def calculate(
        cls,
        records: List[MetricRecord],
        evaluator: "leaf_playground.core.scene_evaluator.SceneEvaluator"
    ) -> "Metric":
        pass


class MetricConfig(_Config):
    metric_name: str = Field(default=...)
    metric_description: str = Field(default=...)
    metric_type: Type[Metric] = Field(default=...)
    metric_record_type: Type[MetricRecord] = Field(default=...)
    chart_type: Optional[Type] = Field(default=None)
    reports_agg_method: Optional[Callable[[List[float]], float]] = Field(default=None)

    @field_serializer("metric_type")
    def serialize_metric_type(self, metric_type: Type[Metric], _info) -> str:
        return metric_type.__name__

    @field_serializer("metric_record_type")
    def serialize_metric_record_type(self, metric_record_type: Type[MetricRecord], _info) -> str:
        return metric_record_type.__name__

    @field_serializer("chart_type")
    def serialize_chart_type(self, chart_type: Optional[Type], _info) -> str:
        return chart_type.__name__ if chart_type else None

    @field_serializer("reports_agg_method")
    def serialize_reports_agg_method(self, reports_agg_method: Optional[Callable[[List[float]], float]], _info) -> str:
        return reports_agg_method.__name__ if reports_agg_method else None


class NestedMetricRecord(Data):
    value: Dict[str, float] = Field(default=...)
    misc: Optional[dict] = Field(default=None)
    target_agent: UUID = Field(default=...)

    @classmethod
    async def _calculate(
        cls,
        target: MessageType,
        evaluator: "leaf_playground.core.scene_evaluator.SceneEvaluator"
    ) -> "NestedMetricRecord":
        raise NotImplemented("Must implement _calculate method in subclass if not provide metric_record_types.")

    @classmethod
    async def calculate(
        cls,
        target: MessageType,
        evaluator: "leaf_playground.core.scene_evaluator.SceneEvaluator",
        metric_record_types: Optional[Dict[str, Type[MetricRecord]]] = None,
    ) -> "NestedMetricRecord":
        if not metric_record_types:
            return await cls._calculate(target, evaluator)
        else:
            records = {
                metric_name: await metric_type.calculate(target, evaluator)
                for metric_name, metric_type in metric_record_types.items()
            }
            misc = {}
            for name, record in records.items():
                misc[name + "_misc"] = record.misc

            return cls(
                value={
                    name: record.value
                    for name, record in records.items()
                },
                target_agent=target.agent_id,
                misc=misc or None
            )


class NestedMetric(Data):
    value: Dict[str, float] = Field(default=...)
    records: List[NestedMetricRecord] = Field(default=...)
    misc: Optional[dict] = Field(default=None)

    @classmethod
    @abstractmethod
    async def calculate(
        cls,
        records: List[NestedMetricRecord],
        evaluator: "leaf_playground.core.scene_evaluator.SceneEvaluator"
    ):
        pass


class NestedMetricConfig(_Config):
    metric_name: str = Field(default=...)
    metric_description: str = Field(default=...)
    metric_type: Type[NestedMetric] = Field(default=...)
    metric_record_type: Type[NestedMetricRecord] = Field(default=...)
    metric_sub_record_types: Optional[Dict[str, Type[MetricRecord]]] = Field(default=None)
    chart_type: Optional[Type] = Field(default=None)
    reports_agg_method: Optional[Callable[[List[float]], float]] = Field(default=None)

    @field_serializer("metric_type")
    def serialize_metric_type(self, metric_type: Type[NestedMetric], _info) -> str:
        return metric_type.__name__

    @field_serializer("metric_record_type")
    def serialize_metric_record_type(self, metric_record_type: Type[NestedMetricRecord], _info) -> str:
        return metric_record_type.__name__

    @field_serializer("metric_sub_record_types")
    def serialize_metric_record_types(
        self,
        metric_sub_record_types: Optional[Dict[str, Type[MetricRecord]]],
        _info
    ) -> Optional[Dict[str, str]]:
        return {
            name: metric_type.__name__ for name, metric_type in metric_sub_record_types.items()
        } if metric_sub_record_types else None

    @field_serializer("chart_type")
    def serialize_chart_type(self, chart_type: Optional[Type], _info) -> str:
        return chart_type.__name__ if chart_type else None

    @field_serializer("reports_agg_method")
    def serialize_reports_agg_method(self, reports_agg_method: Optional[Callable[[List[float]], float]], _info) -> str:
        return reports_agg_method.__name__ if reports_agg_method else None


class Comparison(Data):
    ranking: List[UUID] = Field(default=...)
    misc: Optional[dict] = Field(default=None)

    @classmethod
    @abstractmethod
    async def compare(
        cls,
        candidates: List[MessageType],
        compare_guidance: str,
        evaluator: "leaf_playground.core.scene_evaluator.SceneEvaluator"
    ) -> "Comparison":
        pass


class ComparisonMetric(Data):
    value: Dict[UUID, float] = Field(default=...)
    records: List[Comparison] = Field(default=...)
    misc: Optional[dict] = Field(default=None)

    @classmethod
    @abstractmethod
    async def calculate(
        cls,
        records: List[Comparison],
        evaluator: "leaf_playground.core.scene_evaluator.SceneEvaluator"
    ) -> "ComparisonMetric":
        pass


class ComparisonConfig(_Config):
    comparison_name: str = Field(default=...)
    comparison_description: str = Field(default=...)
    comparison_type: Type[Comparison] = Field(default=...)
    metric_type: Type[ComparisonMetric] = Field(default=...)
    compare_guidance: Optional[str] = Field(default=None)
    chart_type: Optional[Type] = Field(default=None)
    reports_agg_method: Optional[Callable[[List[float]], float]] = Field(default=None)

    @field_serializer("comparison_type")
    def serialize_comparison_type(self, comparison_type: Type[Comparison], _info) -> str:
        return comparison_type.__name__

    @field_serializer("metric_type")
    def serialize_metric_type(self, metric_type: Type[ComparisonMetric], _info) -> str:
        return metric_type.__name__

    @field_serializer("chart_type")
    def serialize_chart_type(self, chart_type: Optional[Type], _info) -> str:
        return chart_type.__name__ if chart_type else None

    @field_serializer("reports_agg_method")
    def serialize_reports_agg_method(self, reports_agg_method: Optional[Callable[[List[float]], float]], _info) -> str:
        return reports_agg_method.__name__ if reports_agg_method else None


class MetricTypes(Enum):
    METRIC = "metric"
    NESTED_METRIC = "nested_metric"
    COMPARISON = "comparison"


__all__ = [
    "MetricRecord",
    "Metric",
    "MetricConfig",
    "NestedMetricRecord",
    "NestedMetric",
    "NestedMetricConfig",
    "Comparison",
    "ComparisonMetric",
    "ComparisonConfig",
    "MetricTypes"
]
