from abc import abstractmethod
from typing import Any, Dict, List, Optional, Type

from pydantic import field_serializer, Field

from .base import Data
from .log_body import LogBody
from .profile import Profile
from .._config import _Config


class MetricRecord(Data):
    value: Any = Field(default=...)
    comment: Optional[str] = Field(default=None)
    target: int = Field(default=...)

    @classmethod
    @abstractmethod
    async def calculate(
        cls,
        target: LogBody,
        evaluator: "leaf_playground.core.scene_evaluator.SceneEvaluator"
    ) -> "MetricRecord":
        pass


class Metric(Data):
    value: Any = Field(default=...)
    records: List[MetricRecord] = Field(default=...)
    metadata: Optional[dict] = Field(default=None)

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

    @field_serializer("metric_type")
    def serialize_metric_type(self, metric_type: Type[Metric], _info) -> str:
        return metric_type.__name__

    @field_serializer("metric_record__type")
    def serialize_metric_type(self, metric_record__type: Type[Metric], _info) -> str:
        return metric_record__type.__name__


class Comparison(Data):
    ranking: List[Profile] = Field(default=...)
    targets: List[int] = Field(default=...)

    @classmethod
    @abstractmethod
    async def compare(
        cls,
        candidates: List[LogBody],
        compare_guidance: str,
        evaluator: "leaf_playground.core.scene_evaluator.SceneEvaluator"
    ) -> "Comparison":
        pass


class ComparisonMetric(Metric):
    value: Dict[str, float] = Field(default=...)
    records: List[Comparison] = Field(default=...)

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
    compare_guidance: str = Field(default=...)

    @field_serializer("comparison_type")
    def serialize_comparison_type(self, comparison_type: Type[Comparison], _info) -> str:
        return comparison_type.__name__

    @field_serializer("metric_type")
    def serialize_comparison_type(self, metric_type: Type[Comparison], _info) -> str:
        return metric_type.__name__


__all__ = [
    "MetricRecord",
    "Metric",
    "MetricConfig",
    "Comparison",
    "ComparisonConfig"
]
