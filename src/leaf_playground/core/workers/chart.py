from abc import abstractmethod, ABC, ABCMeta
from sys import _getframe
from typing import List, Optional

from pydantic import BaseModel, Field

from ..._type import Immutable
from ..scene_definition import CombinedMetricsData, SceneConfig
from ...data.log_body import LogBody
from ...utils.import_util import DynamicObject
from ...utils.type_util import validate_type


_chart_names = set()


class ChartMetaClass(ABCMeta):
    def __new__(
            cls,
            name,
            bases,
            attrs,
            *,
            chart_name: str = None,
            supported_metric_names: List[str] = None,
    ):
        if chart_name in _chart_names:
            raise ValueError(f"Chart name {chart_name} has been used!")
        _chart_names.add(chart_name)
        attrs["chart_name"] = Immutable(chart_name)
        attrs["supported_metric_names"] = Immutable(supported_metric_names)
        attrs["obj_for_import"] = Immutable(DynamicObject(obj=name, module=_getframe(1).f_globals["__name__"]))

        new_cls = super().__new__(cls, name, bases, attrs)

        DynamicObject.bind_dynamic_obj(attrs["obj_for_import"], new_cls)

        if not validate_type(attrs["chart_name"], expect_type=Immutable[Optional[str]]):
            raise TypeError(
                f"class [{name}]'s class attribute [chart_name] should be a str"
            )
        if not validate_type(attrs["supported_metric_names"], expect_type=Immutable[Optional[List[str]]]):
            raise TypeError(
                f"class [{name}]'s class attribute [chart_name] should be a List[str]"
            )

        if ABC not in bases:
            # check if those class attrs are empty when the class is not abstract
            if not new_cls.chart_name:
                raise AttributeError(
                    f"class [{name}] missing class attribute [chart_name], please specify it by "
                    f"doing like: `class {name}(chart_name=your_chart_name)`, where 'your_chart_name' "
                    f"is a string that introduces your chart class"
                )
            if not new_cls.supported_metric_names:
                raise AttributeError(
                    f"class [{name}] missing class attribute [supported_metric_names], please specify it by "
                    f"doing like: `class {name}(supported_metric_names=your_supported_metric_names)`, where "
                    f"'your_supported_metric_names' is a list of string that introduces the metric names "
                    f"supported by your chart class"
                )
        return new_cls

    def __init__(
            cls,
            name,
            bases,
            attrs,
            *,
            chart_name: str = None,
            supported_metric_names: List[str] = None,
    ):
        super().__init__(name, bases, attrs)


class ChartMetadata(BaseModel):
    cls_name: str = Field(default=...)
    obj_for_import: DynamicObject = Field(default=...)
    chart_name: str = Field(default=...)
    supported_metric_names: List[str] = Field(default=...)


class Chart(ABC, metaclass=ChartMetaClass):
    obj_for_import: DynamicObject
    supported_metric_names: List[str]
    chart_name: str

    @classmethod
    def get_metadata(cls) -> ChartMetadata:
        return ChartMetadata(
            cls_name=cls.__name__,
            obj_for_import=cls.obj_for_import,
            chart_name=cls.chart_name,
            supported_metric_names=cls.supported_metric_names
        )

    def generate(
        self,
        metrics: CombinedMetricsData,
        scene_config: SceneConfig,
        evaluator_configs: List["leaf_playground.core.workers.MetricEvaluatorConfig"],
        logs: List[LogBody]
    ) -> Optional[dict]:
        if not metrics:
            return None

        filtered_metrics: CombinedMetricsData = {
            "metrics": {
                k: v
                for k, v in metrics["metrics"].items()
                if k in self.supported_metric_names
            },
            "human_metrics": {
                k: v
                for k, v in metrics["human_metrics"].items()
                if k in self.supported_metric_names
            }
        }
        try:
            return self._generate(filtered_metrics, scene_config, evaluator_configs, logs)
        except Exception as e:
            print(f"Error occurred when generating chart {self.__class__.__name__}: {e}")
            return None

    @abstractmethod
    def _generate(
        self,
        metrics: CombinedMetricsData,
        scene_config: SceneConfig,
        evaluator_configs: List["leaf_playground.core.workers.MetricEvaluatorConfig"],
        logs: List[LogBody]
    ) -> dict:
        pass


__all__ = [
    "Chart",
    "ChartMetadata"
]
