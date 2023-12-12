from enum import Enum
from hashlib import md5
from sys import _getframe
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union
from uuid import UUID

from pydantic import create_model, field_serializer, BaseModel, Field, PrivateAttr

from ...._config import _Config
from ....data.base import Data
from ....data.message import Message


_METRIC_MODELS = {}


class MetricType(Enum):
    SCALAR = "scalar"
    VECTOR = "vector"
    NESTED_SCALAR = "nested_scalar"
    NESTED_VECTOR = "nested_vector"


# TODO: redefine compare, as a standalone concept


MetricType2Annotation = {
    MetricType.SCALAR: Union[bool, int, float],
    MetricType.VECTOR: List[Union[bool, int, float, UUID]],
    MetricType.NESTED_SCALAR: Dict[str, Union[bool, int, float]],
    MetricType.NESTED_VECTOR: Dict[str, List[Union[bool, int, float, UUID]]]
}
MetricType2DefaultValue = {
    MetricType.SCALAR: 0,
    MetricType.VECTOR: [],
    MetricType.NESTED_SCALAR: None,
    MetricType.NESTED_VECTOR: None
}


class _MetricRecordData(Data):
    value: Any = Field(default=...)
    evaluator: str = Field(default=...)
    reason: Optional[str] = Field(default=None)
    misc: Optional[dict] = Field(default=None)
    is_comparison: bool = Field(default=False)
    target_agent: Optional[UUID] = Field(default=None)


class _MetricData(Data):
    value: Any = Field(default=...)
    evaluator: str = Field(default=...)
    records: List[_MetricRecordData] = Field(default=...)
    is_comparison: bool = Field(default=False)
    target_agent: Optional[UUID] = Field(default=None)


class MetricDefinition(BaseModel):
    name: str = Field(default=...)
    description: str = Field(default=...)
    metric_dtype: MetricType = Field(default=...)
    expect_resp_msg_type: Type = Field(default=...)
    is_comparison: bool = Field(default=False)

    _belonged_action: Optional[
        "leaf_playground.core.scene_info.definitions.action.ActionDefinition"
    ] = PrivateAttr(default=None)

    @property
    def belonged_action(self) -> "leaf_playground.core.scene_info.definitions.action.ActionDefinition":
        return self._belonged_action

    @property
    def belonged_chain(self):
        return self.belonged_action.belonged_chain + "." + self.name

    def model_post_init(self, __context: Any) -> None:
        if not self.expect_resp_msg_type:
            raise ValueError(f"valid_msg_types should not be empty")
        if not issubclass(self.expect_resp_msg_type, Message):
            raise TypeError(
                f"expect_resp_msg_type must be a subclass of Message"
            )
        if self.is_comparison and self.metric_dtype in [MetricType.SCALAR, MetricType.NESTED_SCALAR]:
            raise ValueError(
                f"metric_dtype should be one of [MetricType.VECTOR, MetricType.NESTED_VECTOR], got {self.metric_dtype}"
            )

    @field_serializer("expect_resp_msg_type")
    def serialize_valid_msg_types(self, expect_resp_msg_type: Type, _info):
        return expect_resp_msg_type.__name__

    def get_metric_value_annotation(self):
        return MetricType2Annotation[self.metric_dtype]

    def create_data_models(self) -> Tuple[Type[_MetricData], Type[_MetricRecordData]]:
        hash_id = md5(self.model_dump_json().encode()).hexdigest()
        if hash_id in _METRIC_MODELS:
            return _METRIC_MODELS[hash_id]

        base_name = "".join([s.capitalize() for s in self.name.split("_")])
        metric_model_name = base_name + ("Comparison" if self.is_comparison else "Metric")
        metric_record_model_name = base_name + ("ComparisonRecord" if self.is_comparison else "MetricRecord")
        module = _getframe(1).f_globals["__name__"]

        metric_value_annotation = self.get_metric_value_annotation()

        metric_record_model_fields = {
            "value": (metric_value_annotation, Field(default=MetricType2DefaultValue[self.metric_dtype])),
            "is_comparison": (Literal[self.is_comparison], Field(default=self.is_comparison)),
        }
        if not self.is_comparison:
            metric_record_model_fields.update(target_agent=(UUID, Field(default=...)))

        metric_record_model = create_model(
            __model_name=metric_record_model_name,
            __module__=module,
            __base__=_MetricRecordData,
            **metric_record_model_fields
        )

        metric_model_fields = {
            "value": (metric_value_annotation, Field(default=MetricType2DefaultValue[self.metric_dtype])),
            "records": (List[metric_record_model], Field(default=...)),
            "is_comparison": (Literal[self.is_comparison], Field(default=self.is_comparison)),
        }
        if not self.is_comparison:
            metric_record_model_fields.update(target_agent=(UUID, Field(default=...)))

        metric_model = create_model(
            __model_name=metric_model_name,
            __module__=module,
            __base__=_MetricData,
            **metric_model_fields
        )

        _METRIC_MODELS[hash_id] = (metric_model, metric_record_model)

        return metric_model, metric_record_model


class MetricConfig(_Config):
    enable: bool = Field(default=True)


__all__ = [
    "MetricType",
    "MetricType2Annotation",
    "MetricDefinition",
    "MetricConfig"
]
