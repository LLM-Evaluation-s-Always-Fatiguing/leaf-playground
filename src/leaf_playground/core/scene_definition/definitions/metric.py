from enum import Enum
from inspect import signature, Signature, Parameter
from functools import partial
from hashlib import md5
from sys import _getframe
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union
from uuid import UUID

from pydantic import create_model, field_serializer, BaseModel, Field, PrivateAttr

from ...._config import _Config
from ....data.base import Data
from ....data.message import Message
from ....utils.import_util import DynamicFn


_METRIC_MODELS = {}


class ValueDType(Enum):
    SCALAR = "scalar"
    VECTOR = "vector"
    NESTED_SCALAR = "nested_scalar"
    NESTED_VECTOR = "nested_vector"


MetricType2Annotation = {
    ValueDType.SCALAR: Union[bool, int, float],
    ValueDType.VECTOR: List[Union[bool, int, float, UUID]],
    ValueDType.NESTED_SCALAR: Dict[str, Union[bool, int, float]],
    ValueDType.NESTED_VECTOR: Dict[str, Union[List[bool], List[int], List[float], List[UUID]]]
}
MetricType2DefaultValue = {
    ValueDType.SCALAR: 0,
    ValueDType.VECTOR: [],
    ValueDType.NESTED_SCALAR: {},
    ValueDType.NESTED_VECTOR: {}
}


class _RecordData(Data):
    value: Any = Field(default=...)
    evaluator: str = Field(default=...)
    reason: Optional[str] = Field(default=None)
    misc: Optional[dict] = Field(default=None)
    is_comparison: bool = Field(default=False)
    target_agent: Optional[UUID] = Field(default=None)


class _MetricData(Data):
    value: Any = Field(default=...)
    records: List[_RecordData] = Field(default=...)
    is_comparison: bool = Field(default=False)
    target_agent: Optional[UUID] = Field(default=None)


class AggregationMethodOutput(BaseModel):
    value: Any = Field(default=...)
    records: List[_RecordData] = Field(default=...)


class DynamicAggregationFn(DynamicFn):
    @classmethod
    def create_dynamic_fn(cls, fn: Type, default_kwargs: Optional[dict] = None) -> "DynamicAggregationFn":
        new_fn = fn
        new_fn_signature = signature(new_fn)
        if default_kwargs:
            new_fn = partial(fn, **default_kwargs)
            new_fn_signature = signature(new_fn)
            new_fn_signature = new_fn_signature.replace(
                parameters=[p for name, p in new_fn_signature.parameters.items() if name not in default_kwargs]
            )

        required_signature = Signature(
            parameters=[Parameter(name="records", kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=List[_RecordData])],
            return_annotation=AggregationMethodOutput
        )

        if required_signature != new_fn_signature:
            raise TypeError(
                f"expected signature is {required_signature}, "
                f"got {new_fn_signature} which originally is {signature(fn)}"
            )

        dynamic_fn = super().create_dynamic_fn(fn, default_kwargs)
        return cls(**dynamic_fn.model_dump(mode="json", by_alias=True))


DefaultAggregateMethods = Literal["mean", "min", "max", "median", "sum"]
DynamicAggregationMethods = Union[DefaultAggregateMethods, DynamicAggregationFn]


class MetricDefinition(BaseModel):
    name: str = Field(default=...)
    description: str = Field(default=...)
    record_dtype: ValueDType = Field(default=...)
    metric_dtype: ValueDType = Field(default=...)
    expect_resp_msg_type: Type = Field(default=...)
    aggregation_methods: Dict[str, DynamicAggregationMethods] = Field(default={"-": "mean"})
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
        if self.is_comparison and self.metric_dtype in [ValueDType.SCALAR, ValueDType.NESTED_SCALAR]:
            raise ValueError(
                f"metric_dtype should be one of [MetricType.VECTOR, MetricType.NESTED_VECTOR], got {self.metric_dtype}"
            )

    @field_serializer("expect_resp_msg_type")
    def serialize_valid_msg_types(self, expect_resp_msg_type: Type, _info):
        return expect_resp_msg_type.__name__

    def get_record_value_annotation(self):
        return MetricType2Annotation[self.record_dtype]

    def get_metric_value_annotation(self):
        return MetricType2Annotation[self.metric_dtype]

    def create_data_models(self) -> Tuple[Type[_MetricData], Type[_RecordData]]:
        hash_id = md5(self.model_dump_json().encode()).hexdigest()
        if hash_id in _METRIC_MODELS:
            return _METRIC_MODELS[hash_id]

        base_name = "".join([s.capitalize() for s in self.name.split("_")])
        record_model_name = base_name + ("ComparisonRecord" if self.is_comparison else "MetricRecord")
        metric_model_name = base_name + ("Comparison" if self.is_comparison else "Metric")
        module = _getframe(1).f_globals["__name__"]

        record_value_annotation = self.get_record_value_annotation()
        metric_value_annotation = self.get_metric_value_annotation()

        record_model_fields = {
            "value": (record_value_annotation, Field(default=MetricType2DefaultValue[self.metric_dtype])),
            "is_comparison": (Literal[self.is_comparison], Field(default=self.is_comparison)),
        }
        if not self.is_comparison:
            record_model_fields.update(target_agent=(UUID, Field(default=...)))
        else:
            record_model_fields.update(target_agent=(Literal[None], Field(default=None)))

        record_model = create_model(
            __model_name=record_model_name,
            __module__=module,
            __base__=_RecordData,
            **record_model_fields
        )

        metric_model_fields = {
            "value": (metric_value_annotation, Field(default=MetricType2DefaultValue[self.metric_dtype])),
            "records": (List[record_model], Field(default=...)),
            "is_comparison": (Literal[self.is_comparison], Field(default=self.is_comparison)),
        }
        if not self.is_comparison:
            record_model_fields.update(target_agent=(UUID, Field(default=...)))
        else:
            record_model_fields.update(target_agent=(Literal[None], Field(default=None)))

        metric_model = create_model(
            __model_name=metric_model_name,
            __module__=module,
            __base__=_MetricData,
            **metric_model_fields
        )

        _METRIC_MODELS[hash_id] = (metric_model, record_model)

        return metric_model, record_model


class MetricConfig(_Config):
    enable: bool = Field(default=True)


__all__ = [
    "AggregationMethodOutput",
    "DynamicAggregationFn",
    "ValueDType",
    "MetricType2Annotation",
    "MetricDefinition",
    "MetricConfig"
]
