from enum import Enum
from inspect import signature, Signature, Parameter
from functools import partial
from hashlib import md5
from sys import _getframe
from typing import Any, List, Literal, Optional, Tuple, Type, Union

from pydantic import create_model, field_serializer, BaseModel, Field, PrivateAttr

from ...._config import _Config
from ....data.base import Data
from ....data.message import Message
from ....utils.import_util import DynamicFn


_METRIC_MODELS = {}


class ValueDType(Enum):
    INT = int
    FLOAT = float
    BOOLEAN = bool


VALUE_DETYPE_2_DEFAULT_VALUE = {
    ValueDType.INT: 0,
    ValueDType.FLOAT: 0.0,
    ValueDType.BOOLEAN: False
}


class DisplayType(Enum):
    FIVE_STARTS_RATE = "FiveStarsRate"
    NUMBER_INPUT = "NumberInput"
    BOOLEAN_RADIO = "BooleanRadio"


class _RecordData(Data):
    value: Any = Field(default=...)
    evaluator: str = Field(default=...)
    display_type: DisplayType = Field(default=...)
    reason: Optional[str] = Field(default=None)
    misc: Optional[dict] = Field(default=None)
    is_comparison: bool = Field(default=False)
    target_agent: Optional[str] = Field(default=None)


class _MetricData(Data):
    value: Any = Field(default=...)
    records: List[_RecordData] = Field(default=...)
    is_comparison: bool = Field(default=False)
    target_agent: Optional[str] = Field(default=None)


class AggregationMethodOutput(BaseModel):
    value: Any = Field(default=...)


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


DefaultAggregateMethod = Literal["mean", "min", "max", "median", "sum"]
DynamicAggregationMethod = Union[DefaultAggregateMethod, DynamicAggregationFn]

DEFAULT_AGG_METHODS = {
    "mean": lambda x: AggregationMethodOutput(value=sum([each.value for each in x]) / len(x)),
    "min": lambda x: AggregationMethodOutput(value=min([each.value for each in x])),
    "max": lambda x: AggregationMethodOutput(value=max([each.value for each in x])),
    "median": lambda x: AggregationMethodOutput(value=sorted([each.value for each in x])[len(x) // 2]),
    "sum": lambda x: AggregationMethodOutput(value=sum([each.value for each in x]))
}


class CompareMetricDefinition(BaseModel):
    name: str = Field(default=...)
    description: str = Field(default=...)
    expect_resp_msg_type: Type = Field(default=...)
    is_comparison: Literal[True] = Field(default=True)

    _belonged_action: Optional[
        "leaf_playground.core.scene_info.definitions.action.ActionDefinition"
    ] = PrivateAttr(default=None)

    @property
    def belonged_action(self) -> "leaf_playground.core.scene_info.definitions.action.ActionDefinition":
        return self._belonged_action

    @property
    def belonged_chain(self):
        return self.belonged_action.belonged_chain + "." + self.name

    @field_serializer("expect_resp_msg_type")
    def serialize_valid_msg_types(self, expect_resp_msg_type: Type, _info):
        return expect_resp_msg_type.__name__

    def model_post_init(self, __context: Any) -> None:
        if not self.expect_resp_msg_type:
            raise ValueError(f"valid_msg_types should not be empty")
        if not issubclass(self.expect_resp_msg_type, Message):
            raise TypeError(
                f"expect_resp_msg_type must be a subclass of Message"
            )

    def create_data_models(self) -> Tuple[Type[_MetricData], Type[_RecordData]]:
        hash_id = md5(self.model_dump_json().encode()).hexdigest()
        if hash_id in _METRIC_MODELS:
            return _METRIC_MODELS[hash_id]

        base_name = "".join([s.capitalize() for s in self.name.split("_")])
        record_model_name = base_name + "ComparisonRecord"
        metric_model_name = base_name + "Comparison"
        module = _getframe(1).f_globals["__name__"]

        record_model_fields = {
            "value": (List[str], Field(default=[])),
            "display_type": (Literal[None], Field(default=None)),
            "is_comparison": (Literal[self.is_comparison], Field(default=self.is_comparison)),
            "target_agent": (Literal[None], Field(default=None))
        }

        record_model = create_model(
            __model_name=record_model_name,
            __module__=module,
            __base__=_RecordData,
            **record_model_fields
        )

        metric_model_fields = {
            "records": (List[record_model], Field(default=...)),
            "is_comparison": (Literal[self.is_comparison], Field(default=self.is_comparison)),
            "target_agent": (Literal[None], Field(default=None))
        }

        metric_model = create_model(
            __model_name=metric_model_name,
            __module__=module,
            __base__=_MetricData,
            **metric_model_fields
        )

        _METRIC_MODELS[hash_id] = (metric_model, record_model)

        return metric_model, record_model


class MetricDefinition(BaseModel):
    name: str = Field(default=...)
    description: str = Field(default=...)
    record_value_dtype: ValueDType = Field(default=...)
    record_display_type: DisplayType = Field(default=...)
    expect_resp_msg_type: Type = Field(default=...)
    agg_method: DynamicAggregationMethod = Field(default=...)
    is_comparison: Literal[False] = Field(default=False)

    _belonged_action: Optional[
        "leaf_playground.core.scene_info.definitions.action.ActionDefinition"
    ] = PrivateAttr(default=None)

    @property
    def belonged_action(self) -> "leaf_playground.core.scene_info.definitions.action.ActionDefinition":
        return self._belonged_action

    @property
    def belonged_chain(self) -> str:
        return self.belonged_action.belonged_chain + "." + self.name

    def model_post_init(self, __context: Any) -> None:
        if not self.expect_resp_msg_type:
            raise ValueError(f"valid_msg_types should not be empty")
        if not issubclass(self.expect_resp_msg_type, Message):
            raise TypeError(
                f"expect_resp_msg_type must be a subclass of Message"
            )

    @field_serializer("expect_resp_msg_type")
    def serialize_valid_msg_types(self, expect_resp_msg_type: Type, _info):
        return expect_resp_msg_type.__name__

    @field_serializer("record_value_dtype")
    def serialize_record_value_dtype(self, record_value_dtype: ValueDType, _info):
        return record_value_dtype.name

    def get_value_annotation(self):
        return self.record_value_dtype.value

    def create_data_models(self) -> Tuple[Type[_MetricData], Type[_RecordData]]:
        hash_id = md5(self.model_dump_json().encode()).hexdigest()
        if hash_id in _METRIC_MODELS:
            return _METRIC_MODELS[hash_id]

        base_name = "".join([s.capitalize() for s in self.name.split("_")])
        record_model_name = base_name + "MetricRecord"
        metric_model_name = base_name + "Metric"
        module = _getframe(1).f_globals["__name__"]

        value_annotation = self.get_value_annotation()

        record_model_fields = {
            "value": (value_annotation, Field(default=VALUE_DETYPE_2_DEFAULT_VALUE[self.record_value_dtype])),
            "display_type": (Literal[self.record_display_type], Field(default=self.record_display_type)),
            "is_comparison": (Literal[self.is_comparison], Field(default=self.is_comparison)),
            "target_agent": (str, Field(default=...))
        }

        record_model = create_model(
            __model_name=record_model_name,
            __module__=module,
            __base__=_RecordData,
            **record_model_fields
        )

        metric_model_fields = {
            "records": (List[record_model], Field(default=...)),
            "is_comparison": (Literal[self.is_comparison], Field(default=self.is_comparison)),
            "target_agent": (str, Field(default=...))
        }

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
    "DynamicAggregationMethod",
    "DynamicAggregationFn",
    "DEFAULT_AGG_METHODS",
    "ValueDType",
    "DisplayType",
    "VALUE_DETYPE_2_DEFAULT_VALUE",
    "CompareMetricDefinition",
    "MetricDefinition",
    "MetricConfig"
]
