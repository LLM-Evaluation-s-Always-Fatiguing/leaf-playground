from inspect import Signature, Parameter
from warnings import warn
from sys import _getframe
from typing import Any, Annotated, List, Literal, Optional, Type, Union

from pydantic import create_model, field_serializer, BaseModel, Field, PrivateAttr

from .metric import MetricConfig, MetricDefinition
from ...._config import _Config
from ....data.message import Message


class ActionSignatureParameterDefinition(BaseModel):
    name: str = Field(default=...)
    annotation: Optional[Any] = Field(default=Parameter.empty)

    @field_serializer("annotation")
    def serialize_annotation(self, annotation: Optional[Type], _info):
        return str(annotation)


class ActionSignatureDefinition(BaseModel):
    parameters: Optional[List[ActionSignatureParameterDefinition]] = Field(default=None)
    return_annotation: Optional[Type[Message]] = Field(default=Signature.empty)
    is_static_method: bool = Field(default=False)

    def model_post_init(self, __context: Any) -> None:
        if self.return_annotation not in [None, Signature.empty] and not issubclass(self.return_annotation, Message):
            raise TypeError("return_annotation must be a subclass of Message if is explicitly specified")

    @field_serializer("return_annotation")
    def serialize_return_annotation(self, return_annotation: Optional[Type[Message]], _info):
        return str(return_annotation)


class ActionDefinition(BaseModel):
    name: str = Field(default=...)
    description: str = Field(default=...)
    signature: ActionSignatureDefinition = Field(default=...)
    metrics: Optional[List[MetricDefinition]] = Field(default=None)

    _belonged_role: Optional[
        "leaf_playground.core.scene_info.definitions.role.RoleDefinition"
    ] = PrivateAttr(default=None)

    @property
    def belonged_role(self):
        return self._belonged_role

    @property
    def belonged_chain(self):
        return self.belonged_role.name + "." + self.name

    def get_metric_definition(self, metric_name: str) -> MetricDefinition:
        for metric in self.metrics or []:
            if metric.name == metric_name:
                return metric
        raise ValueError(f"metric [{metric_name}] not found")

    def model_post_init(self, __context: Any) -> None:
        if self.signature.return_annotation in [None, Signature.empty] and self.metrics:
            warn(
                f"action [{self.name}] hasn't output, but metrics are specified, "
                f"this may cause some unexpected behaviors"
            )
        if self.metrics and len(set([m.name for m in self.metrics])) != len(self.metrics):
            raise ValueError(f"metrics of action [{self.name}] should have unique names")
        for metric in self.metrics or []:
            if metric.belonged_action:
                raise ValueError(
                    f"[{metric.name}] metric has already been bounded to action [{metric.belonged_action.name}]"
                )
            metric._belonged_action = self
            if metric.expect_resp_msg_type != self.signature.return_annotation:
                raise TypeError(
                    f"metric [{metric.name}] expects using {metric.expect_resp_msg_type.__name__} to evaluate, "
                    f"but action [{self.name}]'s result type is {self.signature.return_annotation.__name__}"
                )

        if not self.signature.is_static_method:
            self.signature.parameters = (
                [ActionSignatureParameterDefinition(name="self")] if not self.signature.parameters else
                [ActionSignatureParameterDefinition(name="self")] + self.signature.parameters
            )

    def get_signature(self) -> Signature:
        return Signature(
            parameters=[
                Parameter(name=p.name, kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=p.annotation)
                for p in (self.signature.parameters or [])
            ],
            return_annotation=self.signature.return_annotation
        )


class ActionMetricsConfig(_Config):
    @classmethod
    def create_config_model(cls, action_definition: ActionDefinition) -> Optional[Type["ActionMetricsConfig"]]:
        if not action_definition.metrics:
            return

        model_name = "".join([each.capitalize() for each in action_definition.name.split("_")]) + cls.__name__
        module = _getframe(1).f_globals["__name__"]
        fields = {}
        for metric in action_definition.metrics:
            fields[metric.name] = (MetricConfig, Field(default=...))
        return create_model(__model_name=model_name, __module__=module, __base__=cls, **fields)

    def get_metric_config(self, metric_name: str) -> MetricConfig:
        return getattr(self, metric_name)


class ActionConfig(_Config):
    metrics_config: Optional[ActionMetricsConfig] = Field(default=...)

    @classmethod
    def create_config_model(cls, action_definition: ActionDefinition) -> Type["ActionConfig"]:
        model_name = "".join([each.capitalize() for each in action_definition.name.split("_")]) + cls.__name__
        module = _getframe(1).f_globals["__name__"]
        metrics_config_model = ActionMetricsConfig.create_config_model(action_definition)

        fields = {}
        if not metrics_config_model:
            fields["metrics_config"] = (Literal[None], Field(default=None))
        else:
            fields["metrics_config"] = (metrics_config_model, Field(default=...))
        return create_model(__model_name=model_name, __module__=module, __base__=cls, **fields)

    def get_metric_config(self, metric_name: str):
        return self.metrics_config.get_metric_config(metric_name)


__all__ = [
    "ActionSignatureParameterDefinition",
    "ActionSignatureDefinition",
    "ActionDefinition",
    "ActionMetricsConfig",
    "ActionConfig"
]
