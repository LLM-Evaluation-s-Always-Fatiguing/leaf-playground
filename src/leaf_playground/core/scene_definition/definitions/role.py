from sys import _getframe
from typing import Any, List, Literal, Optional, Tuple, Type, Union

from pydantic_core import PydanticUndefined
from pydantic import create_model, BaseModel, Field, PositiveInt, PrivateAttr

from .action import ActionConfig, ActionDefinition
from ...._config import _Config
from ....data.profile import Role
from ....utils.import_util import dynamically_import_obj, DynamicObject


class RoleDefinition(BaseModel):
    name: str = Field(default=...)
    description: str = Field(default=...)
    actions: List[ActionDefinition] = Field(default=...)
    num_agents_range: Tuple[PositiveInt, Union[PositiveInt, Literal[-1]]] = Field(default=(1, 1))
    is_static: bool = Field(default=False)

    _agents_cls: List[Type] = PrivateAttr(default=[])
    _role_instance: Optional[Role] = PrivateAttr(default=None)

    @property
    def agents_cls(self) -> List[Type["leaf_playground.core.scene_agent.SceneAgent"]]:
        return self._agents_cls

    @property
    def agents_metadata(self) -> List[BaseModel]:
        if not self.agents_cls:
            raise ValueError(f"no agent class is implemented for role [{self.name}]")
        return [cls.get_metadata() for cls in self.agents_cls]

    @property
    def min_agents_num(self) -> int:
        return self.num_agents_range[0]

    @property
    def max_agents_num(self) -> int:
        return self.num_agents_range[1]

    @property
    def role_instance(self) -> Role:
        if not self._role_instance:
            self._role_instance = Role(name=self.name, description=self.description, is_static=self.is_static)
        return self._role_instance

    def model_post_init(self, __context: Any) -> None:
        # validate num_agents_range
        if not self.is_static:
            if self.num_agents_range[1] != -1 and self.num_agents_range[1] < self.num_agents_range[0]:
                raise ValueError("the second item of num_agents_range must be -1 or greater than the first item")
        elif self.num_agents_range != (1, 1):
            raise ValueError("static role can only have one agent at a time")

        if not self.actions and not self.is_static:
            raise ValueError("dynamic role's actions can't be empty")
        if len(set([a.name for a in self.actions])) != len(self.actions):
            raise ValueError(f"actions of role [{self.name}] should have unique names")
        for action in self.actions:
            if action.belonged_role:
                raise ValueError(
                    f"[{action.name}] action has already been bounded to role [{action.belonged_role.name}]"
                )
            action._belonged_role = self
        # compare metrics can only be used in static role
        for action in self.actions:
            for metric in action.metrics or []:
                if metric.is_comparison and not self.is_static:
                    raise ValueError(
                        f"[{metric.belonged_chain}] metric is a comparison metric, can only be used for static role."
                    )

    def get_action_definition(self, action_name: str) -> ActionDefinition:
        for action in self.actions:
            if action.name == action_name:
                return action
        raise ValueError(f"action [{action_name}] not found")


class RoleActionsConfig(_Config):
    @classmethod
    def create_config_model(cls, role_definition: RoleDefinition) -> Type["RoleActionsConfig"]:
        model_name = "".join([each.capitalize() for each in role_definition.name.split("_")]) + cls.__name__
        module = _getframe(1).f_globals["__name__"]
        fields = {}
        for action in role_definition.actions:
            fields[action.name] = (ActionConfig.create_config_model(action), Field(default=...))
        return create_model(__model_name=model_name, __module__=module, __base__=cls, **fields)

    def get_action_config(self, action_name: str) -> ActionConfig:
        return getattr(self, action_name)


class RoleAgentConfig(_Config):
    config_data: dict = Field(default=...)
    obj_for_import: DynamicObject = Field(default=...)

    _agent_config = PrivateAttr(default=None)
    _agent_cls = PrivateAttr(default=None)

    @property
    def agent_config(self) -> "leaf_playground.core.scene_agent.SceneAgentConfig":
        if self._agent_config is None:
            self._agent_config = dynamically_import_obj(self.obj_for_import).config_cls(**self.config_data)
        return self._agent_config

    @property
    def agent_cls(self) -> Type["leaf_playground.core.scene_agent.SceneAgent"]:
        if self._agent_cls is None:
            self._agent_cls = dynamically_import_obj(self.obj_for_import)
        return self._agent_cls

    def initiate_agent(self) -> "leaf_playground.core.scene_agent.SceneAgent":
        return self.agent_cls(config=self.agent_config)


class RoleConfig(_Config):
    actions_config: RoleActionsConfig = Field(default=...)
    agents_config: List[RoleAgentConfig] = Field(default=...)
    is_static: bool = Field(default=...)

    _role_definition: RoleDefinition = PrivateAttr(default=None)

    def __init_subclass__(cls, _role_definition: RoleDefinition, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._role_definition = _role_definition

    @property
    def role_definition(self) -> RoleDefinition:
        return self._role_definition

    def model_post_init(self, __context: Any) -> None:
        # validate agents_config
        if not self.agents_config:
            raise ValueError("agents_config can't empty")
        num_agents = len(self.agents_config)
        # make sure all agents' name is unique
        if num_agents != len(set([conf.agent_config.profile.name for conf in self.agents_config])):
            raise AttributeError(f"each agents' name should be unique")
        # make sure all agent class can be recognized by role definition
        for agent_config in self.agents_config:
            if agent_config.agent_cls.role_definition != self.role_definition:
                raise TypeError(
                    f"expected agents class for role [{self.role_definition.name}] are: "
                    f"{self.role_definition.agents_cls}, got {agent_config.agent_cls}"
                )

    @classmethod
    def create_config_model(cls, role_definition: RoleDefinition) -> "RoleConfig":
        model_name = "".join([each.capitalize() for each in role_definition.name.split("_")]) + cls.__name__
        module = _getframe(1).f_globals["__name__"]
        cls_kwargs = {"_role_definition": role_definition}
        fields = {
            "actions_config": (RoleActionsConfig.create_config_model(role_definition), Field(default=...)),
            "agents_config": (
                List[RoleAgentConfig],
                Field(
                    default=(
                        ...
                        if not role_definition.is_static
                        else [
                            RoleAgentConfig(
                                config_data={}, obj_for_import=role_definition.agents_cls[0].obj_for_import
                            )
                        ]
                    ),
                    min_items=role_definition.min_agents_num,
                    max_items=(
                        role_definition.max_agents_num if role_definition.max_agents_num != -1 else PydanticUndefined
                    ),
                ),
            ),
            "is_static": (Literal[role_definition.is_static], Field(default=role_definition.is_static)),
        }

        return create_model(
            __model_name=model_name, __module__=module, __base__=cls, __cls_kwargs__=cls_kwargs, **fields
        )

    def get_action_config(self, action_name: str) -> ActionConfig:
        return self.actions_config.get_action_config(action_name)

    def initiate_agents(self) -> List["leaf_playground.core.scene_agent.SceneAgent"]:
        return [conf.initiate_agent() for conf in self.agents_config]


__all__ = ["RoleDefinition", "RoleActionsConfig", "RoleAgentConfig", "RoleConfig"]
