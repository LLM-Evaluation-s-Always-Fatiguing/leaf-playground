from abc import ABC, ABCMeta
from inspect import signature
from sys import _getframe
from typing import Any, Dict, Optional

from pydantic import create_model, BaseModel, Field

from .scene_definition import RoleDefinition
from .._config import _Config, _Configurable
from .._type import Immutable
from ..ai_backend.base import AIBackend, AIBackendConfig
from ..data.environment import EnvironmentVariable
from ..data.profile import Profile
from ..utils.import_util import dynamically_import_obj, DynamicObject
from ..utils.type_util import validate_type


class SceneAgentMetadata(BaseModel):
    cls_name: str = Field(default=...)
    description: str = Field(default=...)
    config_schema: dict = Field(default=...)
    obj_for_import: DynamicObject = Field(default=...)


class SceneAgentMetaClass(ABCMeta):
    def __new__(
        cls,
        name,
        bases,
        attrs,
        *,
        role_definition: RoleDefinition = None,
        cls_description: str = None
    ):
        attrs["role_definition"] = Immutable(role_definition or getattr(bases[0], "role_definition", None))
        attrs["cls_description"] = Immutable(cls_description)
        attrs["obj_for_import"] = Immutable(DynamicObject(obj=name, source_file=_getframe(1).f_code.co_filename))

        new_cls = super().__new__(cls, name, bases, attrs)

        DynamicObject.bind_dynamic_obj(attrs["obj_for_import"], new_cls)

        if not validate_type(attrs["role_definition"], Immutable[Optional[RoleDefinition]]):
            raise TypeError(
                f"class [{name}]'s class attribute [role_definition] should be a [RoleDefinition] instance, "
                f"got [{type(attrs['role_definition']).__name__}] type"
            )
        if not validate_type(attrs["cls_description"], Immutable[Optional[str]]):
            raise TypeError(
                f"class [{name}]'s class attribute [cls_description] should be a [str] instance, "
                f"got [{type(attrs['cls_description']).__name__}] type"
            )

        if ABC not in bases:
            # check if those class attrs are empty when the class is not abstract
            if not new_cls.role_definition:
                raise AttributeError(
                    f"class [{name}] missing class attribute [role_definition], please specify it by "
                    f"doing like: `class {name}(role_definition=your_role_def)`, or you can also "
                    f"specify in the super class [{bases[0].__name__}]"
                )
            if not new_cls.cls_description:
                raise AttributeError(
                    f"class [{name}] missing class attribute [cls_description], please specify it by "
                    f"doing like: `class {name}(cls_description=your_cls_desc)`, where 'your_cls_desc' "
                    f"is a string that introduces your agent class"
                )
            # bind the agent class to its role definition
            new_cls.role_definition._agents_cls.append(new_cls)

        if new_cls.role_definition:
            for action in new_cls.role_definition.actions:
                action_name = action.name
                action_sig = action.get_signature()

                if action_name not in attrs:
                    raise AttributeError(f"missing [{action_name}] action in class [{name}]")
                if not callable(attrs[action_name]):
                    raise TypeError(f"[{action_name}] action must be a method of class [{name}]")
                if signature(attrs[action_name]) != action_sig:
                    raise TypeError(
                        f"expected signature of [{action_name}] action in class [{name}] is {str(action_sig)}, "
                        f"got {str(signature(attrs[action_name]))}"
                    )

        return new_cls

    def __init__(
        cls,
        name,
        bases,
        attrs,
        *,
        role_definition: RoleDefinition = None,
        cls_description: str = None
    ):
        super().__init__(name, bases, attrs)

    def __setattr__(self, key, value):
        # make sure those class attributes immutable in class-wise
        if key in ["role_definition", "cls_description", "obj_for_import"] and hasattr(self, key):
            raise AttributeError(f"class attribute {key} is immutable")
        return super().__setattr__(key, value)

    def __delattr__(self, item):
        # make sure those class attributes can't be deleted
        if item in ["role_definition", "cls_description", "obj_for_import"] and hasattr(self, item):
            raise AttributeError(f"class attribute [{item}] can't be deleted")
        return super().__delattr__(item)


class SceneAgentConfig(_Config):
    profile: Profile = Field(default=...)
    chart_major_color: Optional[str] = Field(default=None, pattern=r"^#[0-9a-fA-F]{6}$")


class SceneAgent(_Configurable, ABC, metaclass=SceneAgentMetaClass):
    config_cls = SceneAgentConfig
    config: config_cls

    # class attributes initialized in metaclass
    role_definition: RoleDefinition
    cls_description: str
    obj_for_import: DynamicObject

    def __init__(self, config: config_cls):
        super().__init__(config=config)

        self._profile = self.config.profile
        self._role = self.role_definition.role_instance
        self._profile.role = self._role
        self._env_vars: Dict[str, EnvironmentVariable] = None

    def bind_env_vars(self, env_vars: Dict[str, EnvironmentVariable]):
        self._env_vars = env_vars

    @property
    def profile(self):
        return self._profile

    @property
    def role(self):
        return self._role

    @property
    def id(self):
        return self.profile.id

    @property
    def name(self):
        return self.profile.name

    @property
    def role_name(self):
        return self.role.name

    @property
    def env_var(self):
        return self._env_vars

    @classmethod
    def from_config(cls, config: config_cls) -> "SceneAgent":
        return cls(config=config)

    @classmethod
    def get_metadata(cls):
        return SceneAgentMetadata(
            cls_name=cls.__name__,
            description=cls.cls_description,
            config_schema=cls.config_cls.get_json_schema(by_alias=True),
            obj_for_import=cls.obj_for_import
        )


class SceneAIAgentConfig(SceneAgentConfig):
    ai_backend_config: AIBackendConfig = Field(default=...)
    ai_backend_obj: DynamicObject = Field(default=..., exclude=True)

    def model_post_init(self, __context) -> None:
        self.valid(self.ai_backend_config, self.ai_backend_obj)

    def create_backend_instance(self) -> AIBackend:
        obj = dynamically_import_obj(self.ai_backend_obj)
        return obj.from_config(config=self.ai_backend_config)

    @staticmethod
    def valid(ai_backend_config: AIBackendConfig, ai_backend_obj: DynamicObject):
        ai_backend_cls = dynamically_import_obj(ai_backend_obj)
        if not issubclass(ai_backend_cls, AIBackend):
            raise TypeError(f"ai_backend_obj {ai_backend_obj.obj} should be a subclass of AIBackend")
        if not isinstance(ai_backend_config, ai_backend_cls.config_cls):
            raise TypeError(f"ai_backend_config should be an instance of {ai_backend_cls.config_cls.__name__}")


class SceneAIAgent(SceneAgent, ABC):
    config_cls = SceneAIAgentConfig
    config: config_cls

    def __init__(self, config: config_cls):
        super().__init__(config=config)

        self.backend = self.config.create_backend_instance()


class SceneHumanAgentConfig(SceneAgentConfig):
    pass


class SceneHumanAgent(SceneAgent, ABC):
    config_cls = SceneHumanAgentConfig
    config: config_cls

    def __init__(self, config: config_cls):
        super().__init__(config=config)

    async def wait_human_text_input(self, *args, **kwargs):
        raise NotImplementedError()  # TODO: impl

    async def wait_human_image_input(self, *args, **kwargs):
        raise NotImplementedError()  # TODO: impl


class SceneStaticAgentConfig(SceneAgentConfig):
    def model_post_init(self, __context: Any) -> None:
        fields = self.model_fields_set
        if fields - {"profile", "chart_major_color"}:
            raise ValueError(f"{self.__class__.__name__} requires profile and chart_major_color only, got {fields}")

    @classmethod
    def create_config_model(cls, role_definition: RoleDefinition) -> "SceneStaticAgentConfig":
        model_name = "".join([each.capitalize() for each in role_definition.name.split("_")]) + "Config"
        module = _getframe(1).f_code.co_filename
        fields = {
            "profile": (Profile, Field(default=Profile(name=role_definition.name), frozen=True, exclude=True)),
            "chart_major_color": (Optional[str], Field(default=None, pattern=r"^#[0-9a-fA-F]{6}$", exclude=True))
        }
        return create_model(
            __model_name=model_name,
            __module__=module,
            __base__=cls,
            **fields
        )


class SceneStaticAgent(SceneAgent, ABC):
    config_cls = SceneStaticAgentConfig
    config: config_cls

    def __init__(self, config: config_cls):
        super().__init__(config=config)


__all__ = [
    "SceneAgentMetadata",
    "SceneAgentConfig",
    "SceneAgent",
    "SceneAIAgentConfig",
    "SceneAIAgent",
    "SceneHumanAgentConfig",
    "SceneHumanAgent",
    "SceneStaticAgentConfig",
    "SceneStaticAgent",
]
