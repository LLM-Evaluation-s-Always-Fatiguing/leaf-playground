from inspect import signature, Signature
from typing import Dict, Optional, Any

from pydantic import BaseModel, Field

from .scene_info import SceneInfo
from .._config import _Config, _Configurable
from ..ai_backend.base import AIBackend, AIBackendConfig
from ..data.profile import Profile, Role
from ..utils.import_util import dynamically_import_obj, DynamicObject


class SceneAgentMetadata(BaseModel):
    cls_name: str = Field(default=...)
    description: str = Field(default=...)
    actions: Dict[str, str] = Field(default=...)


class SceneAgentConfig(_Config):
    profile: Profile = Field(default=...)
    chart_major_color: Optional[str] = Field(default=None, pattern=r"^#[0-9a-fA-F]{6}$")


class SceneAgent(_Configurable):
    config_obj = SceneAgentConfig
    config: config_obj

    _actions: Dict[str, Signature]

    description: str
    obj_for_import: DynamicObject

    def __init__(self, config: config_obj):
        if hasattr(self, f"_{self.__class__.__name__}__valid_class_attributes"):
            getattr(self, f"_{self.__class__.__name__}__valid_class_attributes")()
        self.__valid_class_attributes()
        # check if all actions exist and their signatures are correct
        for action_name, action_signature in self._actions.items():
            if not hasattr(self, action_name):
                raise AttributeError(f"action [{action_name}] not found")
            if not callable(getattr(self, action_name)):
                raise TypeError(f"action [{action_name}] should be callable")
            if action_signature != signature(getattr(self, action_name)):
                raise TypeError(
                    f"signature of action [{action_name}] should be {action_signature}, "
                    f"got {signature(getattr(self, action_name))}"
                )

        super().__init__(config=config)

        self.profile = self.config.profile
        self.scene_info: SceneInfo = None

        self._post_initialized = False

    def __valid_class_attributes(self):
        if not hasattr(self, "_actions"):
            raise AttributeError(f"class attribute _actions not found, must specify in your agent class")
        if not hasattr(self, "description"):
            raise AttributeError(f"class attribute description not found, must specify in your agent class")
        if not hasattr(self, "obj_for_import"):
            raise AttributeError(f"class attribute obj_for_import not found, must specify in your agent class")
        if self.__class__.__name__ != self.obj_for_import.obj:
            raise ValueError(
                f"obj_for_import isn't correct, should be {self.__class__.__name__}, got {self.obj_for_import.obj}"
            )

    def post_init(self, role: Optional[Role], scene_info: SceneInfo):
        if role is not None:
            self.profile.role = role
        self.scene_info = scene_info

        self._post_initialized = True

    @property
    def id(self):
        return self.profile.id

    @property
    def name(self):
        return self.profile.name

    @property
    def role_name(self):
        if not self._post_initialized:
            raise RuntimeError("post_init() should be called before accessing role_name")
        return self.profile.role.name

    @classmethod
    def from_config(cls, config: config_obj) -> "SceneAgent":
        return cls(config=config)

    @classmethod
    def get_metadata(cls):
        return SceneAgentMetadata(
            cls_name=cls.__name__,
            description=cls.description,
            actions={action_name: str(action_signature) for action_name, action_signature in cls._actions.items()},
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
        if not isinstance(ai_backend_config, ai_backend_cls.config_obj):
            raise TypeError(f"ai_backend_config should be an instance of {ai_backend_cls.config_obj.__name__}")


class SceneAIAgent(SceneAgent):
    config_obj = SceneAIAgentConfig
    config: config_obj

    def __init__(self, config: config_obj):
        super().__init__(config=config)

        self.backend = self.config.create_backend_instance()


class SceneHumanAgentConfig(SceneAgentConfig):
    pass


class SceneHumanAgent(SceneAgent):
    config_obj = SceneHumanAgentConfig
    config: config_obj

    def __init__(self, config: config_obj):
        super().__init__(config=config)

    async def wait_human_text_input(self, *args, **kwargs):
        raise NotImplementedError()  # TODO: impl

    async def wait_human_image_input(self, *args, **kwargs):
        raise NotImplementedError()  # TODO: impl


class SceneStaticAgentConfig(SceneAgentConfig):
    def model_post_init(self, __context: Any) -> None:
        fields = list(self.model_fields.keys())
        if set(fields) - {"profile", "chart_major_color"}:
            raise ValueError(f"{self.__class__.__name__} requires profile and chart_major_color only, got {fields}")


class SceneStaticAgent(SceneAgent):
    config_obj = SceneStaticAgentConfig
    config: config_obj

    def __init__(self, config: config_obj):
        if not config.profile.role:
            raise ValueError("role should be specified at initialization for a static agent")
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
