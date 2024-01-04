import asyncio
from abc import abstractmethod, ABC, ABCMeta
from inspect import signature
from sys import _getframe
from typing import Any, Dict, Optional

from pydantic import create_model, BaseModel, Field
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK, ConnectionClosedError

from .workers.socket_handler import SocketHandler
from .scene_definition import RoleDefinition
from .._config import _Config, _Configurable
from .._type import Immutable
from ..ai_backend.base import AIBackend, AIBackendConfig
from ..data.environment import EnvironmentVariable
from ..data.profile import Profile
from ..data.socket_data import SocketEvent
from ..utils.import_util import dynamically_import_obj, DynamicObject
from ..utils.type_util import validate_type


class _ActionHandler:
    def __init__(self, action_fn: callable, action_name: str, exec_timeout: int):
        self.action_fn = action_fn
        self.action_name = action_name
        self.exec_timeout = exec_timeout

    async def execute(self, *args, **kwargs):
        try:
            return await asyncio.wait_for(self.action_fn(*args, **kwargs), timeout=self.exec_timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"action [{self.action_name}] execution exceeded the time limit: {self.exec_timeout}s.")


class SceneAgentMetadata(BaseModel):
    cls_name: str = Field(default=...)
    description: str = Field(default=...)
    config_schema: Optional[dict] = Field(default=...)
    obj_for_import: DynamicObject = Field(default=...)
    is_human: bool = Field(default=...)


class SceneAgentMetaClass(ABCMeta):
    def __new__(
            cls,
            name,
            bases,
            attrs,
            *,
            role_definition: RoleDefinition = None,
            cls_description: str = None,
            action_exec_timeout: int = 30,
    ):
        attrs["role_definition"] = Immutable(role_definition or getattr(bases[0], "role_definition", None))
        attrs["cls_description"] = Immutable(cls_description)
        attrs["action_exec_timeout"] = Immutable(action_exec_timeout)
        attrs["obj_for_import"] = Immutable(DynamicObject(obj=name, module=_getframe(1).f_globals["__name__"]))

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
        if not validate_type(attrs["action_exec_timeout"], Immutable[int]):
            raise TypeError(
                f"class [{name}]'s class attribute [action_exec_timeout] should be a [int] instance, "
                f"got [{type(attrs['action_exec_timeout']).__name__}] type"
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
                if not asyncio.iscoroutinefunction(attrs[action_name]):
                    raise TypeError(f"[{action_name}] action must be an asynchronous method of class [{name}]")
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
            cls_description: str = None,
            action_exec_timeout: int = 30,
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
    action_exec_timeout:int
    obj_for_import: DynamicObject

    def __init__(self, config: config_cls):
        super().__init__(config=config)

        self._profile = self.config.profile
        self._role = self.role_definition.role_instance
        self._profile.role = self._role
        self._env_vars: Dict[str, EnvironmentVariable] = None

        if ABC not in self.__class__.__bases__:
            for action in self.role_definition.actions:
                action_name = action.name
                action_handler = _ActionHandler(
                    getattr(self, action_name),
                    action_name,
                    self.action_exec_timeout,
                )
                setattr(self, action_name, action_handler.execute)

    def bind_env_vars(self, env_vars: Dict[str, EnvironmentVariable]):
        self._env_vars = env_vars

    @property
    def env_vars(self) -> Dict[str, EnvironmentVariable]:
        return self._env_vars

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
            config_schema=cls.config_cls.get_json_schema(by_alias=True) if not cls.role_definition.is_static else None,
            obj_for_import=cls.obj_for_import,
            is_human=False
        )


class SceneDynamicAgentConfig(SceneAgentConfig):

    def model_post_init(self, __context) -> None:
        pass


class SceneDynamicAgent(SceneAgent, ABC):
    config_cls = SceneDynamicAgentConfig
    config: config_cls

    def __init__(self, config: config_cls):
        super().__init__(config=config)

        self.connected = False

    @abstractmethod
    def connect(self, *args, **kwargs):
        pass

    @abstractmethod
    def disconnect(self, *args, **kwargs):
        pass


class SceneAIAgentConfig(SceneDynamicAgentConfig):
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


class SceneAIAgent(SceneDynamicAgent, ABC):
    config_cls = SceneAIAgentConfig
    config: config_cls

    def __init__(self, config: config_cls):
        super().__init__(config=config)

        self.backend = self.config.create_backend_instance()
        self.connected = True

    def connect(self):
        pass

    def disconnect(self):
        pass


class HumanConnection:
    def __init__(self, agent: "SceneHumanAgent", socket: WebSocket, socket_handler: SocketHandler):
        self.agent = agent
        self.socket = socket

        self.events = asyncio.Queue()
        self.socket_handler = socket_handler
        self.state = WebSocketState.CONNECTING

    async def connect(self):
        await self.socket.accept()
        self.agent.connect(self)
        self.state = WebSocketState.CONNECTED

    def disconnect(self):
        self.agent.disconnect()
        self.state = WebSocketState.DISCONNECTED

    def notify_human_to_input(self):
        self.events.put_nowait(
            SocketEvent(event="wait_human_input")
        )

    async def _keep_alive(self):
        try:
            while True:
                await self.socket.send_json(SocketEvent(event="heart_beat").model_dump_json())
                await asyncio.sleep(1)
        except:
            self.disconnect()
            return

    async def _run(self):
        try:
            while True:
                event: SocketEvent = await self.events.get()
                await self.socket.send_json(event.model_dump_json())
                try:
                    human_input = await asyncio.wait_for(
                        self.socket.receive_text(),
                        timeout=self.agent.action_exec_timeout
                    )
                except asyncio.TimeoutError:
                    human_input = None
                while self.agent.human_input is not None:
                    await asyncio.sleep(0.01)
                self.agent.human_input = human_input
        except (WebSocketDisconnect, ConnectionClosed, ConnectionClosedError, ConnectionClosedOK):
            return

    async def run(self):
        await asyncio.gather(
            *[
                self.socket_handler.stream_sockets(self.socket),
                self._run(),
                self._keep_alive()
            ]
        )


class SceneHumanAgentConfig(SceneDynamicAgentConfig):
    pass


class SceneHumanAgent(SceneDynamicAgent, ABC):
    config_cls = SceneHumanAgentConfig
    config: config_cls

    def __init__(self, config: config_cls):
        super().__init__(config=config)

        self.connection: HumanConnection = None
        self.human_input = None
        self.wait_human_input = False

    def connect(self, connection: HumanConnection):
        self.connection = connection
        self.connected = True
        if self.wait_human_input:
            self.connection.notify_human_to_input()

    def disconnect(self):
        self.connection = None
        self.connected = False

    async def wait_human_text_input(self) -> Optional[str]:
        if not self.connected:
            return None
        self.wait_human_input = True
        self.connection.notify_human_to_input()
        while not self.human_input:
            await asyncio.sleep(0.001)
        self.wait_human_input = False
        human_input = self.human_input
        self.human_input = None
        return human_input

    async def wait_human_image_input(self, *args, **kwargs):
        raise NotImplementedError()  # TODO: impl

    @classmethod
    def get_metadata(cls):
        metadata = super().get_metadata()
        metadata.is_human = True
        return metadata


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
    "SceneStaticAgentConfig",
    "SceneStaticAgent",
    "SceneDynamicAgentConfig",
    "SceneDynamicAgent",
    "SceneAIAgentConfig",
    "SceneAIAgent",
    "SceneHumanAgentConfig",
    "SceneHumanAgent",
]
