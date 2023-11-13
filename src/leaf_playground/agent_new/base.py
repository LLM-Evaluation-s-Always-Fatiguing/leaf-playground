import json
from abc import abstractmethod
from typing import Any, List, Optional
from uuid import UUID

from pydantic import Field, FilePath

from .._config import _Config, _Configurable
from ..data.message import Message
from ..data.profile import Profile
from ..data.tool import Tool
from ..utils.import_util import dynamically_import_obj, DynamicObject


class DataObjConfig(_Config):
    data: Optional[dict] = Field(default=None)
    file: Optional[FilePath] = Field(default=None, pattern=r".*.json")
    obj: DynamicObject = Field(
        default=DynamicObject(obj="Data", module="leaf_playground.data.base")
    )

    def model_post_init(self, __context: Any) -> None:
        if self.data is None and self.file is None:
            raise ValueError("at least one of data or file should be specified")

    def create_instance(self):
        if not self.data:
            with open(self.file.as_posix(), "r", encoding="utf-8") as f:
                self.data = json.load(f)
        return dynamically_import_obj(self.obj)(**self.data)


class AgentConfig(_Config):
    profile_: DataObjConfig = Field(default=..., alias="profile")
    tools_: Optional[List[DataObjConfig]] = Field(default=None, alias="tools")
    message_type_: DynamicObject = Field(
        default=DynamicObject(obj="Message", module="leaf_playground.data.message"),
        alias="message_type"
    )

    def model_post_init(self, __context: Any) -> None:
        profile_obj = dynamically_import_obj(self.profile_.obj)
        if not issubclass(profile_obj, Profile):
            raise ValueError(f"{self.profile_.obj.obj} in profile.obj should be a subclass of Profile")
        for tool in (self.tools_ if self.tools_ else []):
            tool_obj = dynamically_import_obj(tool.obj)
            if not issubclass(tool_obj, Tool):
                raise ValueError(f"{tool.obj.obj} in tools.obj should be a subclass of Tool")
        if not issubclass(dynamically_import_obj(self.message_type_), Message):
            raise ValueError(f"{self.message_type_.obj} in message_type should be a subclass of Message")

    @property
    def profile(self) -> Profile:
        return self.profile_.create_instance()

    @property
    def tools(self) -> Optional[List[Tool]]:
        if not self.tools_:
            return
        return [tool.create_instance() for tool in self.tools_]

    @property
    def message_type(self):
        return dynamically_import_obj(self.message_type_)


class Agent(_Configurable):
    config_obj = AgentConfig
    config: config_obj

    def __init__(self, config: config_obj):
        super().__init__(config=config)
        self.profile = self.config.profile
        self.tools = self.config.tools

    @property
    def id(self) -> UUID:
        return self.profile.id

    @property
    def name(self) -> str:
        return self.profile.name

    @property
    def role_name(self) -> str:
        return self.profile.role.name

    def _valid_message(self, *messages: Message):
        for message in messages:
            if not isinstance(message, dynamically_import_obj(self.config.message_type)):
                raise TypeError(f"message should be an instance of {self.config.message_type.__name__}")

    @abstractmethod
    def preprocess(self, messages: List[Message]) -> Any:
        raise NotImplementedError()

    @abstractmethod
    async def a_preprocess(self, messages: List[Message]) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def respond(
        self,
        input: Any,
        *args,
        prefix: Optional[str] = None,
        **kwargs
    ) -> Any:
        raise NotImplementedError()

    @abstractmethod
    async def a_respond(
        self,
        input: Any,
        *args,
        prefix: Optional[str] = None,
        **kwargs
    ) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def postprocess(
        self,
        response: Any,
        response_prefix: Optional[str] = None,
        receiver_ids: Optional[List[UUID]] = None,
        receiver_names: Optional[List[str]] = None,
        receiver_role_names: Optional[List[str]] = None
    ) -> Message:
        raise NotImplementedError()

    @abstractmethod
    async def a_postprocess(
        self,
        response: Any,
        response_prefix: Optional[str] = None,
        receiver_ids: Optional[List[UUID]] = None,
        receiver_names: Optional[List[str]] = None,
        receiver_role_names: Optional[List[str]] = None
    ) -> Message:
        raise NotImplementedError()

    def act(
        self,
        messages: List[Message],
        *args,
        response_prefix: Optional[str] = None,
        receiver_ids: Optional[List[UUID]] = None,
        receiver_names: Optional[List[str]] = None,
        receiver_role_names: Optional[List[str]] = None,
        **kwargs
    ) -> Message:
        self._valid_message(*messages)
        inp = self.preprocess(messages)
        response = self.respond(inp, *args, prefix=response_prefix, **kwargs)
        out = self.postprocess(
            response,
            response_prefix=response_prefix,
            receiver_ids=receiver_ids,
            receiver_names=receiver_names,
            receiver_role_names=receiver_role_names
        )
        self._valid_message(out)
        return out

    async def a_act(
        self,
        messages: List[Message],
        *args,
        response_prefix: Optional[str] = None,
        receiver_ids: Optional[List[UUID]] = None,
        receiver_names: Optional[List[str]] = None,
        receiver_role_names: Optional[List[str]] = None,
        **kwargs
    ) -> Message:
        self._valid_message(*messages)
        inp = await self.a_preprocess(messages)
        response = await self.a_respond(inp, *args, prefix=response_prefix, **kwargs)
        out = await self.a_postprocess(
            response,
            response_prefix=response_prefix,
            receiver_ids=receiver_ids,
            receiver_names=receiver_names,
            receiver_role_names=receiver_role_names
        )
        self._valid_message(out)
        return out

    @classmethod
    def from_config(cls, config: config_obj) -> "Agent":
        return cls(config=config)
