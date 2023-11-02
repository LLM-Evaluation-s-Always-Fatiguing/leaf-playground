import json
from abc import abstractmethod
from datetime import datetime
from typing import Callable, List, Optional, Set, Any
from uuid import UUID

from pydantic import Field, FilePath

from ._config import _Config, _Configurable
from .data.message import Message, TextMessage
from .llm_backend.base import LLMBackend
from .utils.import_util import dynamically_import_obj, dynamically_import_fn, DynamicObject, DynamicFn


class AgentConfig(_Config):
    profile_data: Optional[dict] = Field(default=None)
    profile_file: Optional[FilePath] = Field(default=None, pattern=r".*.json")
    profile_obj: DynamicObject = Field(
        default=DynamicObject(obj="Profile", module="leaf_playground.data.profile")
    )

    def model_post_init(self, __context: Any) -> None:
        if self.profile_data is None and self.profile_file is None:
            raise ValueError("at least one of profile_data or profile_file should be specified")

    @property
    def profile(self):
        if self.profile_data is None:
            with open(self.profile_file.as_posix(), "r", encoding="utf-8") as f:
                self.profile_data = json.load(f)
        return dynamically_import_obj(self.profile_obj)(**self.profile_data)


class Agent(_Configurable):
    config_obj = AgentConfig

    def __init__(self, config: config_obj):
        super().__init__(config=config)
        self.profile = self.config.profile

    @property
    def id(self) -> UUID:
        return self.profile.id

    @property
    def name(self) -> str:
        return self.profile.name

    @property
    def role_name(self) -> str:
        return self.profile.role.name

    @abstractmethod
    def act(self, *args, **kwargs) -> Message:
        raise NotImplementedError()

    @abstractmethod
    def a_act(self, *args, **kwargs) -> Message:
        raise NotImplementedError()

    @classmethod
    def from_config(cls, config: config_obj) -> "Agent":
        return cls(config=config)


class LLMAgentConfig(AgentConfig):
    backend_config_data: Optional[dict] = Field(default=None, alias="backend_config_data")
    backend_config_file: Optional[FilePath] = Field(default=None, pattern=r".*.json")
    backend_obj: DynamicObject = Field(
        default=DynamicObject(obj="LLMBackend", module="leaf_playground.llm_backend.base")
    )

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)

        if self.backend_config_data is None and self.backend_config_file is None:
            raise ValueError("at least one of backend_config_data or backend_config_file should be specified")

    @property
    def backend(self):
        if self.backend_config_data is None:
            with open(self.backend_config_file, "r", encoding="utf-8") as f:
                self.backend_config_data = json.load(f)
        backend_obj = dynamically_import_obj(self.backend_obj)
        backend_config = backend_obj.config_obj(**self.backend_config_data)
        return backend_obj(config=backend_config)


class LLMAgent(Agent):
    config_obj = LLMAgentConfig

    def __init__(self, config: config_obj):
        super().__init__(config=config)
        self.llm_backend: LLMBackend = self.config.backend

    @abstractmethod
    def act(self, *args, **kwargs) -> TextMessage:
        raise NotImplementedError()

    @abstractmethod
    async def a_act(self, *args, **kwargs) -> TextMessage:
        raise NotImplementedError()


class TextCompletionAgentConfig(LLMAgentConfig):
    profile_format_template: str = Field(default=...)
    profile_format_fields: Set[str] = Field(default=...)
    message_format_template: str = Field(default=...)
    message_format_fields: Set[str] = Field(default=...)
    prompt_constructor_obj: DynamicFn = Field(default=...)
    history_window_size: int = Field(default=8)
    message_type_obj: DynamicObject = Field(
        default=DynamicObject(obj="TextMessage", module="leaf_playground.data.message"),
        exclude=True
    )

    @property
    def message_type(self):
        return dynamically_import_obj(self.message_type_obj)

    @property
    def prompt_constructor(self) -> Callable[[str, List[str], Optional[str]], str]:
        return dynamically_import_fn(self.prompt_constructor_obj)

    def model_post_init(self, __context: Any) -> None:
        pass  # TODO: correct way to valid *_format_fields


class TextCompletionAgent(LLMAgent):
    config_obj = TextCompletionAgentConfig

    def _build_prompt(self, messages: List[TextMessage], response_prefix: Optional[str] = None):
        profile = self.profile.format(
            self.config.profile_format_template,
            self.config.profile_format_fields
        )
        histories = messages[-self.config.history_window_size:]
        histories = [
            his.format(self.config.message_format_template, self.config.message_format_fields)
            for his in histories
        ]
        response_prefix = TextMessage(
            sender_id=self.id,
            sender_name=self.name,
            sender_role_name=self.role_name,
            content=response_prefix or ""
        ).format(self.config.message_format_template, self.config.message_format_fields)
        prompt = self.config.prompt_constructor(profile, histories, response_prefix)
        return prompt

    def _respond(self, prompt: str) -> str:
        return self.llm_backend.completion(prompt).strip()

    async def _a_respond(self, prompt: str) -> str:
        return (await self.llm_backend.a_completion(prompt)).strip()

    def _construct_message(
        self,
        response: str,
        receiver_ids: Optional[List[UUID]] = None,
        receiver_names: Optional[List[str]] = None,
        receiver_role_names: Optional[List[str]] = None,
    ) -> TextMessage:
        try:
            return self.config.message_type(
                sender_id=self.id,
                sender_name=self.name,
                sender_role_name=self.role_name,
                receiver_ids=receiver_ids,
                receiver_names=receiver_names,
                receiver_role_names=receiver_role_names,
                content=response,
                timestamp=datetime.utcnow().timestamp()
            )
        except:
            raise TypeError(
                f"Initialize {self.config.message_type.__name__} type message failed, "
                f"you may need to override '_construct_message' method"
            )

    def act(
        self,
        messages: List[TextMessage],
        *args,
        response_prefix: Optional[str] = None,
        receiver_ids: Optional[List[UUID]] = None,
        receiver_names: Optional[List[str]] = None,
        receiver_role_names: Optional[List[str]] = None,
        **kwargs
    ) -> TextMessage:
        prompt = self._build_prompt(messages, response_prefix)
        response = self._respond(prompt)
        return self._construct_message(
            response,
            receiver_ids=receiver_ids,
            receiver_names=receiver_names,
            receiver_role_names=receiver_role_names
        )

    async def a_act(
        self,
        messages: List[TextMessage],
        *args,
        response_prefix: Optional[str] = None,
        receiver_ids: Optional[List[UUID]] = None,
        receiver_names: Optional[List[str]] = None,
        receiver_role_names: Optional[List[str]] = None,
        **kwargs
    ) -> TextMessage:
        prompt = self._build_prompt(messages, response_prefix)
        response = await self._a_respond(prompt)
        return self._construct_message(
            response,
            receiver_ids=receiver_ids,
            receiver_names=receiver_names,
            receiver_role_names=receiver_role_names
        )


class ChatCompletionAgent(LLMAgent):
    pass  # TODO: impl
