import json
from abc import abstractmethod
from datetime import datetime
from typing import Callable, List, Optional, Set, Any
from uuid import UUID

from pydantic import Field, FilePath

from ._config import _Config, _Configurable
from .data.message import Message, TextMessage
from .llm_backend.base import LLMBackend
from .utils.import_util import dynamically_import_obj, DynamicObject


class AgentConfig(_Config):
    profile_file: FilePath = Field(default=..., regex=r".*.json")
    profile_obj: DynamicObject = Field(
        default=DynamicObject(obj="Profile", module="leaf_playground.data.profile")
    )

    @property
    def profile(self):
        with open(self.profile_file.as_posix(), "r", encoding="utf-8") as f:
            profile_dict = json.load(f)
        return dynamically_import_obj(self.profile_obj)(**profile_dict)


class Agent(_Configurable):
    _config_type = AgentConfig

    def __init__(self, config: _config_type):
        super().__init__(config=config)
        self.profile = self.config.profile

    @property
    def id(self) -> UUID:
        return self.profile.id

    @property
    def name(self) -> str:
        return self.profile.name

    @abstractmethod
    def act(self, *args, **kwargs) -> Message:
        raise NotImplementedError()

    @classmethod
    def from_config(cls, config: _config_type) -> "Agent":
        return cls(config=config)


class LLMAgentConfig(AgentConfig):
    backend_config_file: FilePath = Field(default=...)
    backend_config_obj: DynamicObject = Field(
        default=DynamicObject(obj="LLMBackendConfig", module="leaf_playground.llm_backend.base")
    )
    backend_obj: DynamicObject = Field(
        default=DynamicObject(obj="LLMBackend", module="leaf_playground.llm_backend.base")
    )

    @property
    def backend_config(self):
        with open(self.backend_config_file, "r", encoding="utf-8") as f:
            return dynamically_import_obj(self.backend_config_obj)(**json.load(f))


class LLMAgent(Agent):
    _config_type = LLMAgentConfig

    def __init__(self, config: _config_type):
        super().__init__(config=config)
        self.llm_backend: LLMBackend = dynamically_import_obj(
            self.config.backend_obj
        )(config=self.config.backend_config)

    @abstractmethod
    def act(self, *args, **kwargs) -> TextMessage:
        raise NotImplementedError()


class LLMChatterConfig(LLMAgentConfig):
    profile_format_template: str = Field(default=...)
    profile_format_fields: Set[str] = Field(default=...)
    message_format_template: str = Field(default=...)
    message_format_fields: Set[str] = Field(default=...)
    prompt_constructor: Callable[[str, List[str]], str] = Field(default=...)
    max_his_num: int = Field(default=8)
    additional_backend_params: Optional[dict] = Field(default=None)
    message_type: type = Field(default=TextMessage, exclude=True)

    def model_post_init(self, __context: Any) -> None:
        def valid_fields(param, data_cls, param_name):
            fields_space = set(data_cls.__annotations__.keys())
            param_fields = set(param)
            if not param_fields.issubset(fields_space):
                raise ValueError(
                    f"'{param_name}' must be a subset of {fields_space}, "
                    f"but {param_fields - fields_space} are not in it."
                )

        valid_fields(self.profile_format_fields, self.profile.__class__, "profile_format_fields")
        valid_fields(self.message_format_fields, self.message_type, "profile_format_fields")


class LLMChatter(LLMAgent):
    _config_type = LLMChatterConfig

    def _build_prompt(self, messages: List[TextMessage]):
        profile = self.profile.format(
            self.config.profile_format_template,
            self.config.profile_format_fields
        )
        histories = messages[-self.config.max_his_num:]
        histories = [
            his.format(self.config.message_format_template, self.config.message_format_fields)
            for his in histories
        ]
        prompt = self.config.prompt_constructor(profile, histories)
        return prompt

    def _respond(self, prompt: str) -> str:
        return self.llm_backend.completion(prompt)

    def _construct_message(self, response: str) -> TextMessage:
        try:
            return self.config.message_type(
                sender_id=self.id,
                sender_name=self.name,
                content=response,
                timestamp=datetime.utcnow().timestamp()
            )
        except:
            raise TypeError(
                f"Initialize {self.config.message_type.__name__} type message failed, "
                f"you may need to override '_construct_message' method"
            )

    def act(self, messages: List[TextMessage], *args, **kwargs) -> TextMessage:
        prompt = self._build_prompt(messages)
        response = self._respond(prompt)
        return self._construct_message(response)


class LLMGamer(LLMAgent):
    pass  # TODO: impl
