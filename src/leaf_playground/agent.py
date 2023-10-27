from abc import abstractmethod
from datetime import datetime
from typing import Callable, List, Optional, Set, Any
from uuid import UUID

from pydantic import BaseModel, Field

from .data.message import Message, TextMessage, JSONMessage
from .data.game_event import GameEvent
from .data.profile import Profile


class Agent(BaseModel):
    profile: Profile = Field(default=...)

    @property
    def id(self) -> UUID:
        return self.profile.id

    @property
    def name(self) -> str:
        return self.profile.name

    @abstractmethod
    def act(self, *args, **kwargs) -> Message:
        raise NotImplementedError()


class LLMAgent(Agent):
    llm_backend: Callable[[str, Optional[dict]], str] = Field(default=...)

    @abstractmethod
    def act(self, *args, **kwargs) -> TextMessage:
        raise NotImplementedError()


class LLMChatter(LLMAgent):
    profile_format_template: str = Field(default=...)
    profile_format_fields: Set[str] = Field(default=...)
    message_format_template: str = Field(default=...)
    message_format_fields: Set[str] = Field(default=...)
    prompt_constructor: Callable[[str, List[str]], str] = Field(default=...)
    max_his_num: int = Field(default=8)
    additional_backend_params: Optional[dict] = Field(default=None)
    message_type: type = TextMessage

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

    def _build_prompt(self, messages: List[TextMessage]):
        profile = self.profile.format(self.profile_format_template, self.profile_format_fields)
        histories = messages[-self.max_his_num:]
        histories = [his.format(self.message_format_template, self.message_format_fields) for his in histories]
        prompt = self.prompt_constructor(profile, histories)
        return prompt

    def _respond(self, prompt: str) -> str:
        return self.llm_backend(prompt, self.additional_backend_params)

    def _construct_message(self, response: str) -> "LLMChatter.message_type":
        try:
            return self.message_type(
                sender_id=self.id,
                sender_name=self.name,
                content=response,
                timestamp=datetime.utcnow().timestamp()
            )
        except:
            raise TypeError(
                f"Initialize {self.message_type.__name__} type message failed, "
                f"you may need to override '_construct_message' method"
            )

    def act(self, messages: List[TextMessage], *args, **kwargs) -> "LLMChatter.message_type":
        prompt = self._build_prompt(messages)
        response = self._respond(prompt)
        return self._construct_message(response)


class LLMGamer(LLMAgent):
    pass  # TODO: impl
