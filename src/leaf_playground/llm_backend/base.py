from abc import abstractmethod
from typing import List

from pydantic import BaseModel

from .._config import _Config, _Configurable


class LLMBackendChatCompletionInput(BaseModel):
    pass


class LLMBackendConfig(_Config):
    pass


class LLMBackend(_Configurable):
    config_obj = LLMBackendConfig

    @classmethod
    def from_config(cls, config: config_obj) -> "LLMBackend":
        return cls(config=config)

    @abstractmethod
    def completion(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError()

    @abstractmethod
    async def a_completion(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError()

    @abstractmethod
    def chat_completion(self, inputs: List[LLMBackendChatCompletionInput], **kwargs) -> str:
        raise NotImplementedError()

    @abstractmethod
    async def a_chat_completion(self, inputs: List[LLMBackendChatCompletionInput], **kwargs) -> str:
        raise NotImplementedError()
