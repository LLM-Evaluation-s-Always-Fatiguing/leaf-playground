import os
from typing import Callable, Dict, List, Optional, Any

import openai
from openai.util import convert_to_dict
from pydantic import Field

from .base import LLMBackendChatCompletionInput, LLMBackendConfig, LLMBackend


class OpenAIBackendChatCompletionInput(LLMBackendChatCompletionInput):
    content: str = Field(default=...)
    role: str = Field(default=..., regex=r"(system|user|assistant)")
    name: Optional[str] = Field(default=None, max_length=64, regex=r"[0-9a-zA-Z_]{1,64}")


class OpenAIBackendConfig(LLMBackendConfig):
    model: str = Field(default=...)
    _api_key: Optional[str] = Field(default=None, alias="api_key")
    _api_base: Optional[str] = Field(default=None, alias="api_base")
    _api_type: Optional[str] = Field(default=None, alias="api_type")
    _api_version: Optional[str] = Field(default=None, alias="api_version")
    api_key_env_var: str = Field(default="OPENAI_API_KEY")
    api_base_env_var: str = Field(default="OPENAI_API_BASE")
    api_type_env_var: str = Field(default="OPENAI_API_TYPE")
    api_version_env_var: str = Field(default="OPENAI_API_VERSION")

    @property
    def api_key(self) -> str:
        if self._api_key is not None:
            return self._api_key
        api_key = os.environ.get(self.api_key_env_var, None)
        if api_key is None:
            raise ValueError("openai api key neither be specified nor be found in environment variable.")
        return api_key

    @property
    def api_base(self) -> str:
        return self._api_base or os.environ.get(self.api_base_env_var, "https://api.openai.com/v1")

    @property
    def api_type(self):
        return self._api_type or os.environ.get(self.api_type_env_var, "open_ai")

    @property
    def api_version(self):
        return self._api_version or os.environ.get(
            self.api_version_env_var,
            ("2023-05-15" if self.api_type in ("azure", "azure_ad", "azuread") else None)
        )

    @property
    def payload(self) -> Dict[str, str]:
        return {
            "api_key": self.api_key,
            "api_base": self.api_base,
            "api_type": self.api_type,
            "api_version": self.api_version
        }


class OpenAIBackend(LLMBackend):
    _config_type = OpenAIBackendConfig

    def completion(
        self,
        prompt: str,
        max_retries: int = 1,
        result_processor: Callable[[dict], str] = lambda resp: resp["choices"][0]["text"],
        **kwargs
    ) -> str:
        payload = self._construct_completion_payload(prompt=prompt, **kwargs)
        try:
            resp = openai.Completion.create(**payload)
        except openai.error.Timeout as e:
            if max_retries:
                max_retries -= 1
                return self.completion(prompt, max_retries, result_processor, **kwargs)
            raise
        except:
            raise
        else:
            return result_processor(convert_to_dict(resp))

    async def a_completion(
        self,
        prompt: str,
        max_retries: int = 1,
        result_processor: Callable[[dict], str] = lambda resp: resp["choices"][0]["text"],
        **kwargs
    ) -> str:
        payload = self._construct_completion_payload(prompt=prompt, **kwargs)
        try:
            resp = await openai.Completion.acreate(**payload)
        except openai.error.Timeout:
            if max_retries:
                max_retries -= 1
                return await self.a_completion(prompt, max_retries, result_processor, **kwargs)
            raise
        except:
            raise
        else:
            return result_processor(convert_to_dict(resp))

    def chat_completion(
        self,
        inputs: List[OpenAIBackendChatCompletionInput],
        max_retries: int = 1,
        result_processor: Callable[[dict], str] = lambda resp: resp["choices"][0]["message"]["content"],
        **kwargs
    ) -> str:
        payload = self._construct_chat_completion_payload(inputs=inputs, **kwargs)
        try:
            resp = openai.ChatCompletion.create(**payload)
        except openai.error.Timeout:
            if max_retries:
                max_retries -= 1
                return self.chat_completion(inputs, max_retries, result_processor, **kwargs)
            raise
        except:
            raise
        else:
            return result_processor(convert_to_dict(resp))

    async def a_chat_completion(
        self,
        inputs: List[OpenAIBackendChatCompletionInput],
        max_retries: int = 1,
        result_processor: Callable[[dict], str] = lambda resp: resp["choices"][0]["message"]["content"],
        **kwargs
    ) -> str:
        payload = self._construct_chat_completion_payload(inputs=inputs, **kwargs)
        try:
            resp = openai.ChatCompletion.acreate(**payload)
        except openai.error.Timeout:
            if max_retries:
                max_retries -= 1
                return await self.a_chat_completion(inputs, max_retries, result_processor, **kwargs)
            raise
        except:
            raise
        else:
            return result_processor(convert_to_dict(resp))

    def _construct_completion_payload(self, prompt: str, **kwargs):
        payload = {
            "prompt": prompt
        }
        payload.update(self.config.payload)
        if kwargs:
            payload.update(kwargs)
        return payload

    def _construct_chat_completion_payload(self, inputs: List[OpenAIBackendChatCompletionInput], **kwargs):
        payload = {
            "messages": [inp.model_dump() for inp in inputs]
        }
        payload.update(self.config.payload)
        if kwargs:
            payload.update(kwargs)
        return payload
