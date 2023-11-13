import os
from typing import Callable, Dict, List, Optional

import openai
from openai.util import convert_to_dict
from pydantic import Field

from .base import LLMBackendChatCompletionInput, LLMBackendConfig, LLMBackend
from ..utils.import_util import dynamically_import_fn, DynamicFn


class OpenAIBackendChatCompletionInput(LLMBackendChatCompletionInput):
    content: str = Field(default=...)
    role: str = Field(default=..., pattern=r"(system|user|assistant)")
    name: Optional[str] = Field(default=None, max_length=64, pattern=r"[0-9a-zA-Z_]{1,64}")


class OpenAIBackendConfig(LLMBackendConfig):
    model: str = Field(default=...)
    api_key_: Optional[str] = Field(default=None, alias="api_key")
    api_base_: Optional[str] = Field(default=None, alias="api_base")
    api_type_: Optional[str] = Field(default=None, alias="api_type")
    api_version_: Optional[str] = Field(default=None, alias="api_version")
    api_key_env_var: str = Field(default="OPENAI_API_KEY")
    api_base_env_var: str = Field(default="OPENAI_API_BASE")
    api_type_env_var: str = Field(default="OPENAI_API_TYPE")
    api_version_env_var: str = Field(default="OPENAI_API_VERSION")
    max_retries: int = Field(default=1)
    completion_result_processor: Optional[DynamicFn] = Field(default=None)
    chat_completion_result_processor: Optional[DynamicFn] = Field(default=None)
    completion_hyper_params: Optional[dict] = Field(default=None)
    chat_completion_hyper_params: Optional[dict] = Field(default=None)

    @property
    def api_key(self) -> str:
        if self.api_key_ is not None:
            return self.api_key_
        api_key = os.environ.get(self.api_key_env_var, None)
        if api_key is None:
            raise ValueError("openai api key neither be specified nor be found in environment variable.")
        return api_key

    @property
    def api_base(self) -> str:
        return self.api_base_ or os.environ.get(self.api_base_env_var, "https://api.openai.com/v1")

    @property
    def api_type(self):
        return self.api_type_ or os.environ.get(self.api_type_env_var, "open_ai")

    @property
    def api_version(self):
        return self.api_version_ or os.environ.get(
            self.api_version_env_var,
            ("2023-05-15" if self.api_type in ("azure", "azure_ad", "azuread") else None)
        )

    @property
    def payload(self) -> Dict[str, str]:
        return {
            "model": self.model,
            "api_key": self.api_key,
            "api_base": self.api_base,
            "api_type": self.api_type,
            "api_version": self.api_version
        }


class OpenAIBackend(LLMBackend):
    config_obj = OpenAIBackendConfig

    def __init__(self, config: config_obj):
        super().__init__(config=config)

        self._completion_result_processor = lambda resp: resp["choices"][0]["text"]
        self._chat_completion_result_processor = lambda resp: resp["choices"][0]["message"]["content"]
        if self.config.completion_result_processor:
            self._completion_result_processor = dynamically_import_fn(self.config.completion_result_processor)
        if self.config.chat_completion_result_processor:
            self._chat_completion_result_processor = dynamically_import_fn(self.config.chat_completion_result_processor)

    def completion(
        self,
        prompt: str,
        max_retries: Optional[int] = None,
        result_processor: Optional[Callable[[dict], str]] = None
    ) -> str:
        hyper_params = self.config.completion_hyper_params or dict()
        payload = self._construct_completion_payload(prompt=prompt, **hyper_params)
        if max_retries is None:
            max_retries = self.config.max_retries
        if result_processor is None:
            result_processor = self._completion_result_processor
        try:
            resp = openai.Completion.create(**payload)
        except openai.error.Timeout as e:
            if max_retries:
                max_retries -= 1
                return self.completion(prompt, max_retries, result_processor)
            raise
        except:
            raise
        else:
            return result_processor(convert_to_dict(resp))

    async def a_completion(
        self,
        prompt: str,
        max_retries: Optional[int] = None,
        result_processor: Optional[Callable[[dict], str]] = None
    ) -> str:
        hyper_params = self.config.completion_hyper_params or dict()
        payload = self._construct_completion_payload(prompt=prompt, **hyper_params)
        if max_retries is None:
            max_retries = self.config.max_retries
        if result_processor is None:
            result_processor = self._completion_result_processor
        try:
            resp = await openai.Completion.acreate(**payload)
        except openai.error.Timeout:
            if max_retries:
                max_retries -= 1
                return await self.a_completion(prompt, max_retries, result_processor)
            raise
        except:
            raise
        else:
            return result_processor(convert_to_dict(resp))

    def chat_completion(
        self,
        inputs: List[OpenAIBackendChatCompletionInput],
        max_retries: Optional[int] = None,
        result_processor: Callable[[dict], str] = None
    ) -> str:
        hyper_params = self.config.chat_completion_hyper_params or dict()
        payload = self._construct_chat_completion_payload(inputs=inputs, **hyper_params)
        if max_retries is None:
            max_retries = self.config.max_retries
        if result_processor is None:
            result_processor = self._chat_completion_result_processor
        try:
            resp = openai.ChatCompletion.create(**payload)
        except openai.error.Timeout:
            if max_retries:
                max_retries -= 1
                return self.chat_completion(inputs, max_retries, result_processor)
            raise
        except:
            raise
        else:
            return result_processor(convert_to_dict(resp))

    async def a_chat_completion(
        self,
        inputs: List[OpenAIBackendChatCompletionInput],
        max_retries: Optional[int] = None,
        result_processor: Callable[[dict], str] = None
    ) -> str:
        hyper_params = self.config.chat_completion_hyper_params or dict()
        payload = self._construct_chat_completion_payload(inputs=inputs, **hyper_params)
        if max_retries is None:
            max_retries = self.config.max_retries
        if result_processor is None:
            result_processor = self._chat_completion_result_processor
        try:
            resp = openai.ChatCompletion.acreate(**payload)
        except openai.error.Timeout:
            if max_retries:
                max_retries -= 1
                return await self.a_chat_completion(inputs, max_retries, result_processor)
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
