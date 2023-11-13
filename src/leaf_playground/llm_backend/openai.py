import os
from typing import Callable, Dict, List, Optional, Any, Union

import openai
from pydantic import Field

from .base import LLMBackendChatCompletionInput, LLMBackendConfig, LLMBackend
from ..utils.import_util import dynamically_import_fn, DynamicFn


class OpenAIBackendConfig(LLMBackendConfig):
    model: str = Field(default=...)
    api_key: Optional[str] = Field(default=None)
    organization: Optional[str] = Field(default=None)
    base_url: Optional[str] = Field(default=None)
    azure_endpoint: Optional[str] = Field(default=None)
    azure_deployment: Optional[str] = Field(default=None)
    api_version: Optional[str] = Field(default=None)
    is_azure: bool = Field(default=False)
    max_retries: int = Field(default=2)
    timeout: float = Field(default=60)
    completion_result_processor: Optional[DynamicFn] = Field(default=None)
    chat_completion_result_processor: Optional[DynamicFn] = Field(default=None)
    completion_kwargs: Optional[dict] = Field(default=None)
    chat_completion_kwargs: Optional[dict] = Field(default=None)

    def model_post_init(self, __context: Any) -> None:
        if not self.api_key:
            self.api_key = os.environ.get("AZURE_OPENAI_API_KEY" if self.is_azure else "OPENAI_API_KEY", None)
        if not self.api_key and not self.is_azure:
            raise ValueError(
                "Must provide the `api_key` argument, or the `OPENAI_API_KEY` environment variable "
                "when is_azure=False"
            )
        if not self.organization and not self.is_azure:
            self.organization = os.environ.get("OPENAI_ORG_ID", None)
        if not self.base_url and self.is_azure:
            if not self.azure_endpoint:
                self.azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", None)
            if self.azure_endpoint is None:
                raise ValueError(
                    "Must provide one of the `base_url` or `azure_endpoint` arguments, "
                    "or the `AZURE_OPENAI_ENDPOINT` environment variable when is_azure=True"
                )
        if self.base_url and self.azure_endpoint and self.is_azure:
            raise ValueError("base_url and azure_endpoint are mutually exclusive when is_azure=True")

    @property
    def client_init_payload(self):
        payload = {
            "api_key": self.api_key,
            "organization": self.organization,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        if self.is_azure:
            payload.update(
                azure_endpoint=self.azure_endpoint,
                azure_deployment=self.azure_deployment,
                api_version=self.api_version
            )
        return payload

    def creat_client(self) -> Union[openai.OpenAI, openai.AzureOpenAI]:
        obj = openai.OpenAI if not self.is_azure else openai.AzureOpenAI
        return obj(**self.client_init_payload)

    def creat_async_client(self) -> Union[openai.AsyncOpenAI, openai.AsyncAzureOpenAI]:
        obj = openai.AsyncOpenAI if not self.is_azure else openai.AsyncAzureOpenAI
        return obj(**self.client_init_payload)


class OpenAIBackend(LLMBackend):
    config_obj = OpenAIBackendConfig
    config: OpenAIBackendConfig

    def __init__(self, config: config_obj):
        super().__init__(config=config)

        self._client = self.config.creat_client()
        self._async_client = self.config.creat_async_client()
        self._completion_result_processor = lambda resp: resp.choices[0].text
        self._chat_completion_result_processor = lambda resp: resp.choices[0].message.content
        if self.config.completion_result_processor:
            self._completion_result_processor = dynamically_import_fn(self.config.completion_result_processor)
        if self.config.chat_completion_result_processor:
            self._chat_completion_result_processor = dynamically_import_fn(self.config.chat_completion_result_processor)

    def completion(
        self,
        prompt: str,
        **kwargs
    ) -> str:
        payload = self._construct_completion_payload(prompt=prompt, **kwargs)
        resp = self._client.completions.create(**payload)
        return self._completion_result_processor(resp)

    async def a_completion(
        self,
        prompt: str,
        **kwargs
    ) -> str:
        payload = self._construct_completion_payload(prompt=prompt, **kwargs)
        resp = await self._async_client.completions.create(**payload)
        return self._completion_result_processor(resp)

    def chat_completion(
        self,
        inputs: List[openai.types.chat.ChatCompletionMessageParam],
        **kwargs
    ) -> str:
        payload = self._construct_chat_completion_payload(inputs=inputs, **kwargs)
        resp = self._client.chat.completions.create(**payload)
        return self._chat_completion_result_processor(resp)

    async def a_chat_completion(
        self,
        inputs: List[openai.types.chat.ChatCompletionMessageParam],
        **kwargs
    ) -> str:
        payload = self._construct_chat_completion_payload(inputs=inputs, **kwargs)
        resp = await self._async_client.chat.completions.create(**payload)
        return self._chat_completion_result_processor(resp)

    def _construct_completion_payload(self, prompt: str, **kwargs):
        payload = {
            "model": self.config.model,
            "prompt": prompt
        }
        payload.update(**self.config.completion_kwargs)
        if kwargs:
            payload.update(**kwargs)
        return payload

    def _construct_chat_completion_payload(self, inputs: List[openai.types.chat.ChatCompletionMessageParam], **kwargs):
        payload = {
            "model": self.config.model,
            "messages": [inp.model_dump() for inp in inputs]
        }
        payload.update(**self.config.chat_completion_kwargs)
        if kwargs:
            payload.update(**kwargs)
        return payload
