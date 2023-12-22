import os
from typing import Any, Literal, Optional, Union

import openai
from pydantic import Field

from .base import AIBackend, AIBackendConfig


COMPLETION_MODELS = [
    "babbage-002",
    "davinci-002",
    "gpt-3.5-turbo-instruct",
    "text-davinci-003",
    "text-davinci-002",
    "text-davinci-001",
    "code-davinci-002",
    "text-curie-001",
    "text-babbage-001",
    "text-ada-001",
]
CHAT_MODELS = [
    "gpt-4-1106-preview",
    "gpt-4-vision-preview",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-32k",
    "gpt-4-32k-0314",
    "gpt-4-32k-0613",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k-0613",
]


_ValidModels = Literal[
    "babbage-002",
    "davinci-002",
    "gpt-3.5-turbo-instruct",
    "text-davinci-003",
    "text-davinci-002",
    "text-davinci-001",
    "code-davinci-002",
    "text-curie-001",
    "text-babbage-001",
    "text-ada-001",
    "gpt-4-1106-preview",
    "gpt-4-vision-preview",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-32k",
    "gpt-4-32k-0314",
    "gpt-4-32k-0613",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k-0613",
]


class OpenAIBackendConfig(AIBackendConfig):
    model: _ValidModels = Field(default=...)
    api_key: Optional[str] = Field(default=None)
    organization: Optional[str] = Field(default=None)
    base_url: Optional[str] = Field(default=None)
    azure_endpoint: Optional[str] = Field(default=None)
    azure_deployment: Optional[str] = Field(default=None)
    api_version: Optional[str] = Field(default=None)
    is_azure: bool = Field(default=False)
    max_retries: int = Field(default=2)
    timeout: float = Field(default=60)

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
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


class OpenAIBackend(AIBackend):
    config_cls = OpenAIBackendConfig
    config: config_cls

    def __init__(self, config: config_cls):
        super().__init__(config=config)
        self.client = self.config.creat_client()
        self.async_client = self.config.creat_async_client()


__all__ = [
    "COMPLETION_MODELS",
    "CHAT_MODELS",
    "OpenAIBackendConfig",
    "OpenAIBackend"
]
