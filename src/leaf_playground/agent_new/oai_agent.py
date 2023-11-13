import os
from abc import abstractmethod
from typing import Any, Optional, Union, List
from uuid import UUID

import openai
from pydantic import Field

from .base import AgentConfig, Agent
from ..data.message import TextMessage
from ..utils.import_util import DynamicObject

DEFAULT_PROFILE_FORMAT_TEMPLATE = "Your name is {name}, you are a {role.name}, {role.description}"
DEFAULT_PROFILE_FORMAT_FIELDS = {"name", "role.name", "role.description"}

DEFAULT_MESSAGE_FORMAT_TEMPLATE = "{sender_name}({sender_role_name}): {content}"
DEFAULT_MESSAGE_FORMAT_FIELDS = {"sender_name", "sender_role_name", "content"}


class OpenAIAgentConfig(AgentConfig):
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
    completion_kwargs: Optional[dict] = Field(default=None)
    chat_completion_kwargs: Optional[dict] = Field(default=None)
    message_type_: DynamicObject = Field(
        default=DynamicObject(obj="TextMessage", module="leaf_playground.data.message"),
    )

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context=__context)
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


class OpenAIAgent(Agent):
    config_obj = OpenAIAgentConfig
    config: config_obj

    def __init__(self, config: config_obj):
        super().__init__(config=config)
        self.client = self.config.creat_client()
        self.async_client = self.config.creat_async_client()

    def _construct_completion_payload(self, prompt: str, **kwargs):
        payload = {
            "model": self.config.model,
            "prompt": prompt
        }
        payload.update(**self.config.completion_kwargs)
        if kwargs:
            payload.update(**kwargs)
        payload["stream"] = False
        return payload

    def _construct_chat_completion_payload(self, inputs: List[openai.types.chat.ChatCompletionMessageParam], **kwargs):
        payload = {
            "model": self.config.model,
            "messages": [inp.model_dump() for inp in inputs]
        }
        payload.update(**self.config.chat_completion_kwargs)
        if kwargs:
            payload.update(**kwargs)
        payload["stream"] = False
        return payload

    def completion(
        self,
        prompt: str,
        **kwargs
    ):
        payload = self._construct_completion_payload(prompt=prompt, **kwargs)
        resp = self.client.completions.create(**payload)
        return resp

    async def a_completion(
        self,
        prompt: str,
        **kwargs
    ):
        payload = self._construct_completion_payload(prompt=prompt, **kwargs)
        resp = await self.async_client.completions.create(**payload)
        return resp

    def chat_completion(
        self,
        inputs: List[openai.types.chat.ChatCompletionMessageParam],
        **kwargs
    ):
        payload = self._construct_chat_completion_payload(inputs=inputs, **kwargs)
        resp = self.client.chat.completions.create(**payload)
        return resp

    async def a_chat_completion(
        self,
        inputs: List[openai.types.chat.ChatCompletionMessageParam],
        **kwargs
    ):
        payload = self._construct_chat_completion_payload(inputs=inputs, **kwargs)
        resp = await self.async_client.chat.completions.create(**payload)
        return resp

    @abstractmethod
    def preprocess(self, messages: List[TextMessage]) -> Any:
        messages = "\n".join(
            [
                msg.format(DEFAULT_MESSAGE_FORMAT_TEMPLATE, DEFAULT_MESSAGE_FORMAT_FIELDS)
                for msg in messages
            ]
        )
        return messages

    @abstractmethod
    async def a_preprocess(self, messages: List[TextMessage]) -> str:
        messages = "\n".join(
            [
                msg.format(DEFAULT_MESSAGE_FORMAT_TEMPLATE, DEFAULT_MESSAGE_FORMAT_FIELDS)
                for msg in messages
            ]
        )
        return messages

    @abstractmethod
    def respond(
        self,
        input: str,
        *args,
        prefix: Optional[str] = None,
        **kwargs
    ) -> str:
        profile = self.profile.format(DEFAULT_PROFILE_FORMAT_TEMPLATE, DEFAULT_PROFILE_FORMAT_FIELDS)
        prefix = "" if not prefix else TextMessage(
            sender_id=self.id,
            sender_name=self.name,
            sender_role_name=self.role_name,
            content=prefix
        ).format(DEFAULT_MESSAGE_FORMAT_TEMPLATE, DEFAULT_MESSAGE_FORMAT_FIELDS)

        prompt = f"{profile}\n{input}\n{prefix}"

        resp = self.completion(prompt=prompt)
        return resp.choices[0].text

    @abstractmethod
    async def a_respond(
        self,
        input: str,
        *args,
        prefix: Optional[str] = None,
        **kwargs
    ) -> str:
        profile = self.profile.format(DEFAULT_PROFILE_FORMAT_TEMPLATE, DEFAULT_PROFILE_FORMAT_FIELDS)
        prefix = "" if not prefix else TextMessage(
            sender_id=self.id,
            sender_name=self.name,
            sender_role_name=self.role_name,
            content=prefix
        ).format(DEFAULT_MESSAGE_FORMAT_TEMPLATE, DEFAULT_MESSAGE_FORMAT_FIELDS)

        prompt = f"{profile}\n{input}\n{prefix}"

        resp = await self.a_completion(prompt=prompt)
        return resp.choices[0].text

    @abstractmethod
    def postprocess(
        self,
        response: str,
        response_prefix: Optional[str] = None,
        receiver_ids: Optional[List[UUID]] = None,
        receiver_names: Optional[List[str]] = None,
        receiver_role_names: Optional[List[str]] = None
    ) -> TextMessage:
        return TextMessage(
            sender_id=self.id,
            sender_name=self.name,
            sender_role_name=self.role_name,
            receiver_ids=receiver_ids,
            receiver_names=receiver_names,
            receiver_role_names=receiver_role_names,
            content=response
        )

    @abstractmethod
    def a_postprocess(
        self,
        response: str,
        response_prefix: Optional[str] = None,
        receiver_ids: Optional[List[UUID]] = None,
        receiver_names: Optional[List[str]] = None,
        receiver_role_names: Optional[List[str]] = None
    ) -> TextMessage:
        return TextMessage(
            sender_id=self.id,
            sender_name=self.name,
            sender_role_name=self.role_name,
            receiver_ids=receiver_ids,
            receiver_names=receiver_names,
            receiver_role_names=receiver_role_names,
            content=response
        )
