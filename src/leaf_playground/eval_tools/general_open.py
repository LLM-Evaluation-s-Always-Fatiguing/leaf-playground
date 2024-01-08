from typing import Literal, Optional, Dict

from jinja2 import Template
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
from pydantic import Field

from .base import EvalTool, EvalToolConfig
from ..ai_backend.openai import OpenAIBackend, OpenAIBackendConfig


class EvalOpenAIBackendConfig(OpenAIBackendConfig):
    model: Literal["gpt-4-1106-preview", "gpt-4"] = Field(default="gpt-4-1106-preview")


class GeneralOpenEvalToolConfig(EvalToolConfig):
    openai_backend_config: EvalOpenAIBackendConfig = Field(...)


class GeneralOpenEvalTool(EvalTool):
    config_obj = GeneralOpenEvalToolConfig
    config: config_obj

    temperature: float
    response_format: Dict[Literal["type"], Literal["text", "json"]]
    max_tokens: int

    def __init__(self, config: config_obj):
        super().__init__(config)
        self.client = OpenAIBackend.from_config(self.config.openai_backend_config).async_client
        self.temperature = 0.25
        self.response_format = {"type": "text"}
        self.max_tokens = 128

    def set_temperature(self, temperature: float):
        self.temperature = temperature

    def set_response_format(self, response_format: Literal["text", "json"]):
        self.response_format = {"type": response_format}

    def set_max_tokens(self, max_tokens: int):
        self.max_tokens = max_tokens

    async def evaluate(self, system_template: Optional[str], prompt_template: str, value_dict: dict) -> str:
        messages = []
        if system_template:
            tpl = Template(system_template)
            messages.append(ChatCompletionSystemMessageParam(role="system", content=tpl.render(value_dict)))
        tpl = Template(prompt_template)
        messages.append(ChatCompletionUserMessageParam(role="user", content=tpl.render(value_dict)))

        result = (
            (
                await self.client.chat.completions.create(
                    messages=messages,
                    model=self.config.openai_backend_config.model,
                    max_tokens=self.max_tokens,
                    response_format=self.response_format,
                    temperature=self.temperature,
                )
            )
            .choices[0]
            .message.content
        )

        return result
