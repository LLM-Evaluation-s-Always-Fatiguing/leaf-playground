from typing import Dict, List, Optional, Union, Any

from pydantic import Field

from .base import AIBackend, AIBackendConfig
from .._config import _Config
from ..data.tool import HFTool


class HFLocalModelConfig(_Config):
    model_name_or_path: str = Field(default=...)
    tokenizer_name_or_path: Optional[str] = Field(default=None)
    revision: str = Field(default="main")
    low_cpu_mem_usage: bool = Field(default=True)
    torch_dtype: str = Field(default="float16", pattern="(float16|bfloat16)")
    device: Optional[Union[int, str]] = Field(default=None)
    max_memory: Optional[Dict[Union[str, int], str]] = Field(default=None)
    device_map: Union[str, Dict[str, Union[int, str]]] = Field(default="auto")
    use_fast_tokenizer: bool = Field(default=False)
    trust_remote_code: bool = Field(default=False)
    use_safetensors: bool = Field(default=False)

    def prepare_model_tokenizer(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_name_or_path or self.config.model_name_or_path,
            use_fast=self.config.use_fast_tokenizer,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=self.config.trust_remote_code,
        )
        # TODO: specify pad_token_id to eos_token_id if it not exists using new way that transformers now supports

        max_memory = self.config.max_memory
        if max_memory:
            max_memory = {(eval(k) if isinstance(k, str) else k): v for k, v in max_memory.items()}

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.config.model_name_or_path,
            torch_dtype=getattr(torch, self.config.torch_dtype),
            device_map=self.config.device_map,
            max_memory=max_memory,
            low_cpu_mem_usage=self.config.low_cpu_mem_usage,
            revision=self.config.revision,
            trust_remote_code=self.config.trust_remote_code,
        )

        return model, tokenizer


class HFRemoteEndpointConfig(_Config):
    url_endpoint: str = Field(default=...)
    token: Optional[str] = Field(default=None)


class OpenAIConfig(_Config):
    api_key: Optional[str] = Field(default=None)
    model: str = Field(default="text-davinci-003")


class HFAgentBackendConfig(AIBackendConfig):
    remote_endpoint_config: Optional[HFRemoteEndpointConfig] = Field(default=None)
    local_model_config: Optional[HFLocalModelConfig] = Field(default=None)
    openai_config: Optional[OpenAIConfig] = Field(default=None)
    chat_prompt_template: Optional[str] = Field(default=None)
    run_prompt_template: Optional[str] = Field(default=None)
    additional_tools: Optional[List[HFTool]] = Field(default=None)

    def model_post_init(self, __context: Any) -> None:
        if not self.remote_endpoint_config and not self.local_model_config and not self.openai_config:
            raise ValueError(
                "Must provide one of the `remote_endpoint_config`, `local_model_config`, or `openai_config` arguments"
            )
        if sum([bool(self.remote_endpoint_config), bool(self.local_model_config), bool(self.openai_config)]) > 1:
            raise ValueError(
                "`remote_endpoint_config`, `local_model_config` and `openai_config` are mutually exclusive"
            )

    def create_hf_agent(self):
        from transformers import HfAgent, Tool

        payload = {"chat_prompt_template": self.chat_prompt_template, "run_prompt_template": self.run_prompt_template}
        if self.additional_tools:
            payload["additional_tools"] = [Tool.from_hub(**tool.model_dump()) for tool in self.additional_tools]
        if self.remote_endpoint_config:
            payload.update(**self.hf_endpoint.model_dump())
        elif self.local_model_config:
            model, tokenizer = self.local_model_config.prepare_model_tokenizer()
            payload.update(model=model, tokenizer=tokenizer)
        else:
            payload.update(**self.openai_config.model_dump())

        return HfAgent(**payload)


class HFPipelineBackendConfig(HFLocalModelConfig):
    def prepare_pipeline(self):
        from transformers import TextGenerationPipeline

        model, tokenizer = self.prepare_model_tokenizer()
        return TextGenerationPipeline(model=model, tokenizer=tokenizer)


class HFAgentBackend(AIBackend):
    config_cls = HFAgentBackendConfig
    config: config_cls

    def __init__(self, config: config_cls):
        super().__init__(config=config)
        self.hf_agent = self.config.create_hf_agent()


class HFPipelineBackend(AIBackend):
    config_cls = HFPipelineBackendConfig
    config: config_cls

    def __init__(self, config: config_cls):
        super().__init__(config=config)
        self.pipeline = self.config.prepare_pipeline()


__all__ = [
    "HFLocalModelConfig",
    "HFRemoteEndpointConfig",
    "OpenAIConfig",
    "HFAgentBackendConfig",
    "HFAgentBackend",
    "HFPipelineBackendConfig",
    "HFPipelineBackend",
]
