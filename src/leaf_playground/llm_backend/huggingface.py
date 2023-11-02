from typing import Callable, Dict, List, Optional, Union

import torch
from pydantic import Field
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline, GenerationConfig

from .base import LLMBackend, LLMBackendConfig, LLMBackendChatCompletionInput
from ..utils.import_util import dynamically_import_fn, DynamicFn


class HuggingFaceBackendChatCompletionInput(LLMBackendChatCompletionInput):
    pass


class HuggingFaceBackendConfig(LLMBackendConfig):
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
    batch_size: int = Field(default=-1)
    generation_config: dict = Field(default={"max_new_tokens": 32, "num_beams": 1, "do_sample": True})
    result_processor: Optional[DynamicFn] = Field(default=None)


class HuggingFaceBackend(LLMBackend):
    config_obj = HuggingFaceBackendConfig

    def __init__(self, config: config_obj):
        super().__init__(config=config)

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_name_or_path or self.config.model_name_or_path,
            use_fast=self.config.use_fast_tokenizer,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=self.config.trust_remote_code
        )
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id

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
            trust_remote_code=self.config.trust_remote_code
        )

        self._pipeline = TextGenerationPipeline(
            model=model,
            tokenizer=tokenizer,
            batch_size=self.config.batch_size
        )
        self._result_processor = lambda res: [each["generated_text"] for each in res][0]
        if self.config.result_processor:
            self._result_processor = dynamically_import_fn(self.config.result_processor)

    def completion(
        self,
        prompt: str,
        result_processor: Optional[Callable[[List[dict]], str]] = None
    ) -> str:
        res_proc = self._result_processor
        if result_processor:
            res_proc = result_processor
        gen_conf = self.config.generation_config
        res = self._pipeline(
            prompt,
            return_full_text=False,
            handle_long_generation="hole",
            generation_config=GenerationConfig(**gen_conf)
        )
        return res_proc(res)

    async def a_completion(
        self,
        prompt: str,
        result_processor: Optional[Callable[[List[dict]], str]] = None
    ) -> str:
        return self.completion(prompt, result_processor=result_processor)

    def chat_completion(
        self,
        inputs: List[HuggingFaceBackendChatCompletionInput],
        result_processor: Optional[Callable[[List[dict]], str]] = None
    ) -> str:
        raise NotImplementedError()

    async def a_chat_completion(
        self,
        inputs: List[HuggingFaceBackendChatCompletionInput],
        result_processor: Optional[Callable[[List[dict]], str]] = None
    ) -> str:
        raise NotImplementedError()
