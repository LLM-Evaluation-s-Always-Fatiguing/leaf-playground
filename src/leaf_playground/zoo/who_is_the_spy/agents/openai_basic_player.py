from typing import List, Optional

import openai
import tiktoken
from pydantic import Field

from leaf_playground.ai_backend.openai import OpenAIBackendConfig, CHAT_MODELS
from leaf_playground.core.scene_info import SceneInfo
from leaf_playground.data.media import Text
from leaf_playground.data.profile import Profile, Role
from leaf_playground.utils.import_util import DynamicObject
from leaf_playground.zoo.who_is_the_spy.scene_agent import *
from leaf_playground.zoo.who_is_the_spy.scene_info import *


def calculate_num_tokens(text: str, tokenizer: tiktoken.Encoding) -> int:
    return len(tokenizer.encode(text))


def filter_messages(messages: List[MessageTypes], tokenizer: tiktoken.Encoding, max_tokens: int):
    total_tokens = 0
    index2len = {}
    for idx, msg in enumerate(messages):
        num_tokens = calculate_num_tokens(msg.content.text, tokenizer)
        total_tokens += num_tokens
        index2len[idx] = num_tokens
    if total_tokens <= max_tokens:
        return messages
    else:
        kept_messages = []
        for idx, msg in enumerate(messages):
            if idx < 3 or msg.sender_role == "moderator":
                kept_messages.append(msg)
                continue
            total_tokens -= index2len[idx]
            if total_tokens <= max_tokens:
                break
        kept_messages += messages[idx:]
        return kept_messages


class OpenAIBasicPlayerConfig(AIBasePlayerConfig):
    ai_backend_config: OpenAIBackendConfig = Field(default=...)
    ai_backend_obj: DynamicObject = Field(
        default=DynamicObject(obj="OpenAIBackend", module="leaf_playground.ai_backend.openai"),
        exclude=True
    )


class OpenAIBasicPlayer(AIBasePlayer):
    config_obj = OpenAIBasicPlayerConfig
    config: config_obj

    description: str = "AI player using only OpenAI API (without any additional strategy) to play the game."
    obj_for_import: DynamicObject = DynamicObject(
        obj="OpenAIBasicPlayer",
        module="leaf_playground.zoo.who_is_the_spy.agents.openai_basic_player"
    )

    def __init__(self, config: config_obj):
        super().__init__(config=config)

        self.language_model = self.config.ai_backend_config.model
        self.vision_model = "gpt-4-vision-preview"
        self.audio_model = "whisper-1"

        self.tokenizer: tiktoken.Encoding = tiktoken.encoding_for_model(self.language_model)
        self.client: openai.AsyncOpenAI = self.backend.async_client

        self.key_transcript = ""

    def post_init(self, role: Optional[Role], scene_info: SceneInfo):
        super().post_init(role, scene_info)
        self.key_modality: KeyModalities = self.scene_info.get_env_var("key_modality").current_value
        self.has_blank_slate: bool = self.scene_info.get_env_var("has_blank_slate").current_value

    def _prepare_chat_message(self, history: List[MessageTypes]) -> List[dict]:
        messages = [
            {
                "role": "system",
                "content": f"Your name is {self.name}, a player who is playing the Who is the Spy game."
            },
            {"role": "system", "content": history[0].content.text}
        ]
        for msg in history[1:]:
            content = msg.content.text
            if isinstance(msg, ModeratorKeyAssignment):
                content = content.replace(KEY_PLACEHOLDER, self.key_transcript)
            messages.append(
                {
                    "role": "user" if msg.sender_role != "moderator" else "system",
                    "content": content,
                    "name": msg.sender_name
                }
            )
        return messages

    def _prepare_completion_prompt(self, history: List[MessageTypes]) -> str:
        prompt = f"Your name is {self.name}, a player who is playing the Who is the Spy game.\n\n"
        for msg in history:
            content = msg.content.text
            if isinstance(msg, ModeratorKeyAssignment):
                content = content.replace(KEY_PLACEHOLDER, self.key_transcript)
            prompt += f"{msg.sender_name}: {content}\n\n"

        return prompt

    async def _respond(self, history: List[MessageTypes]) -> str:
        kept_history = filter_messages(history, self.tokenizer, self.config.context_max_tokens)
        if self.language_model in CHAT_MODELS:
            resp = await self.client.chat.completions.create(
                messages=self._prepare_chat_message(kept_history),
                model=self.language_model,
                max_tokens=64,
                temperature=0.9
            )
            response = resp.choices[0].message.content
        else:
            resp = await self.client.completions.create(
                prompt=self._prepare_completion_prompt(kept_history),
                model=self.language_model,
                max_tokens=64,
                temperature=0.9
            )
            response = resp.choices[0].text
        return response

    def reset_inner_status(self) -> None:
        self.key_transcript = ""

    async def receive_key(self, key_assignment_message: ModeratorKeyAssignment) -> None:
        if not key_assignment_message.key:
            return
        if self.key_modality == KeyModalities.TEXT:
            self.key_transcript = key_assignment_message.key.text
            return
        # TODO: other modalities

    async def describe_key(self, history: List[MessageTypes], receivers: List[Profile]) -> PlayerDescription:
        description = await self._respond(filter_messages(history, self.tokenizer, self.config.context_max_tokens))
        return PlayerDescription(
            sender=self.profile,
            receivers=receivers,
            content=Text(text=description)
        )

    async def predict_role(self, history: List[MessageTypes], moderator: Profile) -> PlayerPrediction:
        prediction = await self._respond(filter_messages(history, self.tokenizer, self.config.context_max_tokens))
        return PlayerPrediction(
            sender=self.profile,
            receivers=[moderator],
            content=Text(text=prediction)
        )

    async def vote(self, history: List[MessageTypes], moderator: Profile) -> PlayerVote:
        vote = await self._respond(filter_messages(history, self.tokenizer, self.config.context_max_tokens))
        return PlayerVote(
            sender=self.profile,
            receivers=[moderator],
            content=Text(text=vote)
        )


__all__ = [
    "OpenAIBasicPlayerConfig",
    "OpenAIBasicPlayer"
]
