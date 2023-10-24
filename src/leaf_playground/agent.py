from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional
from uuid import UUID

from .data.message import Message, TextMessage
from .data.profile import Profile


@dataclass
class Agent:
    profile: Profile

    @property
    def id(self) -> UUID:
        return self.profile.id

    @property
    def name(self) -> str:
        return self.profile.name

    @abstractmethod
    def act(self) -> Message:
        pass


@dataclass
class LLMAgent(Agent):
    llm_backend: Callable[[str, Optional[dict]], str]
    request_hyper_params: Optional[dict] = None

    def _respond(self, prompt: str) -> TextMessage:
        return TextMessage(
            sender_id=self.id,
            sender_name=self.name,
            content=self.llm_backend(prompt, self.request_hyper_params),
            timestamp=datetime.utcnow().timestamp()
        )

    def act(self) -> TextMessage:
        pass
