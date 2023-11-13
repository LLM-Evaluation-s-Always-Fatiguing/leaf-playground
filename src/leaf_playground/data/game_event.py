from uuid import UUID

from pydantic import Field

from .base import Data


class GameEvent(Data):
    id: UUID = Field(default=...)
    timestamp: float = Field(default=...)
    type: str = Field(default=...)
    description: str = Field(default=...)
    detail: Data = Field(default=...)
