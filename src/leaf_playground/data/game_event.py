from dataclasses import dataclass
from uuid import UUID

from .base import Data


@dataclass
class GameEvent(Data):
    id: UUID
    timestamp: float
    type: str
    description: str
    detail: Data
