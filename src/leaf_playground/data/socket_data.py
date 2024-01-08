from enum import Enum
from typing import Literal

from pydantic import Field

from .base import Data


class SocketOperation(Enum):
    CREATE = "create"
    UPDATE = "update"


class SocketData(Data):
    data: dict = Field(default=...)
    operation: SocketOperation = Field(default=SocketOperation.CREATE)
    type: Literal["socket_data"] = Field(default="data")


class SocketEvent(Data):
    event: str = Field(default=...)
    type: Literal["socket_event"] = Field(default="event")


__all__ = ["SocketData", "SocketEvent", "SocketOperation"]
