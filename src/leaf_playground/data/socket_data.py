from enum import Enum

from pydantic import Field

from .base import Data


class SocketDataType(Enum):
    LOG = "log"


class SocketData(Data):
    type: SocketDataType = Field(default=...)
    data: dict = Field(default=...)


__all__ = [
    "SocketDataType",
    "SocketData",
]