from enum import Enum

from pydantic import Field

from .base import Data


class SocketDataType(Enum):
    LOG = "log"
    METRIC = "metric"
    SUMMARY = "summary"
    ENDING = "ending"


class SocketOperation(Enum):
    CREATE = "create"
    UPDATE = "update"


class SocketData(Data):
    type: SocketDataType = Field(default=...)
    data: dict = Field(default=...)
    operation: SocketOperation = Field(default=SocketOperation.CREATE)


__all__ = [
    "SocketDataType",
    "SocketData",
    "SocketOperation"
]
