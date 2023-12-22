from enum import Enum

from pydantic import Field

from .base import Data


class SocketOperation(Enum):
    CREATE = "create"
    UPDATE = "update"


class SocketData(Data):
    data: dict = Field(default=...)
    operation: SocketOperation = Field(default=SocketOperation.CREATE)


__all__ = [
    "SocketData",
    "SocketOperation"
]
