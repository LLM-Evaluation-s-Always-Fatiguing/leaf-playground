from typing import Optional, Any
from uuid import uuid4, UUID

from pydantic import Field

from .base import Data
from ..utils.import_util import DynamicObject


class Role(Data):
    name: str = Field(default=...)
    description: str = Field(default=...)
    is_static: bool = Field(default=False)


class Profile(Data):
    id: UUID = Field(default_factory=lambda: uuid4())
    name: str = Field(default=...)
    role: Optional[Role] = Field(default=None)


__all__ = [
    "Role",
    "Profile"
]
