from typing import Optional
from uuid import uuid4

from pydantic import Field

from .base import Data


class Role(Data):
    name: str = Field(default=...)
    description: str = Field(default=...)
    is_static: bool = Field(default=False)


class Profile(Data):
    id: str = Field(default_factory=lambda: "agent_" + uuid4().hex[:8])
    name: str = Field(default=...)
    role: Optional[Role] = Field(default=None)


__all__ = [
    "Role",
    "Profile"
]
