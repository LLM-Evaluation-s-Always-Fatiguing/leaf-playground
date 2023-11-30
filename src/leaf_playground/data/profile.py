from typing import Optional, Any
from uuid import uuid4, UUID

from pydantic import Field

from .base import Data
from ..utils.import_util import DynamicObject


class Role(Data):
    name: str = Field(default=...)
    description: str = Field(default=...)
    is_static: bool = Field(default=False)
    agent_type: Optional[DynamicObject] = Field(default=None)

    def model_post_init(self, __context: Any) -> None:
        if self.is_static and not self.agent_type:
            raise ValueError(f"agent_type should be specified when the role is a static role")


class Profile(Data):
    id: UUID = Field(default_factory=lambda: uuid4(), exclude=True)
    name: str = Field(default=...)
    role: Optional[Role] = Field(default=None)


__all__ = [
    "Role",
    "Profile"
]
