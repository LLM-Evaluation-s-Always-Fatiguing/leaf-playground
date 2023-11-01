from typing import Optional
from uuid import uuid4, UUID

from pydantic import Field

from .base import Data


class Role(Data):
    name: str = Field(default=...)
    description: str = Field(default=...)


class Profile(Data):
    id: UUID = Field(default_factory=lambda: uuid4())
    name: str = Field(default=...)
    role: Optional[Role] = Field(default=None)
