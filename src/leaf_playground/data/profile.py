from typing import Optional
from uuid import UUID

from pydantic import Field

from .base import Data


class Role(Data):
    name: str = Field(default=...)
    description: str = Field(default=...)


class Profile(Data):
    id: UUID = Field(default=...)
    name: str = Field(default=...)
    role: Optional[Role] = Field(default=None)


ROLE_TYPES = {"base": Role}
PROFILE_TYPES = {"base": Profile}
