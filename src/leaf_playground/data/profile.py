from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

from .base import Data


@dataclass
class Role(Data):
    name: str
    description: str


@dataclass
class Profile(Data):
    id: UUID
    name: str
    role: Optional[Role] = field(default=None)


ROLE_TYPES = {"base": Role}
PROFILE_TYPES = {"base": Profile}
