from dataclasses import dataclass
from typing import List
from uuid import UUID

from .base import Data


@dataclass
class Profile(Data):
    id: UUID
    name: str


@dataclass
class Role(Data):
    name: str
    description: str


PROFILE_TYPES = {"base": Profile}
ROLE_TYPES = {"base": Role}
