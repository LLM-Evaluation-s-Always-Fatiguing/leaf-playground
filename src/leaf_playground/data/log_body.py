from datetime import datetime
from typing import Optional, Set

from pydantic import Field

from .base import Data


class LogBody(Data):
    time: datetime = Field(default_factory=lambda: datetime.utcnow())
    event: Optional[dict] = Field(default=None)
    message: Optional[dict] = Field(default=None)

    def format(
        self,
        template: str,
        fields: Set[str]
    ) -> str:
        data = {f: self.__getattribute__(f) for f in fields}
        return template.format(**data)
