from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import Field

from .base import Data
from .media import Media, MediaType
from .message import MessageType


class LogBody(Data):
    index: int = Field(default=...)
    created_at: datetime = Field(default_factory=lambda: datetime.utcnow())
    references: Optional[List[MessageType]] = Field(default=None)
    response: MessageType = Field(default=...)
    ground_truth: Optional[Media] = Field(default=None)
    eval_record: Optional[dict] = Field(default=None)
    narrator: Optional[str] = Field(default=None)


__all__ = ["LogBody"]
