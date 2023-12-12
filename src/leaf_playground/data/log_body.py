from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import Field

from .base import Data
from .media import Media
from .message import MessageType


_MetricName = str
_MetricRecords = List[dict]  # TODO: is frontend need schema?


class LogBody(Data):
    index: int = Field(default=...)
    created_at: datetime = Field(default_factory=lambda: datetime.utcnow())
    references: Optional[List[MessageType]] = Field(default=None)
    response: MessageType = Field(default=...)
    ground_truth: Optional[Media] = Field(default=None)
    eval_records: Dict[_MetricName, _MetricRecords] = Field(
        default_factory=lambda: defaultdict(list)
    )
    human_eval_records: Dict[_MetricName, _MetricRecords] = Field(
        default_factory=lambda: defaultdict(list)
    )  # TODO: how to use
    narrator: Optional[str] = Field(default=None)


__all__ = ["LogBody"]
