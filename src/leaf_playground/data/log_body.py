from collections import defaultdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Literal, Optional, Any
from uuid import uuid4, UUID

from pydantic import Field

from .base import Data
from .media import Media
from .message import MessageType


_MetricName = str
_MetricRecords = List[dict]


class LogType(Enum):
    ACTION = "action"
    SYSTEM = "system"


class LogBody(Data):
    id: UUID = Field(default_factory=lambda: uuid4())
    created_at: datetime = Field(default_factory=lambda: datetime.utcnow())
    log_type: LogType = Field(default=...)
    log_msg: str = Field(default=...)


class ActionLogBody(LogBody):
    log_type: Literal[LogType.ACTION] = Field(default=LogType.ACTION)
    references: Optional[List[MessageType]] = Field(default=None)
    response: MessageType = Field(default=...)
    ground_truth: Optional[Media] = Field(default=None)
    eval_records: Dict[_MetricName, _MetricRecords] = Field(default=defaultdict(list))
    compare_records: Dict[_MetricName, _MetricRecords] = Field(default=defaultdict(list))
    human_eval_records: Dict[_MetricName, _MetricRecords] = Field(default=defaultdict(list))  # TODO: how to use
    human_compare_records: Dict[_MetricName, _MetricRecords] = Field(default=defaultdict(list))


class SystemEvent(Enum):
    SIMULATION_START = "simulation_start"
    SIMULATION_FINISHED = "simulation_finished"
    EVALUATION_FINISHED = "evaluation_finished"
    EVERYTHING_DONE = "everything_done"


class SystemLogBody(LogBody):
    log_type: Literal[LogType.SYSTEM] = Field(default=LogType.SYSTEM)
    log_msg: Optional[str] = Field(default=None)
    system_event: SystemEvent = Field(default=...)

    def model_post_init(self, __context: Any) -> None:
        if not self.log_msg:
            self.log_msg = self.system_event.value


__all__ = [
    "LogBody",
    "LogType",
    "ActionLogBody",
    "SystemLogBody",
    "SystemEvent"
]
