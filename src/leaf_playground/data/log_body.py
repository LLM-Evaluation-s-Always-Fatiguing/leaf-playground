from collections import defaultdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Literal, Optional, Any
from uuid import uuid4, UUID

from pydantic import model_validator, Field

from .base import Data
from .media import Media


_MetricName = str
_MetricRecord = dict
_MetricRecords = List[_MetricRecord]
_MessageID = str


class LogType(Enum):
    ACTION = "action"
    SYSTEM = "system"


class LogBody(Data):
    id: str = Field(default_factory=lambda: "log_" + uuid4().hex[:8])
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_update: datetime = Field(default=None)
    log_type: LogType = Field(default=...)
    log_msg: str = Field(default=...)

    def model_post_init(self, __context: Any) -> None:
        if self.last_update is None:
            self.last_update = self.created_at

    @model_validator(mode="before")
    def set_log_type(cls, values):
        log_type = values.get("log_type", None)
        if isinstance(log_type, str):
            log_type = log_type.lower()
            if log_type == "action":
                log_type = LogType.ACTION
            if log_type == "system":
                log_type = LogType.SYSTEM
            values["log_type"] = log_type
        return values


class ActionLogBody(LogBody):
    log_type: Literal[LogType.ACTION] = Field(default=LogType.ACTION)
    references: Optional[List[_MessageID]] = Field(default=None)
    response: _MessageID = Field(default=...)
    action_belonged_chain: Optional[str] = Field(default=...)
    ground_truth: Optional[Media] = Field(default=None)
    eval_records: Dict[_MetricName, _MetricRecords] = Field(default=defaultdict(list))
    compare_records: Dict[_MetricName, _MetricRecords] = Field(default=defaultdict(list))
    human_eval_records: Dict[_MetricName, _MetricRecord] = Field(default={})
    human_compare_records: Dict[_MetricName, _MetricRecord] = Field(default={})


class SystemEvent(Enum):
    SIMULATION_START = "simulation_start"
    SIMULATION_PAUSED = "simulation_paused"
    SIMULATION_RESUME = "simulation_resume"
    SIMULATION_FAILED = "simulation_failed"
    SIMULATION_INTERRUPTED = "simulation_interrupted"
    SIMULATION_FINISHED = "simulation_finished"
    EVALUATION_FINISHED = "evaluation_finished"
    EVERYTHING_DONE = "everything_done"


class SystemLogBody(LogBody):
    log_type: Literal[LogType.SYSTEM] = Field(default=LogType.SYSTEM)
    log_msg: Optional[str] = Field(default=None)
    system_event: SystemEvent = Field(default=...)

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        if not self.log_msg:
            self.log_msg = self.system_event.value


__all__ = ["LogBody", "LogType", "ActionLogBody", "SystemLogBody", "SystemEvent"]
