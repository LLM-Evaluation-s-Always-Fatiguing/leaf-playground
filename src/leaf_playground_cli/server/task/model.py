import asyncio
from datetime import datetime
from enum import Enum
from typing import List

from sqlmodel import SQLModel, Field, Column, Relationship, DateTime, JSON

from leaf_playground import __version__ as leaf_version
from leaf_playground.core.scene_engine import SceneEngineState
from leaf_playground.data.log_body import LogBody, LogType
from leaf_playground.data.message import Message as LEAFMessage
from leaf_playground.utils.import_util import DynamicObject


class TaskRunTimeEnv(Enum):
    LOCAL = "local"
    DOCKER = "docker"


class TaskDBLifeCycle(Enum):
    LIVING = "living"
    DELETING = "deleting"


class Task(SQLModel):
    id: str = Field(default=...)
    project_id: str = Field(default=...)
    project_version: str = Field(default=...)
    leaf_version: str = Field(default=leaf_version)
    port: int = Field(default=...)
    host: str = Field(default="http://127.0.0.1")
    payload: str = Field(default=...)
    status: str = Field(default=SceneEngineState.PENDING.value)
    runtime_env: TaskRunTimeEnv = Field(default=TaskRunTimeEnv.LOCAL)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    results_dir: str = Field(default=...)

    @property
    def base_http_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def base_ws_url(self) -> str:
        return f"ws://{self.host}:{self.port}"


class TaskTable(Task, table=True):
    __tablename__ = "task"

    id: str = Field(default=..., primary_key=True, index=True)
    project_id: str = Field(default=..., index=True)
    status: str = Field(default=SceneEngineState.PENDING.value, index=True)
    runtime_env: TaskRunTimeEnv = Field(default=TaskRunTimeEnv.LOCAL, index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, sa_column=Column(DateTime, index=True))

    logs: List["LogTable"] = Relationship(back_populates="task")
    messages: List["MessageTable"] = Relationship(back_populates="task")
    task_results: "TaskResultsTable" = Relationship(back_populates="task")

    live_cycle: TaskDBLifeCycle = Field(default=TaskDBLifeCycle.LIVING)

    def to_task(self) -> Task:
        return Task(**self.model_dump())

    @classmethod
    def from_task(cls, task: Task) -> "TaskTable":
        return cls(**task.model_dump())


class TaskResults(SQLModel):
    __tablename__ = "task_results"

    id: str = Field(default=...)
    scene_config: dict = Field(default=...)
    evaluator_configs: list = Field(default=...)
    metrics: dict = Field(default=...)
    charts: dict = Field(default=...)
    logs: dict = Field(default=...)


class TaskResultsTable(TaskResults, table=True):
    id: str = Field(default=..., primary_key=True, foreign_key="task.id", index=True)
    scene_config: dict = Field(default=..., sa_column=Column(JSON))
    evaluator_configs: list = Field(default=..., sa_column=Column(JSON))
    metrics: dict = Field(default=..., sa_column=Column(JSON))
    charts: dict = Field(default=..., sa_column=Column(JSON))
    logs: dict = Field(default=..., sa_column=Column(JSON))

    task: TaskTable = Relationship(back_populates="task_results")

    def to_task_results(self) -> TaskResults:
        return TaskResults(**self.model_dump())

    @classmethod
    def from_task_results(cls, task_results: TaskResults) -> "TaskResultsTable":
        return cls(**task_results.model_dump())


class Log(SQLModel):
    __tablename__ = "log"

    id: str = Field(default=...)
    log_type: LogType = Field(default=...)
    log_msg: str = Field(default=...)
    created_at: datetime = Field(default=...)
    last_update: datetime = Field(default=...)
    db_last_update: datetime = Field(default_factory=datetime.utcnow)
    tid: str = Field(default=...)
    data: dict = Field(default=...)

    @classmethod
    def init_from_log_body(cls, log: LogBody, tid: str) -> "Log":
        log_dict = {"tid": tid, "data": {}}
        for k, v in log.model_dump(mode="json", by_alias=True).items():
            if k in ["id", "log_type", "log_msg", "created_at", "last_update"]:
                log_dict[k] = getattr(log, k)
            else:
                log_dict["data"][k] = v
        return cls(**log_dict)


class LogTable(Log, table=True):
    id: str = Field(default=..., primary_key=True, index=True)
    created_at: datetime = Field(default=..., sa_column=Column(DateTime, index=True))
    last_update: datetime = Field(default=..., sa_column=Column(DateTime, index=True))
    db_last_update: datetime = Field(default_factory=datetime.utcnow, sa_column=Column(DateTime, index=True))
    tid: str = Field(default=..., foreign_key="task.id", index=True)
    data: dict = Field(default=..., sa_column=Column(JSON))

    task: TaskTable = Relationship(back_populates="logs")

    def to_log(self) -> Log:
        return Log(**self.model_dump())

    @classmethod
    def from_log(cls, log: Log) -> "LogTable":
        return cls(**log.model_dump())


class Message(SQLModel):
    __tablename__ = "message"

    id: str = Field(default=...)
    tid: str = Field(default=...)
    data: dict = Field(default=...)
    obj: dict = Field(default=...)

    @classmethod
    def init_from_message(cls, message: LEAFMessage, tid: str) -> "Message":
        msg_dict = {
            "id": message.id,
            "tid": tid,
            "data": message.model_dump(mode="json", by_alias=True),
            "obj": DynamicObject.create_dynamic_obj(message.__class__).model_dump(mode="json", by_alias=True),
        }

        return cls(**msg_dict)


class MessageTable(Message, table=True):
    id: str = Field(default=..., primary_key=True)
    tid: str = Field(default=..., foreign_key="task.id", index=True)
    data: dict = Field(default=..., sa_column=Column(JSON))
    obj: dict = Field(default=..., sa_column=Column(JSON))

    task: TaskTable = Relationship(back_populates="messages")

    def to_message(self) -> Message:
        return Message(**self.model_dump())

    @classmethod
    def from_message(cls, message: Message) -> "MessageTable":
        return cls(**message.model_dump())


__all__ = [
    "TaskRunTimeEnv",
    "TaskDBLifeCycle",
    "Task",
    "TaskTable",
    "TaskResults",
    "TaskResultsTable",
    "Log",
    "LogTable",
    "Message",
    "MessageTable"
]
