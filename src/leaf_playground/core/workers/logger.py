import asyncio
import json
import warnings
import os
from abc import abstractmethod
from typing import Any, Dict, List, Literal, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field

from ...data.log_body import LogBody, ActionLogBody
from ...data.message import MessagePool
from ...utils.thread_util import run_asynchronously
from ..._type import Singleton


class Logger(Singleton):
    def __init__(self):
        self.message_pool: MessagePool = MessagePool.get_instance()

        self._logs: Dict[str, LogBody] = {}
        self._stopped: bool = False
        self._handlers: List["LogHandler"] = []

    def registry_handler(self, handler: "LogHandler"):
        self._handlers.append(handler)

    def add_log(self, log_body: LogBody):
        self._logs[log_body.id] = log_body

        for handler in self._handlers:
            asyncio.ensure_future(run_asynchronously(handler.notify_create, log_body))

    def add_action_log_record(
        self,
        log_id: str,
        records: Dict[str, dict],
        field_name: Literal["eval_records", "compare_records", "human_eval_records", "human_compare_records"],
    ):
        log_body = self._logs[log_id]
        for name, record in records.items():
            if "human" not in field_name:
                getattr(log_body, field_name)[name].append(record)
            else:
                getattr(log_body, field_name)[name] = record

        for handler in self._handlers:
            asyncio.ensure_future(run_asynchronously(handler.notify_update, log_body))

    @property
    def logs(self) -> List[LogBody]:
        return list(self._logs.values())

    def is_log_exists(self, log_id: str) -> bool:
        return log_id in self._logs

    def get_log_by_id(self, log_id: str) -> LogBody:
        return self._logs[log_id]


class LogHandler:
    @abstractmethod
    async def notify_create(self, log: LogBody):
        pass

    @abstractmethod
    async def notify_update(self, log: LogBody):
        pass


_KEPT_LOG_FILE_NAME = ".log"


class LogExporter(BaseModel):
    file_name: str = Field(default="log")
    extension: Literal["json", "jsonl", "csv"] = Field(default="jsonl")

    def model_post_init(self, __context: Any) -> None:
        if self.file_name in _KEPT_LOG_FILE_NAME:
            raise ValueError(f"can't set file_name to {_KEPT_LOG_FILE_NAME}")
        if self.extension not in ["json", "jsonl", "csv"]:
            raise NotImplementedError(f"file extension {self.extension} isn't support yet.")

    def _export_to_json(self, logger: Logger) -> Union[List[dict], dict]:
        logs = []
        for log in logger.logs:
            log_dict = log.model_dump(mode="json")
            if isinstance(log, ActionLogBody):
                if log.references is not None:
                    log_dict["references"] = (
                        [logger.message_pool.get_message_by_id(ref).model_dump(mode="json") for ref in log.references]
                        if log.references
                        else None
                    )
                log_dict["response"] = logger.message_pool.get_message_by_id(log.response).model_dump(mode="json")
            logs.append(log_dict)

        return logs

    def _export_to_jsonl(self, logger: Logger) -> List[dict]:
        return self._export_to_json(logger)

    def _export_to_csv(self, logger: Logger) -> dict:
        data = {"log_id": [], "sender": [], "receivers": [], "message": []}
        for log in [log for log in logger.logs if isinstance(log, ActionLogBody)]:
            response = logger.message_pool.get_message_by_id(log.response)
            data["log_id"].append(log.id)
            data["sender"].append(response.sender_name)
            data["receivers"].append([receiver.name for receiver in response.receivers])
            data["message"].append(response.content.display_text)
        return data

    def export(self, logger: Logger, save_dir: Optional[str] = None) -> Optional[Union[dict, List[dict]]]:
        func_name = f"_export_to_{self.extension}"
        try:
            logs = getattr(self, func_name)(logger)
        except:
            warnings.warn(f"{self.__class__.__name__}.{func_name} export log failed.")
            return None
        else:
            if not save_dir:
                return logs
            save_path = os.path.join(save_dir, f"{self.file_name}.{self.extension}")
            if self.extension == "csv":
                pd.DataFrame(logs).to_csv(save_path, encoding="utf-8", index=False)
            elif self.extension == "json":
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(logs, f, indent=4, ensure_ascii=False)
            else:
                with open(save_path, "w", encoding="utf-8") as f:
                    for log in logs:
                        f.write(json.dumps(log) + "\n")
            return logs


class _KeptLogExporter(LogExporter):
    file_name: str = Field(default=_KEPT_LOG_FILE_NAME)

    def model_post_init(self, __context: Any) -> None:
        pass


__all__ = ["Logger", "LogHandler", "LogExporter"]
