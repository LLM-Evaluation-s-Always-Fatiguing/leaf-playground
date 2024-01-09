import json
import warnings
import os
from typing import Dict, Literal, Any
from uuid import UUID

import pandas as pd
from pydantic import BaseModel, Field

from ...data.log_body import LogBody, ActionLogBody


class Logger:
    def __init__(self):
        self._logs = []
        self._id2log = {}
        self._stopped = False

        self._handlers = []

    def registry_handler(self, handler):
        self._handlers.append(handler)

    def add_log(self, log_body: LogBody):
        self._logs.append(log_body)
        self._id2log[log_body.id] = log_body

        for handler in self._handlers:
            handler.notify_create(log_body)

    def add_action_log_record(
        self,
        log_id: UUID,
        records: Dict[str, dict],
        field_name: Literal["eval_records", "compare_records", "human_eval_records", "human_compare_records"],
    ):
        log_body = self._id2log[log_id]
        for name, record in records.items():
            if "human" not in field_name:
                getattr(log_body, field_name)[name].append(record)
            else:
                getattr(log_body, field_name)[name] = record

        for handler in self._handlers:
            handler.notify_update(log_body)

    @property
    def logs(self):
        return self._logs


_KEPT_LOG_FILE_NAME = ".log"


class LogExporter(BaseModel):
    file_name: str = Field(default="log")
    extension: Literal["json", "jsonl", "csv"] = Field(default="jsonl")

    def model_post_init(self, __context: Any) -> None:
        if self.file_name in _KEPT_LOG_FILE_NAME:
            raise ValueError(f"can't set file_name to {_KEPT_LOG_FILE_NAME}")
        if self.extension not in ["json", "jsonl", "csv"]:
            raise NotImplementedError(f"file extension {self.extension} isn't support yet.")

    def _export_to_json(self, logger: Logger, save_path: str):
        logs = [log.model_dump(mode="json") for log in logger.logs]
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=4, ensure_ascii=False)

    def _export_to_jsonl(self, logger: Logger, save_path: str):
        with open(save_path, "w", encoding="utf-8") as f:
            for log in logger.logs:
                f.write(log.model_dump_json() + "\n")

    def _export_to_csv(self, logger: Logger, save_path: str):
        data = {"log_id": [], "sender": [], "receivers": [], "message": []}
        for log in [log for log in logger.logs if isinstance(log, ActionLogBody)]:
            response = log.response
            data["log_id"].append(log.id.hex)
            data["sender"].append(response.sender_name)
            data["receivers"].append([receiver.name for receiver in response.receivers])
            data["message"].append(response.content.display_text)
        pd.DataFrame(data).to_csv(save_path, encoding="utf-8", index=False)

    def export(self, logger: Logger, save_dir: str):
        save_path = os.path.join(save_dir, f"{self.file_name}.{self.extension}")
        func_name = f"_export_to_{self.extension}"
        try:
            getattr(self, func_name)(logger, save_path)
        except:
            if os.path.exists(save_path):
                os.remove(save_path)
            warnings.warn(f"{self.__class__.__name__}.{func_name} export log failed.")


class _KeptLogExporter(LogExporter):
    file_name: str = Field(default=_KEPT_LOG_FILE_NAME)

    def model_post_init(self, __context: Any) -> None:
        pass


__all__ = ["Logger", "LogExporter"]
