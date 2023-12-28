import asyncio
from typing import Any, Callable, Dict, Literal
from uuid import UUID

from ...data.log_body import LogBody
from ...utils.thread_util import run_asynchronously


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
        field_name: Literal["eval_records", "compare_records", "human_eval_records", "human_compare_records"]
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

    async def stream_logs(self, log_handler: Callable[[LogBody], Any] = print):
        cur = 0
        while not self._stopped:
            if cur >= len(self.logs):
                await asyncio.sleep(0.001)
            else:
                await run_asynchronously(log_handler, self.logs[cur])
                await asyncio.sleep(0.001)
                cur += 1
        for log in self.logs[cur:]:
            await run_asynchronously(log_handler, log)

    def stop(self):
        self._stopped = True


__all__ = ["Logger"]
