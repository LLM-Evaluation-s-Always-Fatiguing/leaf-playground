import asyncio
import io
import json
import random
import traceback
from packaging import version

import pandas as pd
import websockets
import zipfile
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

import aiohttp
from fastapi import status, APIRouter, BackgroundTasks, Depends, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from leaf_playground import __version__ as leaf_version
from leaf_playground._type import Singleton
from leaf_playground.core.scene_engine import (
    SceneEngineState,
    SceneObjConfig,
    MetricEvaluatorObjsConfig,
    ReporterObjConfig,
)
from leaf_playground.data.log_body import LogType
from leaf_playground.data.socket_data import SocketData, SocketOperation

from .db import *
from .model import *
from .runtime_env import RunTimeEnv
from .runtime_env import *
from ..hub import Hub
from ..utils import get_local_ip
from ...utils.debug_utils import DebuggerConfig


task_router = APIRouter(prefix="/task")


class LogEvalMetricRecord(BaseModel):
    value: Any = Field(default=...)
    reason: Optional[str] = Field(default=None)
    metric_name: str = Field(default=...)


class LogEvalCompareRecord(LogEvalMetricRecord):
    value: List[str] = Field(default=...)


class TaskCreationPayload(BaseModel):
    project_id: str = Field(default=...)
    scene_obj_config: SceneObjConfig = Field(default=...)
    metric_evaluator_objs_config: MetricEvaluatorObjsConfig = Field(default=...)
    reporter_obj_config: ReporterObjConfig = Field(default=...)

    @property
    def workdir(self):
        return Hub.get_instance().get_project(self.project_id).work_dir


class TaskManager(Singleton):
    def __init__(
        self,
        server_port: int = 8000,
        server_host: str = "127.0.0.1",
        runtime_env: TaskRunTimeEnv = TaskRunTimeEnv.LOCAL,
        debugger_config: DebuggerConfig = DebuggerConfig(port=3457),
        evaluator_debugger_config: DebuggerConfig = DebuggerConfig(port=3458),
    ):
        self.hub: Hub = Hub.get_instance()
        self.db: DB = DB.get_instance()
        self.http_session = aiohttp.ClientSession()

        self.server_port = server_port
        self.server_host = server_host
        self._task_ports = set(list(range(10000, 65535))) - {server_port}
        self._port_lock = Lock()

        if debugger_config.debug and runtime_env != TaskRunTimeEnv.LOCAL:
            raise NotImplementedError(f"running projects in {runtime_env.value} with debug mode is not supported yet.")

        self.runtime_env = runtime_env
        self.runtime_cls = RUNTIME_LOOKUP_TABLE[self.runtime_env]
        self.debugger_config = debugger_config
        self.evaluator_debugger_config = evaluator_debugger_config

        self._tasks: Dict[str, RunTimeEnv] = {}

    def acquire_port(self) -> int:
        with self._port_lock:
            port = random.choice(list(self._task_ports))
            self._task_ports -= {port}
        return port

    def release_port(self, port: int) -> None:
        with self._port_lock:
            self._task_ports.add(port)

    def _check_version_compatible(self, origin: str, current: str):
        origin_version = version.Version(origin)
        current_version = version.Version(current)
        return origin_version.major == current_version.major and origin_version.minor == current_version.minor

    async def create_task(self, payload: TaskCreationPayload) -> Task:
        tid = f"task_{payload.project_id}_" + datetime.utcnow().strftime("%Y%m%d%H%M%S") + "_" + uuid4().hex[:8]
        project = self.hub.get_project(payload.project_id)
        if not self._check_version_compatible(project.leaf_version, leaf_version):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"project [{project.id}]'s leaf_version [{project.leaf_version}] "
                       f"not compatible with current leaf_version [{leaf_version}]"
            )
        task = Task(
            id=tid,
            project_id=payload.project_id,
            project_version=project.version,
            port=self.acquire_port(),
            host=get_local_ip(),
            payload=payload.model_dump_json(by_alias=True),
            runtime_env=self.runtime_env,
        )
        self._tasks[tid] = self.runtime_cls(task, self)

        await self._tasks[tid].start()

        return task

    async def get_task(self, task_id: str) -> Task:
        return await self.db.get_task(task_id)

    async def get_and_validate_task(self, task_id: str, secret_key: str) -> Task:
        task = await self.get_task(task_id)
        if secret_key != self._tasks[task.id].secret_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="incorrect secret key, not allowed to update task status.",
            )
        return task

    async def get_history_tasks(self) -> List[Task]:
        return await self.db.get_task_by_life_cycle(life_cycle=TaskDBLifeCycle.LIVING)

    async def update_task_status(self, task_id: str, task_status: str, secret_key: str):
        await self.get_and_validate_task(task_id, secret_key)
        await self.db.update_task_status(task_id, task_status)
        if task_status in [SceneEngineState.FAILED.value, SceneEngineState.INTERRUPTED.value]:
            self._tasks[task_id].shutdown_event.set()

    async def get_task_agents_connected(self, task_id: str) -> dict:
        task = await self.get_task(task_id)
        url = f"{task.base_http_url}/agents_connected"

        async with self.http_session.get(url=url) as resp:
            if resp.status != 200:
                raise HTTPException(status_code=resp.status, detail=await resp.text())
            return await resp.json()

    async def _control_task_status(self, task_id: str, action: Literal["pause", "resume", "interrupt", "close"]):
        task = await self.get_task(task_id)
        url = f"{task.base_http_url}/{action}"

        async with self.http_session.post(url=url) as resp:
            if resp.status != 200:
                raise HTTPException(status_code=resp.status, detail=await resp.text())

    async def ping_task_service(self, task_id: str) -> str:
        task = await self.get_task(task_id)
        url = f"{task.base_http_url}/hello"

        async with self.http_session.get(url=url) as resp:
            if resp.status != 200:
                raise HTTPException(status_code=resp.status, detail=await resp.text())
            return await resp.text()

    async def pause_task(self, task_id: str):
        await self._control_task_status(task_id, "pause")

    async def resume_task(self, task_id: str):
        await self._control_task_status(task_id, "resume")

    async def interrupt_task(self, task_id: str):
        await self._control_task_status(task_id, "interrupt")

    async def close_task(self, task_id: str):
        await self._control_task_status(task_id, "close")
        self._tasks[task_id].shutdown_event.set()

    async def delete_task(self, task_id: str):
        await self.db.delete_task(task_id)

    async def save_task_results(self, task_id: str):
        task = await self.get_task(task_id)
        url = f"{task.base_http_url}/save"

        async with self.http_session.post(url=url) as resp:
            if resp.status != 200:
                raise HTTPException(status_code=resp.status, detail=await resp.text())

    async def save_task_results_to_db(self, task_id: str, secret_key: str, task_results: TaskResults):
        await self.get_and_validate_task(task_id, secret_key)
        await self.db.save_task_results(task_results)

    async def get_task_results(self, task_id: str):
        return await self.db.get_task_results_by_id(task_id)

    async def insert_task_log(self, task_id: str, secret_key: str, log: Log):
        await self.get_and_validate_task(task_id, secret_key)
        await self.db.insert_log(log)

    async def update_task_log(self, task_id: str, secret_key: str, log: Log):
        await self.get_and_validate_task(task_id, secret_key)
        await self.db.update_log(log)

    async def _transform_log(self, log: Log) -> dict:
        log_data = log.model_dump(mode="json")
        log_data.update(**log_data.pop("data"))
        if log.log_type == LogType.ACTION:
            if log_data["references"]:
                references = []
                for ref in log_data["references"]:
                    references.append(await self.db.get_message_by_id(ref))
                log_data["references"] = [ref.data for ref in references]
            response = await self.db.get_message_by_id(log_data["response"])
            log_data["response"] = response.data
        return log_data

    async def get_logs_paginate(
        self, task_id: str, skip: int = 0, limit: int = 20, log_type: Optional[LogType] = None
    ) -> List[dict]:
        logs = await self.db.get_logs_by_tid_paginate(task_id, skip, limit, log_type)
        if not logs:
            return []

        logs_data = []
        for log in logs:
            logs_data.append(await self._transform_log(log))

        return logs_data

    async def count_logs(self, task_id: str, log_type: Optional[LogType] = None) -> int:
        return await self.db.count_num_logs_by_tid(task_id, log_type)

    async def websocket_connection(self, task_id: str, websocket: WebSocket, human_id: Optional[str] = None):
        async def _get_and_send_logs():
            nonlocal last_check_time

            logs = await self.db.get_logs_by_tid_with_update_time_constraint(task_id, last_checked_dt=last_check_time)
            for log in logs:
                log_data = await self._transform_log(log)
                socket_operation = (
                    SocketOperation.CREATE if log.created_at == log.last_update else SocketOperation.UPDATE
                )
                if last_check_time is None:
                    socket_operation = SocketOperation.CREATE
                await websocket.send_json(
                    SocketData(data=log_data, operation=socket_operation).model_dump(mode="json")
                )

            if logs:
                last_check_time = logs[-1].db_last_update

        async def _stream_log():
            while True:
                try:
                    await _get_and_send_logs()
                    await asyncio.sleep(0.001)
                except (WebSocketDisconnect, asyncio.CancelledError):
                    break
                except:
                    traceback.print_exc()
                    await websocket.close(code=500, reason=traceback.format_exc())

        last_check_time = None
        await self.get_task(task_id)

        task = await self.get_task(task_id)
        stream_log_task = asyncio.ensure_future(_stream_log())

        if not human_id:
            while True:
                try:
                    text = await websocket.receive_text()
                    if text == "disconnect":
                        raise WebSocketDisconnect()
                except WebSocketDisconnect:
                    stream_log_task.cancel()
                    return
        else:
            async with websockets.connect(f"{task.base_ws_url}/ws/human/{human_id}") as connection:
                while True:
                    try:
                        event = json.loads(await connection.recv())
                        await websocket.send_json(event)
                        if event["event"] == "wait_human_input":
                            human_input = await websocket.receive_text()
                            await connection.send(human_input)
                    except WebSocketDisconnect:
                        stream_log_task.cancel()
                        return
                    except:
                        traceback.print_exc()
                        return

    async def update_task_log_metric_record(
        self, task_id: str, log_id: str, agent_id: str, record: LogEvalMetricRecord
    ):
        task = await self.get_task(task_id)
        url = f"{task.base_http_url}/logs/{log_id}/record/metric/update?agent_id={agent_id}"

        async with self.http_session.post(
            url=url, headers={"Content-Type": "application/json"}, json=record.model_dump(mode="json", by_alias=True)
        ) as resp:
            if resp.status != 200:
                raise HTTPException(status_code=resp.status, detail=await resp.text())

    async def update_task_log_compare_record(self, task_id: str, log_id: str, record: LogEvalCompareRecord):
        task = await self.get_task(task_id)
        url = f"{task.base_http_url}/logs/{log_id}/record/compare/update"

        async with self.http_session.post(
            url=url, headers={"Content-Type": "application/json"}, json=record.model_dump(mode="json", by_alias=True)
        ) as resp:
            if resp.status != 200:
                raise HTTPException(status_code=resp.status, detail=await resp.text())

    async def insert_task_message(self, task_id: str, secret_key: str, message: Message):
        await self.get_and_validate_task(task_id, secret_key)
        await self.db.insert_message(message)

    @classmethod
    async def http_get_task_by_id(cls, task_id: str) -> Task:
        try:
            db: DB = cls.get_instance()
        except KeyError as e:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="service not ready.")

        return await db.get_task(task_id)

    @classmethod
    async def http_get_task_results_by_id(cls, task_id: str) -> TaskResults:
        return await TaskManager.get_instance().get_task_results(task_id)


@task_router.post("/create", response_model=Task, response_model_include={"id"})
async def create_task(
    payload: TaskCreationPayload, task_manager: TaskManager = Depends(TaskManager.get_instance)
) -> Task:
    return await task_manager.create_task(payload)


@task_router.get("/{task_id}/hello")
async def ping_task_service(task_id: str, task_manager: TaskManager = Depends(TaskManager.get_instance)):
    return await task_manager.ping_task_service(task_id)


@task_router.post("/{task_id}/pause")
async def pause_task(task_id: str, task_manager: TaskManager = Depends(TaskManager.get_instance)):
    await task_manager.pause_task(task_id)


@task_router.post("/{task_id}/resume")
async def resume_task(task_id: str, task_manager: TaskManager = Depends(TaskManager.get_instance)):
    await task_manager.resume_task(task_id)


@task_router.post("/{task_id}/interrupt")
async def interrupt_task(task_id: str, task_manager: TaskManager = Depends(TaskManager.get_instance)):
    await task_manager.interrupt_task(task_id)


@task_router.post("/{task_id}/close")
async def close_task(task_id: str, task_manager: TaskManager = Depends(TaskManager.get_instance)):
    await task_manager.close_task(task_id)


@task_router.post("/{task_id}/delete")
async def delete_task(
    task_id: str, background_tasks: BackgroundTasks, task_manager: TaskManager = Depends(TaskManager.get_instance)
):
    background_tasks.add_task(task_manager.delete_task, task_id)


@task_router.get("/{task_id}/agents_connected", response_class=JSONResponse)
async def get_task_agents_connected(
    task_id: str, task_manager: TaskManager = Depends(TaskManager.get_instance)
) -> JSONResponse:
    return JSONResponse(content=await task_manager.get_task_agents_connected(task_id))


@task_router.get("/{task_id}/payload", response_model=TaskCreationPayload)
async def get_task_payload(task: Task = Depends(TaskManager.http_get_task_by_id)) -> TaskCreationPayload:
    return TaskCreationPayload(**json.loads(task.payload))


@task_router.get("/{task_id}/status", response_class=JSONResponse)
async def get_task_status(task: Task = Depends(TaskManager.http_get_task_by_id)) -> JSONResponse:
    return JSONResponse(content={"status": task.status})


@task_router.patch("/{task_id}/status")
async def update_task_status(
    task_id: str, task_status: str, secret_key: str, task_manager: TaskManager = Depends(TaskManager.get_instance)
):
    await task_manager.update_task_status(task_id=task_id, task_status=task_status, secret_key=secret_key)


@task_router.post("/{task_id}/save")
async def save_task_results(task_id: str, task_manager: TaskManager = Depends(TaskManager.get_instance)):
    await task_manager.save_task_results(task_id)


@task_router.post("/{task_id}/results/save")
async def save_task_results_to_db(
    task_id: str,
    secret_key: str,
    task_results: TaskResults,
    task_manager: TaskManager = Depends(TaskManager.get_instance),
):
    await task_manager.save_task_results_to_db(task_id, secret_key, task_results)


@task_router.get("/{task_id}/results/download")
async def download_task_results(
    task_results: TaskResults = Depends(TaskManager.http_get_task_results_by_id),
):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
        zip_file.writestr(
            "scene_config.json", json.dumps(task_results.scene_config, ensure_ascii=False)
        )
        zip_file.writestr(
            "evaluator_configs.json", json.dumps(task_results.evaluator_configs, ensure_ascii=False)
        )
        zip_file.writestr("metrics.json", json.dumps(task_results.metrics, ensure_ascii=False))
        zip_file.writestr("charts.json", json.dumps(task_results.charts, ensure_ascii=False))
        for name, logs in task_results.logs.items():
            ext = name.split(".")[-1]
            if ext == "json":
                zip_file.writestr(name, json.dumps(logs, ensure_ascii=False))
            if ext == "jsonl":
                zip_file.writestr(name, "\n".join([json.dumps(log, ensure_ascii=False) for log in logs]))
            if ext == "csv":
                zip_file.writestr(name, pd.DataFrame(logs).to_csv(index=False, encoding="utf_8_sig"))  # TODO: fix Chinese

    zip_buffer.seek(0)

    # 创建流式响应
    return StreamingResponse(zip_buffer, media_type="application/x-zip-compressed")


@task_router.get(
    "/{task_id}/metrics_and_charts",
    response_model=TaskResults,
    response_model_include={"metrics", "charts"}
)
async def get_metrics_and_charts(
    task_results: TaskResults = Depends(TaskManager.http_get_task_results_by_id)
):
    return task_results


@task_router.get("/history", response_class=JSONResponse)
async def get_history_tasks(task_manager: TaskManager = Depends(TaskManager.get_instance)):
    tasks = await task_manager.get_history_tasks()
    return JSONResponse(
        content=[
            task.model_dump(mode="json", include={"id", "project_id", "status", "created_at"})
            for task in tasks
        ]
    )


@task_router.post("/{task_id}/logs/insert")
async def insert_log(
    task_id: str, secret_key: str, log: Log, task_manager: TaskManager = Depends(TaskManager.get_instance)
):
    await task_manager.insert_task_log(task_id, secret_key, log)


@task_router.patch("/{task_id}/logs/update")
async def update_log(
    task_id: str, secret_key: str, log: Log, task_manager: TaskManager = Depends(TaskManager.get_instance)
):
    await task_manager.update_task_log(task_id, secret_key, log)


@task_router.get("/{task_id}/logs", response_class=JSONResponse)
async def get_logs_paginate(
    task_id: str,
    skip: int = 0,
    limit: int = 20,
    log_type: Optional[LogType] = None,
    task_manager: TaskManager = Depends(TaskManager.get_instance),
) -> JSONResponse:
    return JSONResponse(content=await task_manager.get_logs_paginate(task_id, skip, limit, log_type))


@task_router.get("/{task_id}/logs/count", response_class=JSONResponse)
async def count_num_logs(
    task_id: str, log_type: Optional[LogType] = None, task_manager: TaskManager = Depends(TaskManager.get_instance)
):
    return JSONResponse(content={"count": await task_manager.count_logs(task_id, log_type)})


@task_router.websocket("/{task_id}/logs/ws")
async def stream_logs(
    task_id: str, websocket: WebSocket, task_manager: TaskManager = Depends(TaskManager.get_instance)
):
    await websocket.accept()
    await task_manager.websocket_connection(task_id, websocket)


@task_router.websocket("/{task_id}/human/{agent_id}/ws")
async def human_connection(
    task_id: str, agent_id: str, websocket: WebSocket, task_manager: TaskManager = Depends(TaskManager.get_instance)
):
    await websocket.accept()
    await task_manager.websocket_connection(task_id, websocket, agent_id)


# TODO: support updating log records after task finished and task service closed.


@task_router.post("/{task_id}/logs/{log_id}/record/metric/update")
async def update_metric_record(
    task_id: str,
    log_id: str,
    agent_id: str,
    record: LogEvalMetricRecord,
    task_manager: TaskManager = Depends(TaskManager.get_instance),
):
    await task_manager.update_task_log_metric_record(task_id, log_id, agent_id, record)


@task_router.post("/{task_id}/logs/{log_id}/record/compare/update")
async def update_compare_record(
    task_id: str,
    log_id: str,
    record: LogEvalCompareRecord,
    task_manager: TaskManager = Depends(TaskManager.get_instance),
):
    await task_manager.update_task_log_compare_record(task_id, log_id, record)


@task_router.post("/{task_id}/messages/insert")
async def insert_message(
    task_id: str, secret_key: str, message: Message, task_manager: TaskManager = Depends(TaskManager.get_instance)
):
    await task_manager.insert_task_message(task_id, secret_key, message)


__all__ = [
    "task_router",
    "TaskCreationPayload",
    "TaskManager",
    "LogEvalMetricRecord",
    "LogEvalCompareRecord"
]
