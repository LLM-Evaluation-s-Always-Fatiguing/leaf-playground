import asyncio
import json
import os
import random
import subprocess
import sys
import traceback
import websockets
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Literal, Optional, Union
from uuid import uuid4

import aiohttp
from fastapi import status, APIRouter, BackgroundTasks, Depends, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from leaf_playground._type import Singleton
from leaf_playground.core.scene_engine import (
    SceneEngineState, SceneObjConfig, MetricEvaluatorObjsConfig, ReporterObjConfig
)
from leaf_playground.data.log_body import LogType
from leaf_playground.data.socket_data import SocketData, SocketOperation

from .db import *
from .model import *
from ..hub import Hub
from ..utils import get_local_ip


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


class TaskProxy:
    def __init__(self, tid: str):
        self.id = tid
        self.db: DB = DB.get_instance()

        self._secret_key: str = uuid4().hex
        self._shutdown_event: asyncio.Event = asyncio.Event()
        self._runtime_id: Union[int, str] = None  # pid or docker container name

        asyncio.create_task(self.maybe_shutdown(), name=f"{self.id}_shutdown")

    async def maybe_shutdown(self):
        await self._shutdown_event.wait()
        await asyncio.sleep(3)

        pid = self.runtime_id
        task = await self.db.get_task(self.id)
        if task.runtime_env == "docker":
            subprocess.run(f"docker stop {pid}".split())

    async def get_task(self) -> Task:
        return await self.db.get_task(self.id)

    @property
    def secret_key(self) -> str:
        return self._secret_key

    @property
    def shutdown_event(self) -> asyncio.Event:
        return self._shutdown_event

    @property
    def runtime_id(self) -> int:
        return self._runtime_id

    @runtime_id.setter
    def runtime_id(self, pid: int):
        self._runtime_id = pid


class TaskManager(Singleton):
    def __init__(
        self,
        server_port: int = 8000,
        server_host: str = "127.0.0.1",
        runtime_env: TaskRunTimeEnv = TaskRunTimeEnv.LOCAL
    ):
        self.hub: Hub = Hub.get_instance()
        self.db: DB = DB.get_instance()
        self.http_session = aiohttp.ClientSession()
        self.result_dir = os.path.join(self.hub.hub_dir, ".leaf_workspace", "results")
        self.tmp_dir = os.path.join(self.hub.hub_dir, ".leaf_workspace", "tmp")
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.tmp_dir, exist_ok=True)

        self.server_port = server_port
        self.server_host = server_host
        self._task_ports = set(list(range(10000, 65535))) - {server_port}
        self._port_lock = Lock()

        self.runtime_env = runtime_env

        self._tasks: Dict[str, TaskProxy] = {}

    def acquire_port(self) -> int:
        with self._port_lock:
            port = random.choice(list(self._task_ports))
            self._task_ports -= {port}
        return port

    def release_port(self, port: int) -> None:
        with self._port_lock:
            self._task_ports.add(port)

    async def create_task(self, payload: TaskCreationPayload) -> Task:
        tid = f"task_{payload.project_id}_" + datetime.utcnow().strftime("%Y%m%d%H%M%S") + "_" + uuid4().hex[:8]
        task = Task(
            id=tid,
            project_id=payload.project_id,
            port=self.acquire_port(),
            host=get_local_ip(),
            payload=payload.model_dump_json(by_alias=True),
            runtime_env=self.runtime_env,
            results_dir=os.path.join(self.result_dir, tid)
        )
        self._tasks[tid] = TaskProxy(tid)

        await self.db.insert_task(task)

        if task.runtime_env == "docker":
            self._tasks[tid].runtime_id = self._create_task_in_docker(task)
        else:
            self._tasks[tid].runtime_id = self._create_task_in_local(task)

        return task

    def _create_task_in_local(self, task: Task):
        work_dir = self.hub.get_project(json.loads(task.payload)["project_id"]).work_dir
        process = subprocess.Popen(
            (
                f"{sys.executable} {os.path.join(work_dir, '.leaf', 'app.py')} "
                f"--id {task.id} "
                f"--port {task.port} "
                f"--host {task.host} "
                f"--secret_key {self._tasks[task.id].secret_key} "
                f"--server_url http://{self.server_host}:{self.server_port} "
            ).split()
        )
        return process.pid

    def _create_task_in_docker(self, task: Task):
        container_name = f"leaf-scene-{task.id}"
        image_name = f"leaf-scene-{self.hub.get_project(json.loads(task.payload)['project_id']).name}"
        work_dir = "/app"

        image_check = subprocess.run(f"docker images -q {image_name}", shell=True, capture_output=True, text=True)
        if not image_check.stdout.strip():
            print("Image not found, building Docker image...")
            subprocess.run(f"cd {work_dir} && docker build . -t {image_name}", shell=True)

        subprocess.Popen(
            (
                f"docker run --rm "
                f"-p {task.port}:{task.port} "
                f"-v {task.results_dir}:/tmp/result "
                f"--name {container_name} "
                f"{image_name} .leaf/app.py "
                f"--id {task.id} "
                f"--port {task.port} "
                f"--host 0.0.0.0 "
                f"--secret_key {self._tasks[task.id].secret_key} "
                f"--server_url http://{self.server_host}:{self.server_port} "
                f"--docker"
            ).split()
        )
        return container_name

    async def get_task(self, task_id: str) -> Task:
        return await self.db.get_task(task_id)

    async def get_and_validate_task(self, task_id: str, secret_key: str) -> Task:
        task = await self.get_task(task_id)
        if secret_key != self._tasks[task.id].secret_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="incorrect secret key, not allowed to update task status."
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

    async def insert_task_log(self, task_id: str, secret_key: str, log: Log):
        await self.get_and_validate_task(task_id, secret_key)
        await self.db.insert_log(log)

    async def update_task_log(self, task_id: str, secret_key: str, log: Log):
        await self.get_and_validate_task(task_id, secret_key)
        await self.db.update_log(log)

    async def websocket_connection(
        self,
        task_id: str,
        websocket: WebSocket,
        human_id: Optional[str] = None
    ):
        async def _get_and_send_logs():
            nonlocal last_check_time

            logs = await self.db.get_logs_by_tid_with_update_time_constraint(task_id, last_checked_dt=last_check_time)
            for log in logs:
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
                socket_operation = (
                    SocketOperation.CREATE if log.created_at == log.last_update else SocketOperation.UPDATE
                )
                if last_check_time is None:
                    socket_operation = SocketOperation.CREATE
                await websocket.send_json(
                    SocketData(
                        data=log_data,
                        operation=socket_operation
                    ).model_dump(mode="json")
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
        self,
        task_id: str,
        log_id: str,
        agent_id: str,
        record: LogEvalMetricRecord,
    ):
        task = await self.get_task(task_id)
        url = f"{task.base_http_url}/logs/{log_id}/record/metric/update?agent_id={agent_id}"

        async with self.http_session.post(
            url=url,
            headers={'Content-Type': 'application/json'},
            json=record.model_dump(mode="json", by_alias=True)
        ) as resp:
            if resp.status != 200:
                raise HTTPException(status_code=resp.status, detail=await resp.text())

    async def update_task_log_compare_record(
        self,
        task_id: str,
        log_id: str,
        record: LogEvalCompareRecord
    ):
        task = await self.get_task(task_id)
        url = f"{task.base_http_url}/logs/{log_id}/record/compare/update"

        async with self.http_session.post(
            url=url,
            headers={'Content-Type': 'application/json'},
            json=record.model_dump(mode="json", by_alias=True)
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


@task_router.post("/create", response_model=Task, response_model_include={"id"})
async def create_task(
    payload: TaskCreationPayload,
    task_manager: TaskManager = Depends(TaskManager.get_instance)
) -> Task:
    return await task_manager.create_task(payload)


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
    task_id: str,
    background_tasks: BackgroundTasks,
    task_manager: TaskManager = Depends(TaskManager.get_instance)
):
    background_tasks.add_task(task_manager.delete_task, task_id)


@task_router.get("/{task_id}/agents_connected", response_class=JSONResponse)
async def get_task_agents_connected(
    task_id: str,
    task_manager: TaskManager = Depends(TaskManager.get_instance)
) -> JSONResponse:
    return JSONResponse(content=await task_manager.get_task_agents_connected(task_id))


@task_router.get("/{task_id}/results_dir", response_class=JSONResponse)
async def get_task_results_dir(task: Task = Depends(TaskManager.http_get_task_by_id)) -> JSONResponse:
    return JSONResponse(content={"results_dir": task.results_dir})


@task_router.get("/{task_id}/payload", response_model=TaskCreationPayload)
async def get_task_payload(task: Task = Depends(TaskManager.http_get_task_by_id)) -> TaskCreationPayload:
    return TaskCreationPayload(**json.loads(task.payload))


@task_router.get("/{task_id}/status", response_class=JSONResponse)
async def get_task_status(task: Task = Depends(TaskManager.http_get_task_by_id)) -> JSONResponse:
    return JSONResponse(content={"status": task.status})


@task_router.patch("/{task_id}/status")
async def update_task_status(
    task_id: str,
    task_status: str,
    secret_key: str,
    task_manager: TaskManager = Depends(TaskManager.get_instance)
):
    await task_manager.update_task_status(task_id=task_id, task_status=task_status, secret_key=secret_key)


@task_router.get("/history", response_class=JSONResponse)
async def get_history_tasks(task_manager: TaskManager = Depends(TaskManager.get_instance)):
    tasks = await task_manager.get_history_tasks()
    return JSONResponse(
        content=[
            task.model_dump(mode="json", include={"id", "project_id", "status", "created_at", "results_dir"})
            for task in tasks
        ]
    )


@task_router.post("/{task_id}/logs/insert")
async def insert_log(
    task_id: str,
    secret_key: str,
    log: Log,
    task_manager: TaskManager = Depends(TaskManager.get_instance)
):
    await task_manager.insert_task_log(task_id, secret_key, log)


@task_router.patch("/{task_id}/logs/update")
async def update_log(
    task_id: str,
    secret_key: str,
    log: Log,
    task_manager: TaskManager = Depends(TaskManager.get_instance)
):
    await task_manager.update_task_log(task_id, secret_key, log)


@task_router.websocket("/{task_id}/logs/ws")
async def stream_logs(
    task_id: str,
    websocket: WebSocket,
    task_manager: TaskManager = Depends(TaskManager.get_instance),
):
    await websocket.accept()
    await task_manager.websocket_connection(task_id, websocket)


@task_router.websocket("/{task_id}/human/{agent_id}/ws")
async def human_connection(
    task_id: str,
    agent_id: str,
    websocket: WebSocket,
    task_manager: TaskManager = Depends(TaskManager.get_instance),
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
    task_manager: TaskManager = Depends(TaskManager.get_instance)
):
    await task_manager.update_task_log_metric_record(task_id, log_id, agent_id, record)


@task_router.post("/{task_id}/logs/{log_id}/record/compare/update")
async def update_compare_record(
    task_id: str,
    log_id: str,
    record: LogEvalCompareRecord,
    task_manager: TaskManager = Depends(TaskManager.get_instance)
):
    await task_manager.update_task_log_compare_record(task_id, log_id, record)


@task_router.post("/{task_id}/messages/insert")
async def insert_message(
    task_id: str,
    secret_key: str,
    message: Message,
    task_manager: TaskManager = Depends(TaskManager.get_instance)
):
    await task_manager.insert_task_message(task_id, secret_key, message)


__all__ = ["task_router", "TaskCreationPayload", "TaskManager", "LogEvalMetricRecord", "LogEvalCompareRecord"]
