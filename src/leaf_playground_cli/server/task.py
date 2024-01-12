import asyncio
import json
import os
import random
import signal
import subprocess
import sys
from enum import Enum
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Union
from uuid import uuid4

from fastapi import status, APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, PrivateAttr

from leaf_playground.core.scene_engine import (
    SceneEngineState, SceneObjConfig, MetricEvaluatorObjsConfig, ReporterObjConfig
)

from .hub import Hub
from .utils import get_local_ip
from ._type import SingletonClass


task_router = APIRouter(prefix="/task")


class TaskCreationPayload(BaseModel):
    project_id: str = Field(default=...)
    scene_obj_config: SceneObjConfig = Field(default=...)
    metric_evaluator_objs_config: MetricEvaluatorObjsConfig = Field(default=...)
    reporter_obj_config: ReporterObjConfig = Field(default=...)

    @property
    def workdir(self):
        return Hub.get_instance().get_project(self.project_id).work_dir


class TaskRunTimeEnv(Enum):
    LOCAL = "local"
    DOCKER = "docker"


class Task(BaseModel):
    id: str = Field(default=...)
    port: int = Field(default=...)
    host: str = Field(default="http://127.0.0.1")
    payload: TaskCreationPayload = Field(default=...)
    status: str = Field(default=SceneEngineState.PENDING.value)
    runtime_env: TaskRunTimeEnv = Field(default=TaskRunTimeEnv.LOCAL)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    _secret_key: str = PrivateAttr(default_factory=lambda: uuid4().hex)
    _shutdown_event: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)
    _runtime_id: Union[int, str] = PrivateAttr(default=None)  # pid or docker container name

    def model_post_init(self, __context: Any) -> None:
        asyncio.create_task(self.maybe_shutdown(), name=f"{self.id}_shutdown")

    async def maybe_shutdown(self):
        await self._shutdown_event.wait()
        await asyncio.sleep(3)

        pid = self.runtime_id
        if self.runtime_env == "docker":
            subprocess.run(f"docker stop {pid}".split())

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


class TaskManager(SingletonClass):
    def __init__(
        self,
        hub: Hub,
        server_port: int = 8000,
        server_host: str = "127.0.0.1",
        runtime_env: TaskRunTimeEnv = TaskRunTimeEnv.LOCAL
    ):
        self.hub = hub
        self.result_dir = os.path.join(hub.hub_dir, ".leaf_workspace", "results")
        self.tmp_dir = os.path.join(hub.hub_dir, ".leaf_workspace", "tmp")
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.tmp_dir, exist_ok=True)

        self.server_port = server_port
        self.server_host = server_host
        self._task_ports = set(list(range(10000, 65535))) - {server_port}
        self._port_lock = Lock()

        self.runtime_env = runtime_env

        self._tasks: Dict[str, Task] = {}

    def acquire_port(self) -> int:
        with self._port_lock:
            port = random.choice(list(self._task_ports))
            self._task_ports -= {port}
        return port

    def release_port(self, port: int) -> None:
        with self._port_lock:
            self._task_ports.add(port)

    def create_task(self, payload: TaskCreationPayload) -> Task:
        task = Task(
            id=f"task_{payload.project_id}_" + datetime.utcnow().strftime("%Y%m%d%H%M%S") + "_" + uuid4().hex[:8],
            port=self.acquire_port(),
            host=get_local_ip(),
            payload=payload,
            runtime_env=self.runtime_env
        )
        self._tasks[task.id] = task

        if task.runtime_env == "docker":
            task.runtime_id = self._run_in_docker(task)
        else:
            task.runtime_id = self._run_local(task)

        return task

    def _run_local(self, task: Task):
        work_dir = self.hub.get_project(task.payload.project_id).work_dir
        process = subprocess.Popen(
            (
                f"{sys.executable} {os.path.join(work_dir, '.leaf', 'app.py')} "
                f"--id {task.id} "
                f"--port {task.port} "
                f"--host {task.host} "
                f"--secret_key {task.secret_key} "
                f"--save_dir {self.result_dir} "
                f"--server_url http://{self.server_host}:{self.server_port}"
            ).split()
        )
        return process.pid

    def _run_in_docker(self, task: Task):
        container_name = f"leaf-scene-{task.id}"
        image_name = f"leaf-scene-{self.hub.get_project(task.payload.project_id).name}"
        work_dir = "/app"

        image_check = subprocess.run(f"docker images -q {image_name}", shell=True, capture_output=True, text=True)
        if not image_check.stdout.strip():
            print("Image not found, building Docker image...")
            subprocess.run(f"cd {work_dir} && docker build . -t {image_name}", shell=True)

        subprocess.Popen(
            (
                f"docker run --rm "
                f"-p {task.port}:{task.port} "
                f"-v {self.result_dir}:/tmp/result "
                f"--name {container_name} "
                f"{image_name} .leaf/app.py "
                f"--payload /tmp/payload.json "
                f"--id {task.id} "
                f"--port {task.port} "
                f"--host 0.0.0.0 "
                f"--secret_key {task.secret_key} "
                f"--save_dir {self.result_dir} "
                f"--server_url http://{self.server_host}:{self.server_port}"
            ).split()
        )
        return container_name

    def get_task(self, task_id: str) -> Task:
        try:
            task = self._tasks[task_id]
        except KeyError:
            raise KeyError(f"task [{task_id}] not exists.")
        return task

    @classmethod
    def http_get_task_by_id(cls, task_id: str) -> Task:
        try:
            manager = cls.get_instance()
        except KeyError as e:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="service not ready.")

        try:
            task = manager.get_task(task_id)
        except KeyError as e:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

        return task


@task_router.post("/create", response_class=JSONResponse)
async def create_task(
    payload: TaskCreationPayload,
    task_manager: TaskManager = Depends(TaskManager.get_instance)
) -> JSONResponse:
    task_manager.hub.http_get_project_by_id(payload.project_id)
    task = task_manager.create_task(payload)
    return JSONResponse(content=task.model_dump(mode="json", exclude={"payload", "status", "runtime_env"}))


@task_router.get("/{task_id}/payload", response_model=TaskCreationPayload)
async def get_task_payload(task: Task = Depends(TaskManager.http_get_task_by_id)) -> TaskCreationPayload:
    return task.payload


@task_router.get("/{task_id}/status", response_class=JSONResponse)
async def get_task_status(task: Task = Depends(TaskManager.http_get_task_by_id)) -> JSONResponse:
    return JSONResponse(content={"status": task.status})


@task_router.post("/{task_id}/status")
async def update_task_status(
    task_status: str,
    secret_key: str,
    task: Task = Depends(TaskManager.http_get_task_by_id),
):
    if secret_key != task.secret_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="incorrect secret key, not allowed to update task status."
        )

    if task_status not in [s.value for s in SceneEngineState]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"incorrect task status, should be one of {[s.value for s in SceneEngineState]}"
        )
    task.status = task_status
    if task_status in [
        SceneEngineState.RESULT_SAVED.value,
        SceneEngineState.FAILED.value,
        SceneEngineState.INTERRUPTED.value,
    ]:
        task.shutdown_event.set()


__all__ = ["task_router", "TaskCreationPayload", "TaskRunTimeEnv", "Task", "TaskManager"]
