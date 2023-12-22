import json
import os
import random
import signal
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, DirectoryPath, PrivateAttr

from leaf_playground.core.scene_engine import SceneEngineState, SceneObjConfig, MetricEvaluatorObjsConfig

app = FastAPI()
service_config: "ServiceConfig" = None


class ServiceConfig(BaseModel):
    zoo_dir: DirectoryPath = Field(default=...)
    port: int = Field(default=...)


class SceneFull(BaseModel):
    scene_metadata: dict = Field(default=...)
    agents_metadata: Dict[str, List[dict]] = Field(default=...)
    evaluators_metadata: Optional[List[dict]] = Field(default=...)
    work_dir: DirectoryPath = Field(default=...)


class AppPaths(BaseModel):
    zoo_dir: DirectoryPath = Field(default=...)
    result_dir: DirectoryPath = Field(default=...)


class AppInfo(BaseModel):
    paths: AppPaths = Field(default=...)


def scan_scenes(zoo_dir: DirectoryPath) -> List[SceneFull]:
    scenes = []

    if os.path.exists(os.path.join(zoo_dir.as_posix(), ".leaf")):
        work_dir = zoo_dir
        with open(os.path.join(work_dir.as_posix(), ".leaf", "project_config.json"), "r", encoding="utf-8") as f:
            proj_metadata = json.load(f)["metadata"]
        if proj_metadata:
            scenes.append(
                SceneFull(**proj_metadata, work_dir=work_dir)
            )
    else:
        for root, dirs, _ in os.walk(zoo_dir.as_posix()):
            for work_dir in dirs:
                scenes += scan_scenes(Path(str(os.path.join(root, work_dir))))

    return scenes


class TaskCreationPayload(BaseModel):
    scene_obj_config: SceneObjConfig = Field(default=...)
    metric_evaluator_objs_config: MetricEvaluatorObjsConfig = Field(default=...)
    work_dir: DirectoryPath = Field(default=...)


class Task(BaseModel):
    id: str = Field(default=...)
    port: int = Field(default=...)
    host: str = Field(default="http://127.0.0.1")
    status: str = Field(default=SceneEngineState.PENDING.value)
    payload_path: str = Field(default=...)

    _pid: int = PrivateAttr(default=None)

    @property
    def pid(self) -> int:
        return self._pid

    @pid.setter
    def pid(self, pid: int):
        self._pid = pid


class TaskManager:
    def __init__(self):
        self._task_ports = set(list(range(1000, 9999))) - {service_config.port}
        self._port_lock = Lock()

        self.result_dir = os.path.join(service_config.zoo_dir, ".leaf_workspace", "results")
        self.tmp_dir = os.path.join(service_config.zoo_dir, ".leaf_workspace", "tmp")
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.tmp_dir, exist_ok=True)

        self._tasks: Dict[str, Task] = {}

    def acquire_port(self) -> int:
        with self._port_lock:
            port = random.choice(list(self._task_ports))
            self._task_ports -= {port}
        return port

    def release_port(self, port: int) -> None:
        with self._port_lock:
            self._task_ports.add(port)

    def create_task(self, payload: TaskCreationPayload):
        task_id = "task_" + datetime.utcnow().strftime("%Y%m%d%H%M%S") + "_" + uuid4().hex[:8]
        payload_tmp_path = os.path.join(self.tmp_dir, f"task_payload_{task_id}.json")
        port = self.acquire_port()

        task = Task(id=task_id, port=port, payload_path=payload_tmp_path)
        self._tasks[task_id] = task

        with open(payload_tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload.model_dump(mode="json"), f, indent=4, ensure_ascii=False)
        process = subprocess.Popen(
            (
                f"{sys.executable} {os.path.join(payload.work_dir.as_posix(), '.leaf', 'app.py')} "
                f"--payload {payload_tmp_path} "
                f"--port {port} "
                f"--save_dir {self.result_dir} "
                f"--callback http://127.0.0.1:{service_config.port}/task/status/update "
                f"--id {task_id}"
            ).split()
        )
        task.pid = process.pid

        return task

    def update_task_status(self, task_id: str, task_status: str):
        self._tasks[task_id].status = task_status

    def get_task_states(self, task_id: str):
        return self._tasks[task_id].status

    def destroy_task(self, task_id: str):
        pid = self._tasks[task_id].pid

        is_windows = os.name == "nt"
        if is_windows:
            import ctypes
            ctypes.windll.kernel32.GenerateConsoleCtrlEvent(0, pid)  # TODO: truly kill
        else:
            os.kill(pid, signal.SIGINT)


task_manager: TaskManager = None


@app.get("/", response_class=JSONResponse)
async def list_scenes() -> JSONResponse:
    return JSONResponse(
        content=[scene_full.model_dump(mode="json", by_alias=True) for scene_full in
                 scan_scenes(service_config.zoo_dir)],
        media_type="application/json"
    )


@app.get("/info", response_model=AppInfo)
async def get_app_info() -> AppInfo:
    return AppInfo(paths=AppPaths(zoo_dir=service_config.zoo_dir, result_dir=task_manager.result_dir))


@app.get("/task/status/{task_id}")
async def get_task_status(task_id: str):
    return JSONResponse(content={"status": task_manager.get_task_states(task_id)})


class TaskStatusUpdatePayload(BaseModel):
    id: str = Field(default=...)
    status: str = Field(default=...)


@app.post("/task/status/update")
async def update_task_status(payload: TaskStatusUpdatePayload):
    task_manager.update_task_status(task_id=payload.id, task_status=payload.status)


@app.post("/task/create", response_model=Task)
async def create_task(task_creation_payload: TaskCreationPayload) -> Task:
    task = task_manager.create_task(task_creation_payload)
    return task


def start_service(config: ServiceConfig):
    global service_config, task_manager

    service_config = config
    task_manager = TaskManager()

    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=config.port)
