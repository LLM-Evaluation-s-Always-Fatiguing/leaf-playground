import asyncio
import json
import os
import random
import signal
import socket
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Literal, Union, Any
from uuid import uuid4

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, DirectoryPath, PrivateAttr

from leaf_playground.core.scene_engine import (
    SceneEngineState,
    SceneObjConfig,
    MetricEvaluatorObjsConfig,
    ReporterObjConfig,
)

app = FastAPI()
service_config: "ServiceConfig" = None


class ServiceConfig(BaseModel):
    zoo_dir: DirectoryPath = Field(default=...)
    port: int = Field(default=...)
    runtime_env: Literal["docker", "local"] = Field(default=...)


class SceneFull(BaseModel):
    scene_metadata: dict = Field(default=...)
    agents_metadata: Dict[str, List[dict]] = Field(default=...)
    evaluators_metadata: Optional[List[dict]] = Field(default=...)
    charts_metadata: Optional[List[dict]] = Field(default=...)
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
            scenes.append(SceneFull(**proj_metadata, work_dir=work_dir))
    else:
        for root, dirs, _ in os.walk(zoo_dir.as_posix()):
            for work_dir in dirs:
                scenes += scan_scenes(Path(str(os.path.join(root, work_dir))))

    return scenes


def get_local_ip() -> str:
    try:
        # Create a temporary socket
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # Connect to a public DNS server (Google's)
            s.connect(("8.8.8.8", 80))
            # Get the local IP address from the socket
            local_ip = s.getsockname()[0]
            return local_ip
    except Exception:
        return "127.0.0.1"


class TaskCreationPayload(BaseModel):
    scene_obj_config: SceneObjConfig = Field(default=...)
    metric_evaluator_objs_config: MetricEvaluatorObjsConfig = Field(default=...)
    reporter_obj_config: ReporterObjConfig = Field(default=...)
    work_dir: DirectoryPath = Field(default=...)


class Task(BaseModel):
    id: str = Field(default=...)
    port: int = Field(default=...)
    host: str = Field(default="http://127.0.0.1")
    status: str = Field(default=SceneEngineState.PENDING.value)
    runtime_env: Literal["docker", "local"] = Field(default="local")
    payload_path: str = Field(default=...)

    _shutdown_event = PrivateAttr(default=asyncio.Event())

    _runtime_id: Union[int, str] = PrivateAttr(default=None)  # pid or docker container name

    def model_post_init(self, __context: Any) -> None:
        asyncio.create_task(self.maybe_shutdown(), name=f"{self.id}_shutdown")

    async def maybe_shutdown(self):
        await self._shutdown_event.wait()
        await asyncio.sleep(3)

        pid = self.runtime_id
        if os.path.exists(self.payload_path):
            os.remove(self.payload_path)

        if self.runtime_env == "docker":
            subprocess.run(f"docker stop {pid}".split())
        else:
            is_windows = os.name == "nt"
            if is_windows:
                import ctypes

                ctypes.windll.kernel32.GenerateConsoleCtrlEvent(0, pid)  # TODO: truly kill
            else:
                os.kill(pid, signal.SIGINT)

    @property
    def shutdown_event(self) -> asyncio.Event:
        return self._shutdown_event

    @property
    def runtime_id(self) -> int:
        return self._runtime_id

    @runtime_id.setter
    def runtime_id(self, pid: int):
        self._runtime_id = pid


class TaskManager:
    def __init__(self):
        self._task_ports = set(list(range(10000, 65535))) - {service_config.port}
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
        scene_name = Path(payload.work_dir).name
        origin_work_dir = payload.work_dir

        port = self.acquire_port()
        host = get_local_ip()
        task = Task(id=task_id, port=port, host=host, payload_path=payload_tmp_path)
        self._tasks[task_id] = task

        if service_config.runtime_env == "docker":
            # TODO: temp workaround
            payload.work_dir = "/app"

        with open(payload_tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload.model_dump(mode="json"), f, indent=4, ensure_ascii=False)

        task.runtime_env = service_config.runtime_env

        if task.runtime_env == "docker":
            task.runtime_id = self._run_in_docker(host, scene_name, origin_work_dir, payload_tmp_path, port, task_id)
        else:
            task.runtime_id = self._run_local(host, payload, payload_tmp_path, port, task_id)

        return task

    def _run_local(self, host, payload, payload_tmp_path, port, task_id):
        process = subprocess.Popen(
            (
                f"{sys.executable} {os.path.join(payload.work_dir.as_posix(), '.leaf', 'app.py')} "
                f"--payload {payload_tmp_path} "
                f"--port {port} "
                f"--host {host} "
                f"--save_dir {self.result_dir} "
                f"--callback http://{host}:{service_config.port}/task/status/update "
                f"--id {task_id}"
            ).split()
        )
        return process.pid

    def _run_in_docker(self, host, scene_name, work_dir, payload_tmp_path, port, task_id):
        container_name = f"leaf-scene-{task_id}"
        image_name = f"leaf-scene-{scene_name}"

        image_check = subprocess.run(f"docker images -q {image_name}", shell=True, capture_output=True, text=True)

        if not image_check.stdout.strip():
            print("Image not found, building Docker image...")
            subprocess.run(f"cd {work_dir} && docker build . -t {image_name}", shell=True)
        subprocess.Popen(
            (
                f"docker run --rm "
                f"-p {port}:{port} "
                f"-v {payload_tmp_path}:/tmp/payload.json "
                f"-v {self.result_dir}:/tmp/result "
                f"--name {container_name} "
                f"{image_name} .leaf/app.py "
                f"--payload /tmp/payload.json "
                f"--port {port} "
                f"--host 0.0.0.0 "
                f"--save_dir /tmp/result "
                f"--callback http://{host}:{service_config.port}/task/status/update "
                f"--id {task_id}"
            ).split()
        )
        return container_name

    def update_task_status(self, task_id: str, task_status: str):
        self._tasks[task_id].status = task_status

    def shutdown_task(self, task_id: str):
        self._tasks[task_id].shutdown_event.set()

    def get_task_states(self, task_id: str):
        return self._tasks[task_id].status


task_manager: TaskManager = None


@app.get("/", response_class=JSONResponse)
async def list_scenes() -> JSONResponse:
    return JSONResponse(
        content=[
            scene_full.model_dump(mode="json", by_alias=True) for scene_full in scan_scenes(service_config.zoo_dir)
        ],
        media_type="application/json",
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
    if payload.status in [
        SceneEngineState.RESULT_SAVED.value,
        SceneEngineState.FAILED.value,
        SceneEngineState.INTERRUPTED.value,
    ]:
        task_manager.shutdown_task(task_id=payload.id)


@app.post("/task/create", response_model=Task)
async def create_task(task_creation_payload: TaskCreationPayload) -> Task:
    task = task_manager.create_task(task_creation_payload)
    return task


def init_service(config: ServiceConfig):
    global service_config, task_manager

    service_config = config
    task_manager = TaskManager()
