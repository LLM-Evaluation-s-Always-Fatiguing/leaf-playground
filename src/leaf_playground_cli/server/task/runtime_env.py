import asyncio
import json
import os
import subprocess
import sys
import traceback
from abc import abstractmethod
from typing import Union
from uuid import uuid4

from leaf_playground.utils.thread_util import run_asynchronously

from .db import DB
from .model import Task, TaskRunTimeEnv
from ..hub import Hub


class RunTimeEnv:
    def __init__(self, task: Task, manager: "leaf_playground_cli.server.task.api.TaskManager"):
        self.id = task.id
        self.manager = manager

        self._task = task
        self._secret_key: str = uuid4().hex
        self._shutdown_event: asyncio.Event = asyncio.Event()

    async def start(self):
        await DB.get_instance().insert_task(self._task)
        try:
            proc = await run_asynchronously(self._start, self._task)
        except:
            traceback.print_exc()
            await DB.get_instance().update_task_status(self.id, "failed")
            try:
                await run_asynchronously(self._gc, await self.get_task())
            except:
                if self.manager.debugger_config.debug:
                    raise
        else:
            asyncio.ensure_future(self._maybe_shutdown())
            asyncio.ensure_future(self._handle_subproc_failure(proc))

    async def _maybe_shutdown(self):
        await self._shutdown_event.wait()
        await asyncio.sleep(3)

        try:
            await run_asynchronously(self._gc, await self.get_task())
        except:
            if self.manager.debugger_config.debug:
                raise

    async def _handle_subproc_failure(self, proc: subprocess.Popen):
        while proc.poll() is None:
            if self._shutdown_event.is_set():
                break
            await asyncio.sleep(0.001)
        if self._shutdown_event.is_set():
            return
        if proc.returncode != 0:
            await DB.get_instance().update_task_status(self.id, "failed")
            self._shutdown_event.set()

    async def get_task(self) -> Task:
        self._task = await DB.get_instance().get_task(self.id)
        return self._task

    @property
    def secret_key(self) -> str:
        return self._secret_key

    @property
    def shutdown_event(self) -> asyncio.Event:
        return self._shutdown_event

    @abstractmethod
    async def _start(self, task: Task) -> subprocess.Popen:
        pass

    @abstractmethod
    async def _gc(self, task: Task):
        pass


class Local(RunTimeEnv):
    def _start(self, task: Task) -> subprocess.Popen:
        work_dir = Hub.get_instance().get_project(json.loads(task.payload)["project_id"]).work_dir
        cmd = (
            f"{sys.executable} {os.path.join(work_dir, '.leaf', 'app.py')} "
            f"--id {task.id} "
            f"--port {task.port} "
            f"--host {task.host} "
            f"--secret_key {self.secret_key} "
            f"--server_url http://{self.manager.server_host}:{self.manager.server_port} "
            f"{'' if not self.manager.debugger_config.debug else '--debug '}"
            f"--debug_ide {self.manager.debugger_config.ide_type.value} "
            f"--debugger_server_host {self.manager.debugger_config.host} "
            f"--debugger_server_port {self.manager.debugger_config.port} "
            f"--debugger_server_port_evaluator {self.manager.evaluator_debugger_config.port}"
        )
        self.process = subprocess.Popen(cmd.split())
        return self.process

    def _gc(self, task: Task):
        try:
            self.process.kill()
        except:
            traceback.print_exc()
            if self.manager.debugger_config.debug:
                raise


class Docker(RunTimeEnv):
    def _start(self, task: Task) -> subprocess.Popen:
        container_name = f"leaf-scene-{task.id}"
        project = Hub.get_instance().get_project(json.loads(task.payload)['project_id'])
        image_name = f"leaf-scene-{project.name}"

        image_check = subprocess.run(
            f"docker images -q {image_name}", shell=True, capture_output=True, text=True
        )
        if not image_check.stdout.strip():
            print("Image not found, building Docker image...")
            subprocess.run(f"cd {project.work_dir} && docker build . -t {image_name}", shell=True)

        cmd = (
            "docker run --rm "
            f"-p {task.port}:{task.port} "
            f"--name {container_name} "
            f"{image_name} .leaf/app.py "
            f"--id {task.id} "
            f"--port {task.port} "
            "--host 0.0.0.0 "
            f"--secret_key {self.secret_key} "
            f"--server_url http://{self.manager.server_host}:{self.manager.server_port} "
            "--docker"
        )
        proc = subprocess.Popen(cmd.split())
        self.container_name = container_name

        return proc

    def _gc(self, task: Task):
        try:
            subprocess.run(f"docker stop {self.container_name}".split())
        except:
            traceback.print_exc()
            if self.manager.debugger_config.debug:
                raise


RUNTIME_LOOKUP_TABLE = {
    TaskRunTimeEnv.LOCAL: Local,
    TaskRunTimeEnv.DOCKER: Docker
}


__all__ = ["Local", "Docker", "RUNTIME_LOOKUP_TABLE"]
