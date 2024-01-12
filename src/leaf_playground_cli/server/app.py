from contextlib import asynccontextmanager
from typing import List, Literal

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, DirectoryPath

from leaf_playground import __version__ as leaf_version
from leaf_playground.core.scene_engine import SceneEngineState

from .hub import *
from .task import *


class AppConfig(BaseModel):
    hub_dir: DirectoryPath = Field(default=...)
    server_port: int = Field(default=...)
    server_host: str = Field(default=...)
    runtime_env: TaskRunTimeEnv = Field(default=...)


class AppInfo(BaseModel):
    name: Literal["leaf-playground-local-server"] = Field(default="leaf-playground-local-server")
    version: str = Field(default=leaf_version)
    hub_dir: str = Field(default=...)


hub: Hub = None
task_manager: TaskManager = None
app_config: AppConfig = None
app_info: AppInfo = None


def config_server(config: AppConfig):
    global app_config, app_info
    app_config = config
    app_info = AppInfo(hub_dir=app_config.hub_dir.as_posix())


@asynccontextmanager
async def lifespan(app: FastAPI):
    global hub, task_manager

    hub = Hub(hub_dir=app_config.hub_dir.as_posix())
    task_manager = TaskManager(hub=hub, server_port=app_config.server_port, runtime_env=app_config.runtime_env)

    try:
        yield
    except:
        pass


app = FastAPI(title="leaf-playground-local-server", version=leaf_version, lifespan=lifespan)
app.include_router(hub_router)
app.include_router(task_router)


@app.get("/", response_class=JSONResponse)
async def main_page() -> JSONResponse:
    """response is a json dict with two fields: projects(a list of project id), app_info(metadata of the server)"""
    return JSONResponse(
        content={
            "projects": list(hub.projects.keys()),
            "app_info": app_info.model_dump(mode="json")
        }
    )


__all__ = [
    "config_server", "AppConfig"
]
