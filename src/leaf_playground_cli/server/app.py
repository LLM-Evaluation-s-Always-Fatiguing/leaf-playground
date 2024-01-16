import os.path
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, DirectoryPath

from leaf_playground import __version__ as leaf_version

from .hub import *
from .task import *
from .task.model import TaskRunTimeEnv


class AppConfig(BaseModel):
    hub_dir: DirectoryPath = Field(default=...)
    server_port: int = Field(default=...)
    server_host: str = Field(default=...)
    runtime_env: TaskRunTimeEnv = Field(default=...)


class AppInfo(BaseModel):
    name: Literal["leaf-playground-local-server"] = Field(default="leaf-playground-local-server")
    version: str = Field(default=leaf_version)
    hub_dir: str = Field(default=...)


app_config: AppConfig = None
app_info: AppInfo = None


def config_server(config: AppConfig):
    global app_config, app_info
    app_config = config
    app_info = AppInfo(hub_dir=app_config.hub_dir.as_posix())


@asynccontextmanager
async def lifespan(app: FastAPI):
    DB(f"sqlite+aiosqlite:///{os.path.join(app_config.hub_dir, '.leaf_workspace', '.leaf_playground.db')}")
    Hub(hub_dir=app_config.hub_dir.as_posix())
    TaskManager(server_port=app_config.server_port, runtime_env=app_config.runtime_env)

    try:
        yield
    except:
        pass


app = FastAPI(title="leaf-playground-local-server", version=leaf_version, lifespan=lifespan)
app.include_router(hub_router)
app.include_router(task_router)


@app.get("/", response_class=JSONResponse)
async def main_page(
    hub: Hub = Depends(Hub.get_instance)
) -> JSONResponse:
    """response is a json dict with two fields: projects(simple information for projects),
    app_info(metadata of the server)"""
    projects = []
    for proj in hub.projects.values():
        projects.append(
            {
                "name": proj.name,
                "id": proj.id,
                "description": proj.metadata.scene_metadata["scene_definition"]["description"]
            }
        )
    return JSONResponse(
        content={
            "projects": list(hub.projects.keys()),
            "app_info": app_info.model_dump(mode="json")
        }
    )


__all__ = [
    "config_server", "AppConfig"
]
