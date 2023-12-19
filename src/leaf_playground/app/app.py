import os
from threading import Thread
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import status, FastAPI, HTTPException, WebSocket, WebSocketException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .const import *
from ..core.scene import Scene, SceneMetadata
from ..core.scene_agent import SceneAgentMetadata
from ..core.scene_engine import (
    SceneEngine, SceneObjConfig, MetricEvaluatorObjsConfig
)
from ..core.scene_definition import SceneDefinition
from ..core.workers import MetricEvaluatorMetadata
from ..utils.import_util import relevantly_find_subclasses
from ..zoo_new import *  # TODO: remove when everything done


class SceneFull(BaseModel):
    scene_metadata: SceneMetadata = Field(default=...)
    agents_metadata: Dict[str, List[SceneAgentMetadata]] = Field(default=...)
    evaluators_metadata: Optional[List[MetricEvaluatorMetadata]] = Field(default=...)


class AppPaths(BaseModel):
    root: str = Field(default=ROOT)
    zoo_root: str = Field(default=ZOO_ROOT)
    save_root: str = Field(default=SAVE_ROOT)

    def model_post_init(self, __context: Any) -> None:
        os.makedirs(self.save_root, exist_ok=True)


class AppInfo(BaseModel):
    paths: AppPaths = Field(default=AppPaths())


app = FastAPI()


def scan_scenes() -> List[SceneFull]:
    scenes = []
    for scene_class in relevantly_find_subclasses(
        root_path=ZOO_ROOT, prefix="leaf_playground.zoo_new", base_class=Scene
    ):
        scene_class: Scene
        scene_metadata = scene_class.get_metadata()
        agents_metadata = {}
        for role_def in scene_class.scene_definition.roles:
            agents_metadata[role_def.name] = role_def.agents_metadata
        # TODO: scan evaluators

        # temporarily mocked logic
        scenes.append(
            SceneFull(
                scene_metadata=scene_metadata,
                agents_metadata=agents_metadata,
                evaluators_metadata=None
            )
        )

    return scenes


SCENES: List[SceneFull] = scan_scenes()
TASK_CACHE: Dict[UUID, SceneEngine] = {}  # TODO: impl task manager(a thread pool)


# Don't provide this endpoint for now, because current obj cache mechanism can't handle the update of existing obj.
# @app.get("/refresh_zoo")
# async def refresh_zoo():
#     global SCENES
#     SCENES = scan_scenes()


@app.get("/", response_class=JSONResponse)
async def list_scenes() -> JSONResponse:
    return JSONResponse(
        content=[scene_full.model_dump(mode="json", by_alias=True) for scene_full in SCENES],
        media_type="application/json"
    )


@app.get("/info", response_model=AppInfo)
async def get_app_info() -> AppInfo:
    return AppInfo()


class TaskCreationPayload(BaseModel):
    scene_obj_config: SceneObjConfig = Field(default=...)
    metric_evaluator_objs_config: MetricEvaluatorObjsConfig = Field(default=...)


def _create_task(payload: TaskCreationPayload) -> SceneEngine:
    scene_engine = SceneEngine(
        scene_config=payload.scene_obj_config,
        evaluators_config=payload.metric_evaluator_objs_config
    )
    task_id = scene_engine.id
    TASK_CACHE[task_id] = scene_engine
    return scene_engine


class TaskCreationResponse(BaseModel):
    task_id: UUID = Field(default=...)
    save_dir: str = Field(default=...)
    scene_config: dict = Field(default=...)
    evaluator_configs: List[dict] = Field(default=...)


@app.post("/task/create", response_model=TaskCreationResponse)
async def create_scene(task_creation_payload: TaskCreationPayload) -> TaskCreationResponse:
    scene_engine = _create_task(payload=task_creation_payload)
    save_dir = os.path.join(SAVE_ROOT, scene_engine.id.hex)
    scene_engine.save_dir = save_dir

    Thread(target=scene_engine.run, daemon=True).start()  # TODO: optimize, this is ugly

    return TaskCreationResponse(
        task_id=scene_engine.id,
        save_dir=save_dir,
        scene_config=scene_engine.get_scene_config(mode="dict"),
        evaluator_configs=scene_engine.get_evaluator_configs(mode="dict")
    )


@app.get("/task/status/{task_id}", response_class=JSONResponse)
async def get_task_status(task_id: UUID) -> JSONResponse:
    if task_id not in TASK_CACHE:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="task not found")

    scene_engine = TASK_CACHE[task_id]

    return JSONResponse(content={"status": scene_engine.state.value})


@app.websocket("/task/ws/{task_id}")
async def stream_task_info(websocket: WebSocket, task_id: UUID) -> None:
    if task_id not in TASK_CACHE:
        raise WebSocketException(code=status.WS_1011_INTERNAL_ERROR, reason="task not found")

    await websocket.accept()

    scene_engine = TASK_CACHE[task_id]

    await scene_engine.socket_handler.stream_sockets(websocket)


@app.post("/task/save/{task_id}")
async def save_task(task_id: UUID):
    if task_id not in TASK_CACHE:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="task not found")

    scene_engine = TASK_CACHE[task_id]

    scene_engine.save()


# TODO: more apis to pause, resume, interrupt, update log, etc.


@app.post("/test/create_task/", response_class=JSONResponse)
async def test_create_task(task_creation_payload: TaskCreationPayload) -> JSONResponse:
    _create_task(payload=task_creation_payload)

    return JSONResponse(content="create task successfully")
