import asyncio
from typing import Any, Dict, List, Optional, Tuple, Type, TypedDict

from fastapi import status, FastAPI, HTTPException, WebSocket, WebSocketException, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import create_model, BaseModel, Field
from pydantic.fields import FieldInfo

from .const import ZOO_ROOT
from .._config import _Config
from ..core.scene import (
    Scene,
    SceneConfig,
    SceneInfoObjConfig,
    SceneAgentObjConfig,
    SceneAgentsObjConfig,
    SceneEvaluatorObjConfig,
    SceneEvaluatorsObjConfig
)
from ..core.scene_agent import SceneAgentConfig, SceneAgent, SceneAgentMetadata
from ..core.scene_evaluator import SceneEvaluatorConfig, SceneEvaluator, SceneEvaluatorMetadata
from ..core.scene_info import SceneMetaData, SceneInfo, SceneInfoConfigBase
from ..utils.import_util import dynamically_import_obj, find_subclasses, DynamicObject


class SceneFull(TypedDict):
    id: str
    scene_metadata: SceneMetaData
    role_agents_num: Dict[str, int]
    max_agents_num: int
    min_agents_num: int
    scene_class: Type[Scene]
    scene_config_class: Type[SceneConfig]
    scene_info_class: Type[SceneInfo]
    scene_info_config_class: Type[SceneInfoConfigBase]
    agent_classes: Dict[str, Type[SceneAgent]]
    agent_config_classes: Dict[str, Type[SceneAgentConfig]]
    agents_metadata: Dict[str, SceneAgentMetadata]
    evaluator_classes: Optional[Dict[str, Type[SceneEvaluator]]]
    evaluator_config_classes: Optional[Dict[str, Type[SceneEvaluatorConfig]]]
    evaluators_metadata: Optional[Dict[str, SceneEvaluatorMetadata]]
    scene_config_additional_fields: Dict[str, Tuple[Type, FieldInfo]]


class SceneDetail(BaseModel):
    id: str = Field(default=...)
    scene_metadata: SceneMetaData = Field(default=...)
    agents_metadata: Dict[str, SceneAgentMetadata] = Field(default=...)
    role_agents_num: Dict[str, int] = Field(default=...)
    max_agents_num: int = Field(default=...)
    min_agents_num: int = Field(default=...)
    scene_info_config_schema: dict = Field(default=...)
    agents_config_schemas: Dict[str, dict] = Field(default=...)
    evaluators_config_schemas: Optional[Dict[str, dict]] = Field(default=...)
    additional_config_schema: dict = Field(default=...)


class SceneBrief(BaseModel):
    id: str = Field(default=...)
    metadata: SceneMetaData = Field(default=...)


class ScenesBrief(BaseModel):
    scenes: List[SceneBrief] = Field(default=...)


class SceneAgentConfigPayload(BaseModel):
    agent_id: str = Field(default=...)
    agent_config_data: dict = Field(default=...)


class SceneEvaluatorConfigPayload(BaseModel):
    evaluator_name: str = Field(default=...)
    evaluator_config_data: dict = Field(default=...)


class SceneCreatePayload(BaseModel):
    id: str = Field(default=...)
    scene_info_config_data: dict = Field(default=...)
    scene_agents_config_data: List[SceneAgentConfigPayload] = Field(default=...)
    scene_evaluators_config_data: Optional[List[SceneEvaluatorConfigPayload]] = Field(default=...)
    additional_config_data: Dict[str, Any] = Field(default=...)


app = FastAPI()


def get_scenes() -> Dict[str, SceneFull]:
    scenes_tmp = [
        (hash(dynamic_obj), dynamically_import_obj(dynamic_obj)) for dynamic_obj
        in find_subclasses(package_path=ZOO_ROOT, base_class=Scene)
    ]
    scenes = {}
    for scene_id, scene_class in scenes_tmp:
        scene_id = str(scene_id)
        scene_class: Type[Scene]

        scene_metadata = scene_class.get_metadata()
        scene_config_class: Type[SceneConfig] = scene_class.config_obj
        scene_info_class: Type[SceneInfo] = scene_class.get_scene_info_class()
        scene_info_config_class: Type[SceneInfoConfigBase] = scene_info_class.config_obj
        agent_classes: Dict[str, Type[SceneAgent]] = {
            str(hash(dynamic_obj)): dynamically_import_obj(dynamic_obj) for dynamic_obj
            in scene_class.get_dynamic_agent_classes()
        }
        agent_config_classes: Dict[str, Type[SceneAgentConfig]] = {
            agent_id: agent_cls.config_obj for agent_id, agent_cls in agent_classes.items()
        }
        agents_metadata: Dict[str, SceneAgentMetadata] = {
            agent_id: agent_cls.get_metadata() for agent_id, agent_cls in agent_classes.items()
        }
        scene_config_additional_fields = {
            field_name: (field_info.annotation, field_info) for field_name, field_info
            in scene_config_class.model_fields.items() if field_name not in SceneConfig.model_fields.keys()
        }

        evaluator_classes = None
        evaluator_config_classes = None
        evaluators_metadata = None
        if scene_class.get_evaluator_classes():
            evaluator_classes = {evaluator.__name__: evaluator for evaluator in scene_class.get_evaluator_classes()}
            evaluator_config_classes = {evaluator.__name__: evaluator.config_obj for evaluator in evaluator_classes.values()}
            evaluators_metadata = {evaluator.__name__: evaluator.get_metadata() for evaluator in evaluator_classes.values()}

        scene_full = SceneFull(
            id=scene_id,
            scene_metadata=scene_metadata,
            role_agents_num=scene_metadata.get_roles_agent_num(),
            max_agents_num=scene_metadata.get_max_dynamic_agents_num(),
            min_agents_num=scene_metadata.get_min_dynamic_agents_num(),
            scene_class=scene_class,
            scene_config_class=scene_config_class,
            scene_info_class=scene_info_class,
            scene_info_config_class=scene_info_config_class,
            agent_classes=agent_classes,
            agent_config_classes=agent_config_classes,
            agents_metadata=agents_metadata,
            evaluator_classes=evaluator_classes,
            evaluator_config_classes=evaluator_config_classes,
            evaluators_metadata=evaluators_metadata,
            scene_config_additional_fields=scene_config_additional_fields
        )

        scenes[scene_id] = scene_full

    return scenes


SCENES: Dict[str, SceneFull] = get_scenes()


# Don't provide this endpoint for now, because current obj cache mechanism can't handle the update of existing obj.
# @app.get("/refresh_zoo")
# async def refresh_zoo():
#     global SCENES
#     SCENES = get_scenes()


@app.get("/", response_model=ScenesBrief)
async def list_scenes() -> ScenesBrief:
    briefs = []
    for scene_id, scene_full in SCENES.items():
        briefs.append(
            SceneBrief(id=scene_id, metadata=scene_full["scene_metadata"])
        )
    return ScenesBrief(scenes=briefs)


@app.get("/scene/{scene_id}", response_model=SceneDetail)
async def get_scene(scene_id: str) -> SceneDetail:
    scene_full = SCENES[scene_id]
    scene_detail = SceneDetail(
        id=scene_full["id"],
        scene_metadata=scene_full["scene_metadata"],
        agents_metadata=scene_full["agents_metadata"],
        role_agents_num=scene_full["role_agents_num"],
        max_agents_num=scene_full["max_agents_num"],
        min_agents_num=scene_full["min_agents_num"],
        scene_info_config_schema=scene_full["scene_info_config_class"].get_json_schema(by_alias=True),
        agents_config_schemas={
            agent_id: agent_config_class.get_json_schema(by_alias=True) for agent_id, agent_config_class in
            scene_full["agent_config_classes"].items()
        },
        evaluators_config_schemas=None if not scene_full["evaluator_config_classes"] else {
            evaluator_name: evaluator_config_class.get_json_schema(by_alias=True) for
            evaluator_name, evaluator_config_class in scene_full["evaluator_config_classes"].items()
        },
        additional_config_schema=create_model(
            __model_name="AdditionalConfigTemp",
            __base__=_Config,
            **scene_full["scene_config_additional_fields"]
        ).get_json_schema(by_alias=True)
    )

    return scene_detail


def _create_scene(payload: SceneCreatePayload) -> Scene:
    scene_full = SCENES[payload.id]
    scene_config_class = scene_full["scene_config_class"]

    scene_info_obj_config = SceneInfoObjConfig(
        scene_info_config_data=payload.scene_info_config_data,
        scene_info_obj=scene_full["scene_info_class"].obj_for_import
    )
    scene_agents_obj_config = SceneAgentsObjConfig(
        agents=[
            SceneAgentObjConfig(
                agent_config_data=agent_config_payload.agent_config_data,
                agent_obj=scene_full["agent_classes"][agent_config_payload.agent_id].obj_for_import
            ) for agent_config_payload in payload.scene_agents_config_data
        ]
    )
    scene_evaluators_obj_config = SceneEvaluatorsObjConfig(
        evaluators=[
            SceneEvaluatorObjConfig(
                evaluator_config_data=evaluator_config_payload.evaluator_config_data,
                evaluator_obj=scene_full["evaluator_classes"][evaluator_config_payload.evaluator_name].obj_for_import
            ) for evaluator_config_payload in payload.scene_evaluators_config_data or []
        ]
    )
    scene_config = scene_config_class(
        scene_info=scene_info_obj_config,
        scene_agents=scene_agents_obj_config,
        scene_evaluators=scene_evaluators_obj_config,
        **payload.additional_config_data
    )
    scene = scene_full["scene_class"].from_config(config=scene_config)
    return scene


@app.websocket("/run_scene")
async def run_scene(websocket: WebSocket) -> None:
    await websocket.accept()

    scene_creation_payload = SceneCreatePayload(**await websocket.receive_json())
    scene = _create_scene(payload=scene_creation_payload)
    await asyncio.gather(scene.a_start(), scene.stream_sockets(websocket))


@app.post("/test/create_scene/", response_class=JSONResponse)
async def test_create_scene(scene_creation_payload: SceneCreatePayload) -> JSONResponse:
    _create_scene(payload=scene_creation_payload)

    return JSONResponse(content="create scene successfully")
