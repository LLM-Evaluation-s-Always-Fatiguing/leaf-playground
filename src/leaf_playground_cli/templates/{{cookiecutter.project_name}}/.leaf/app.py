import asyncio
import json
import os
import time

import requests
import sys
from argparse import ArgumentParser
from contextlib import asynccontextmanager
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from fastapi import status, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.websockets import WebSocketState
from leaf_playground.core.scene_agent import HumanConnection
from leaf_playground.core.scene_engine import SceneEngine
from leaf_playground.core.scene_definition.definitions.metric import DisplayType
from leaf_playground_cli.service import TaskCreationPayload
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = ArgumentParser()
parser.add_argument("--payload", type=str)
parser.add_argument("--port", type=int)
parser.add_argument("--host", type=str)
parser.add_argument("--save_dir", type=str)
parser.add_argument("--callback", type=str)
parser.add_argument("--id", type=str)
args = parser.parse_args()


def create_engine(payload: TaskCreationPayload):
    global scene_engine

    scene_engine = SceneEngine(
        scene_config=payload.scene_obj_config,
        evaluators_config=payload.metric_evaluator_objs_config,
        reporter_config=payload.reporter_obj_config,
        state_change_callbacks=[update_scene_engine_status]
    )
    scene_engine.save_dir = os.path.join(args.save_dir, args.id)
    Thread(target=scene_engine.run, daemon=True).start()


def update_scene_engine_status():
    try:
        requests.post(args.callback, json={"id": args.id, "status": scene_engine.state.value})
    except:
        pass


@asynccontextmanager
async def lifespan(application: FastAPI):
    with open(args.payload, "r", encoding="utf-8") as f:
        create_engine(TaskCreationPayload(**json.load(f)))
    os.remove(args.payload)

    global human_conn_manager

    yield


app = FastAPI(lifespan=lifespan)
scene_engine: SceneEngine = None


@app.get("/status", response_class=JSONResponse)
async def get_task_status() -> JSONResponse:
    return JSONResponse(content={"status": scene_engine.state.value})


@app.get("/agents_connected")
async def agents_connected() -> JSONResponse:
    return JSONResponse(
        content={
            agent_id: agent.connected for agent_id, agent in scene_engine.scene.dynamic_agents.items()
        }
    )


@app.websocket("/ws")
async def stream_task_info(websocket: WebSocket) -> None:
    await websocket.accept()
    closed = await scene_engine.socket_handler.stream_sockets(websocket)
    if closed:
        return
    while True:
        try:
            text = await websocket.receive_text()
            if text == "disconnect":
                scene_engine.socket_handler.stop()
        except WebSocketDisconnect:
            return


@app.websocket("/ws/human/{agent_id}")
async def human_input(websocket: WebSocket, agent_id: str):
    try:
        agent = scene_engine.scene.get_dynamic_agent(agent_id)
    except KeyError:
        await websocket.close(reason=f"agent [{agent_id}] not exists.")
        return
    if agent.connected:
        await websocket.close(reason=f"agent [{agent_id}] already connected.")
        return
    connection = HumanConnection(
        agent=agent,
        socket=websocket,
        socket_handler=scene_engine.socket_handler
    )
    await connection.connect()
    await connection.run()


@app.post("/save")
async def save_task():
    scene_engine.save()


class MetricEvalRecordUpdatePayload(BaseModel):
    log_id: UUID = Field(default=...)
    metric_name: str = Field(default=...)
    value: Any = Field(default=...)
    display_type: DisplayType = Field(default=...)
    target_agent: str = Field(default=...)
    reason: Optional[str] = Field(default=None)


@app.post("/record/eval/update")
async def update_metric_record(payload: MetricEvalRecordUpdatePayload):
    data = {
        "value": payload.value,
        "evaluator": "human",
        "target_agent": payload.target_agent,
        "reason": payload.reason
    }
    reporter = scene_engine.reporter
    if payload.metric_name not in reporter.metric_definitions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"expected metrics are [{list(reporter.metric_definitions.keys())}], got [{payload.metric_name}]"
        )
    _, record_model = reporter.metric_definitions[payload.metric_name].create_data_models()
    try:
        record_data = record_model(**data)
    except:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="can not parse data properly, this is mainly because the mismatch of value dtype, expected value "
                   f"dtype is {str(record_model.model_fields['value'].annotation)}"
        )
    data["display_type"] = payload.display_type.value
    scene_engine.logger.add_action_log_record(
        log_id=payload.log_id,
        records={
            payload.metric_name: data
        },
        field_name="human_eval_records"
    )
    reporter.put_human_record(record=record_data, metric_belonged_chain=payload.metric_name, log_id=payload.log_id)


class MetricCompareRecordUpdatePayload(BaseModel):
    log_id: UUID = Field(default=...)
    metric_name: str = Field(default=...)
    value: List[str] = Field(default=...)
    reason: Optional[str] = Field(default=None)


@app.post("/record/compare/update")
async def update_metric_compare(payload: MetricCompareRecordUpdatePayload):
    data = {
        "value": payload.value,
        "reason": payload.reason,
        "evaluator": "human"
    }
    reporter = scene_engine.reporter
    if payload.metric_name not in reporter.metric_definitions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"expected metrics are [{list(reporter.metric_definitions.keys())}], got [{payload.metric_name}]"
        )
    _, record_model = reporter.metric_definitions[payload.metric_name].create_data_models()
    try:
        record_data = record_model(**data)
    except:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="can not parse data properly, this is mainly because the mismatch of value dtype, expected value "
                   f"dtype is {str(record_model.model_fields['value'].annotation)}"
        )
    scene_engine.logger.add_action_log_record(
        log_id=payload.log_id,
        records={
            payload.metric_name: data
        },
        field_name="human_compare_records"
    )
    reporter.put_human_record(record=record_data, metric_belonged_chain=payload.metric_name, log_id=payload.log_id)


# TODO: more apis to pause, resume, interrupt, update log, etc.


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
