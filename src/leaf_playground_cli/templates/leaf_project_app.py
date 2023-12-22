import json
import os
import requests
import sys
from argparse import ArgumentParser
from threading import Thread
from typing import Any, List, Optional
from uuid import UUID

from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
from leaf_playground.core.scene_engine import SceneEngine
from leaf_playground.core.scene_definition.definitions.metric import DisplayType
from leaf_playground_cli.service import TaskCreationPayload
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


parser = ArgumentParser()
parser.add_argument("--payload", type=str)
parser.add_argument("--port", type=int)
parser.add_argument("--save_dir", type=str)
parser.add_argument("--callback", type=str)
parser.add_argument("--id", type=str)
args = parser.parse_args()


app = FastAPI()
scene_engine: SceneEngine = None


def create_engine(payload: TaskCreationPayload):
    global scene_engine

    scene_engine = SceneEngine(
        scene_config=payload.scene_obj_config,
        evaluators_config=payload.metric_evaluator_objs_config,
        state_change_callbacks=[update_scene_engine_status]
    )
    scene_engine.save_dir = os.path.join(args.save_dir, args.id)
    Thread(target=scene_engine.run, daemon=True).start()


def update_scene_engine_status():
    try:
        requests.post(args.callback, json={"id": args.id, "status": scene_engine.state.value})
    except:
        pass


@app.on_event("startup")
async def startup():
    with open(args.payload, "r", encoding="utf-8") as f:
        create_engine(TaskCreationPayload(**json.load(f)))
    os.remove(args.payload)


@app.get("/status", response_class=JSONResponse)
async def get_task_status() -> JSONResponse:
    return JSONResponse(content={"status": scene_engine.state.value})


@app.websocket("/ws")
async def stream_task_info(websocket: WebSocket) -> None:
    await websocket.accept()
    await scene_engine.socket_handler.stream_sockets(websocket)


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
    scene_engine.logger.add_action_log_record(
        log_id=payload.log_id,
        records={
            payload.metric_name: {
                "value": payload.value,
                "display_type": payload.display_type.value,
                "target_agent": payload.target_agent,
                "reason": payload.reason
            }
        },
        field_name="human_eval_records"
    )


class MetricCompareRecordUpdatePayload(BaseModel):
    log_id: UUID = Field(default=...)
    metric_name: str = Field(default=...)
    value: List[str] = Field(default=...)
    reason: Optional[str] = Field(default=None)


@app.post("/record/compare/update")
async def update_metric_compare(payload: MetricCompareRecordUpdatePayload):
    scene_engine.logger.add_action_log_record(
        log_id=payload.log_id,
        records={
            payload.metric_name: {
                "value": payload.value,
                "reason": payload.reason
            }
        },
        field_name="human_compare_records"
    )


# TODO: more apis to pause, resume, interrupt, update log, etc.


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=args.port)
