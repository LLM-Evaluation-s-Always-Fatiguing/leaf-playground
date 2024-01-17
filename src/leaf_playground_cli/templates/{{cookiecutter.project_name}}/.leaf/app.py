import asyncio
import os
import signal
import traceback
from datetime import datetime

import aiohttp
import requests
import sys
from argparse import ArgumentParser
from contextlib import asynccontextmanager

from fastapi import status, FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from leaf_playground._type import Singleton
from leaf_playground.core.workers import Logger, LogHandler
from leaf_playground.core.scene_agent import HumanConnection
from leaf_playground.core.scene_engine import SceneEngine, SceneEngineState
from leaf_playground.data.log_body import LogBody, ActionLogBody
from leaf_playground.data.message import MessagePool
from leaf_playground_cli.server.task import *

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = ArgumentParser()
parser.add_argument("--id", type=str)
parser.add_argument("--port", type=int)
parser.add_argument("--host", type=str)
parser.add_argument("--secret_key", type=str)
parser.add_argument("--server_url", type=str)
parser.add_argument("--docker", action="store_true")
args = parser.parse_args()


class DBLogHandler(Singleton, LogHandler):
    def __init__(self):
        super().__init__()

        self._message_pool = MessagePool()
        self._submitted_messages = set()
        self._http_session = aiohttp.ClientSession(base_url=args.server_url)

        self._queue = asyncio.Queue()

        asyncio.ensure_future(self.db_write_loop())

    async def db_write_loop(self):
        while True:
            log_body: LogBody = await self._queue.get()
            is_update = log_body.created_at != log_body.last_update
            if not is_update:
                if isinstance(log_body, ActionLogBody):
                    message = self._message_pool.get_message_by_id(log_body.response)
                    if message.id not in self._submitted_messages:
                        self._submitted_messages.add(message.id)
                        async with self._http_session.post(
                            f"/task/{args.id}/messages/insert?secret_key={args.secret_key}",
                            headers={'Content-Type': 'application/json'},
                            json=Message.init_from_message(message, args.id).model_dump(mode="json", by_alias=True)
                        ) as resp:
                            if resp.status != 200:
                                print(f"task [{args.id}] insert message [{message.id}] to database failed.")
                                print(await resp.text())
                async with self._http_session.post(
                    f"/task/{args.id}/logs/insert?secret_key={args.secret_key}",
                    headers={'Content-Type': 'application/json'},
                    json=Log.init_from_log_body(log_body, args.id).model_dump(mode="json", by_alias=True)
                ) as resp:
                    if resp.status != 200:
                        print(f"task [{args.id}] insert log [{log_body.id}] to database failed.")
                        print(await resp.text())
            else:
                async with self._http_session.patch(
                    f"/task/{args.id}/logs/insert?update={args.secret_key}",
                    headers={'Content-Type': 'application/json'},
                    json=Log.init_from_log_body(log_body, args.id).model_dump(mode="json", by_alias=True)
                ) as resp:
                    if resp.status != 200:
                        print(f"task [{args.id}] update log [{log_body.id}] to database failed.")
                        print(await resp.text())

    async def notify_create(self, log_body: LogBody):
        self._queue.put_nowait(log_body)

    async def notify_update(self, log_body: LogBody):
        log_body.last_update = datetime.utcnow()
        self._queue.put_nowait(log_body)


def create_engine():
    DBLogHandler()
    try:
        resp = requests.get(f"{args.server_url}/task/{args.id}/payload")
        payload = TaskCreationPayload(**resp.json())

        if args.docker:
            results_dir = "/tmp/result"
        else:
            resp = requests.get(f"{args.server_url}/task/{args.id}/results_dir")
            results_dir = resp.json()["results_dir"]

        scene_engine = SceneEngine(
            scene_config=payload.scene_obj_config,
            evaluators_config=payload.metric_evaluator_objs_config,
            reporter_config=payload.reporter_obj_config,
            results_dir=results_dir,
            state_change_callbacks=[scene_engine_state_change_callback],
            log_handlers=[DBLogHandler.get_instance()]
        )
        asyncio.create_task(scene_engine.run())
    except:
        traceback.print_exc()
        update_task_status(SceneEngineState.FAILED.value)


def update_task_status(task_status: str):
    try:
        requests.patch(
            f"{args.server_url}/task/{args.id}/status?task_status={task_status}&secret_key={args.secret_key}",
        )
    except:
        pass


def scene_engine_state_change_callback():
    try:
        scene_engine = SceneEngine.get_instance()
    except:
        scene_engine = None
    task_status = scene_engine.state.value if scene_engine is not None else SceneEngineState.PENDING.value
    update_task_status(task_status)


class AppManager(Singleton):
    def __init__(self):
        self.shutdown_event = asyncio.Event()
        self.shutdown_task = asyncio.create_task(self.maybe_shutdown(), name="shutdown_service")

    async def maybe_shutdown(self):
        await self.shutdown_event.wait()
        await asyncio.sleep(3)
        os.kill(os.getpid(), signal.SIGTERM)


@asynccontextmanager
async def lifespan(application: FastAPI):
    create_engine()
    app_manager = AppManager()

    try:
        yield
    except:
        traceback.print_exc()

    app_manager.shutdown_task.cancel()


app = FastAPI(lifespan=lifespan)


@app.get("/agents_connected")
async def agents_connected(scene_engine: SceneEngine = Depends(SceneEngine.get_instance)) -> JSONResponse:
    return JSONResponse(
        content={
            agent_id: agent.connected for agent_id, agent in scene_engine.scene.dynamic_agents.items()
        }
    )


@app.websocket("/ws/human/{agent_id}")
async def human_input(
    websocket: WebSocket,
    agent_id: str,
    scene_engine: SceneEngine = Depends(SceneEngine.get_instance)
):
    try:
        agent = scene_engine.scene.get_dynamic_agent(agent_id)
    except KeyError:
        await websocket.close(code=403, reason=f"agent [{agent_id}] not exists.")
        return
    if agent.connected:
        await websocket.close(code=403, reason=f"agent [{agent_id}] already connected.")
        return
    connection = HumanConnection(
        agent=agent,
        socket=websocket
    )
    await connection.connect()
    await connection.run()


@app.post("/pause")
async def pause_engine(scene_engine: SceneEngine = Depends(SceneEngine.get_instance)):
    scene_engine.pause()


@app.post("/resume")
async def resume_engine(scene_engine: SceneEngine = Depends(SceneEngine.get_instance)):
    scene_engine.resume()


@app.post("/interrupt")
async def interrupt_engine(
    scene_engine: SceneEngine = Depends(SceneEngine.get_instance),
    app_manager: AppManager = Depends(AppManager.get_instance)
):
    scene_engine.interrupt()
    app_manager.shutdown_event.set()


@app.post("/close")
async def close_engine(
    scene_engine: SceneEngine = Depends(SceneEngine.get_instance),
    app_manager: AppManager = Depends(AppManager.get_instance)
):
    if scene_engine.state not in [SceneEngineState.INTERRUPTED, SceneEngineState.FAILED, SceneEngineState.FINISHED]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="task not done!")
    app_manager.shutdown_event.set()


@app.post("/logs/{log_id}/record/metric/update")
async def update_metric_record(
    log_id: str,
    agent_id: str,
    record: LogEvalMetricRecord,
    logger: Logger = Depends(Logger.get_instance),
    scene_engine: SceneEngine = Depends(SceneEngine.get_instance)
):
    if not logger.is_log_exists(log_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"log [{log_id}] not exist in task [{args.id}]"
        )

    data = {
        "value": record.value,
        "evaluator": "human",
        "target_agent": agent_id,
        "reason": record.reason
    }
    reporter = scene_engine.reporter
    if record.metric_name not in reporter.metric_definitions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"expected metrics are [{list(reporter.metric_definitions.keys())}], got [{record.metric_name}]"
        )
    _, record_model = reporter.metric_definitions[record.metric_name].create_data_models()
    try:
        record_data = record_model(**data)
    except:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="can not parse data properly, this is mainly because the mismatch of value dtype, expected value "
                   f"dtype is {str(record_model.model_fields['value'].annotation)}"
        )
    logger.add_action_log_record(
        log_id=log_id,
        records={
            record.metric_name: data
        },
        field_name="human_eval_records"
    )
    reporter.put_human_record(record=record_data, metric_belonged_chain=record.metric_name, log_id=log_id)


@app.post("/logs/{log_id}/record/compare/update")
async def update_compare_record(
    log_id: str,
    record: LogEvalCompareRecord,
    logger: Logger = Depends(Logger.get_instance),
    scene_engine: SceneEngine = Depends(SceneEngine.get_instance)
):
    if not logger.is_log_exists(log_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"log [{log_id}] not exist in task [{args.id}]"
        )

    data = {
        "value": record.value,
        "reason": record.reason,
        "evaluator": "human"
    }
    reporter = scene_engine.reporter
    if record.metric_name not in reporter.metric_definitions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"expected metrics are [{list(reporter.metric_definitions.keys())}], got [{record.metric_name}]"
        )
    _, record_model = reporter.metric_definitions[record.metric_name].create_data_models()
    try:
        record_data = record_model(**data)
    except:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="can not parse data properly, this is mainly because the mismatch of value dtype, expected value "
                   f"dtype is {str(record_model.model_fields['value'].annotation)}"
        )
    logger.add_action_log_record(
        log_id=log_id,
        records={
            record.metric_name: data
        },
        field_name="human_compare_records"
    )
    reporter.put_human_record(record=record_data, metric_belonged_chain=record.metric_name, log_id=log_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
