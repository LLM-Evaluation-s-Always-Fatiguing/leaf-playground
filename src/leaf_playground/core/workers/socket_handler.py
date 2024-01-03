import asyncio
from typing import Callable, Optional

from fastapi import WebSocket, WebSocketDisconnect

from ...data.log_body import LogBody
from ...data.socket_data import SocketData, SocketOperation
from ...utils.thread_util import run_asynchronously


class SocketHandler:
    def __init__(self):
        self._stopped = False

        self._socket_cache = []

    def notify_create(self, log_body: LogBody):
        self._socket_cache.append(
            SocketData(
                data=log_body.model_dump(mode="json", by_alias=True),
                operation=SocketOperation.CREATE
            )
        )

    def notify_update(self, log_body: LogBody):
        self._socket_cache.append(
            SocketData(
                data=log_body.model_dump(mode="json", by_alias=True),
                operation=SocketOperation.UPDATE
            )
        )

    async def stream_sockets(self, websocket: WebSocket):
        cur = 0
        try:
            while not self._stopped:
                if cur >= len(self._socket_cache):
                    await asyncio.sleep(0.001)
                else:
                    await websocket.send_json(self._socket_cache[cur].model_dump_json())
                    await asyncio.sleep(0.001)
                    cur += 1
            for socket in self._socket_cache[cur:]:
                await websocket.send_json(socket.model_dump_json())
        except WebSocketDisconnect:
            pass

    def stop(self):
        self._stopped = True


__all__ = ["SocketHandler"]
