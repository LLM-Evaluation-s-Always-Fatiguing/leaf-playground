import asyncio

from fastapi import WebSocket

from .logger import LogHandler
from ...data.log_body import LogBody, ActionLogBody
from ...data.message import MessagePool
from ...data.socket_data import SocketData, SocketOperation


class SocketHandler(LogHandler):
    def __init__(self):
        self._stopped = False

        self._message_pool: MessagePool = MessagePool.get_instance()
        self._socket_cache = []

    def notify_create(self, log_body: LogBody):
        log_dict = log_body.model_dump(mode="json")
        if isinstance(log_body, ActionLogBody):
            log_dict["response"] = self._message_pool.get_message_by_id(log_body.response).model_dump(mode="json")
        self._socket_cache.append(
            SocketData(data=log_dict, operation=SocketOperation.CREATE)
        )

    def notify_update(self, log_body: LogBody):
        log_dict = log_body.model_dump(mode="json")
        if isinstance(log_body, ActionLogBody):
            log_dict["response"] = self._message_pool.get_message_by_id(log_body.response).model_dump(mode="json")
        self._socket_cache.append(
            SocketData(data=log_dict, operation=SocketOperation.UPDATE)
        )

    async def stream_sockets(self, websocket: WebSocket) -> bool:
        cur = 0
        closed = False
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
        except:
            closed = True
        return closed

    def stop(self):
        self._stopped = True


__all__ = ["SocketHandler"]
