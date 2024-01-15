from abc import abstractmethod

from fastapi import WebSocket

from .logger import LogHandler
from ...data.log_body import LogBody
from ...data.message import MessagePool


class SocketHandler(LogHandler):
    def __init__(self):
        self._stopped = False
        self._message_pool: MessagePool = MessagePool.get_instance()

    @abstractmethod
    async def notify_create(self, log_body: LogBody):
        pass

    @abstractmethod
    async def notify_update(self, log_body: LogBody):
        pass

    @abstractmethod
    async def stream_sockets(self, websocket: WebSocket) -> bool:
        pass

    def stop(self):
        self._stopped = True


__all__ = ["SocketHandler"]
