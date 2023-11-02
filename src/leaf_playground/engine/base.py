import asyncio
import json
from abc import abstractmethod
from collections import defaultdict
from enum import Enum
from functools import partial
from typing import Any, Callable, List, Optional, Set

from pydantic import Field, FilePath

from .._config import _Config, _Configurable
from ..agent import Agent
from ..data.message import Message, MessagePool
from ..data.log_body import LogBody
from ..scene.base import Scene
from ..utils.import_util import dynamically_import_obj, DynamicObject


class ObjConfig(_Config):
    config_data: Optional[dict] = Field(default=None)
    config_file: Optional[FilePath] = Field(default=None)
    obj: DynamicObject = Field(default=...)

    def model_post_init(self, __context: Any) -> None:
        if not self.config_data and not self.config_file:
            raise ValueError("at least config_data or config_file should be specified")

    @property
    def instance(self):
        if not self.config_data:
            with open(self.config_file, "r", encoding="utf-8") as f:
                self.config_data = json.load(f)
        obj = dynamically_import_obj(self.obj)
        config = obj.config_obj(**self.config_data)
        return obj(config=config)


class EngineConfig(_Config):
    agents_obj: List[ObjConfig] = Field(default=...)
    scene_obj: ObjConfig = Field(default=...)

    @property
    def agents(self) -> List[Agent]:
        return [agent_obj.instance for agent_obj in self.agents_obj]

    @property
    def scene(self) -> Scene:
        return self.scene_obj.instance


class EngineState(Enum):
    PENDING = 0
    RUNNING = 1
    FINISHED = 2


class Engine(_Configurable):
    config_obj = EngineConfig

    def __init__(self, config: config_obj):
        super().__init__(config=config)

        self.scene: Scene = self.config.scene
        self.participants = self._assign_roles(self.config.agents)

        role2participants = defaultdict(list)
        for participant in self.participants:
            role2participants[participant.role_name].append(participant.name)
        for role_name, participant_names in role2participants.items():
            role_num = self.scene.schema.role_schema.get_definition(role_name).role_num
            if len(participant_names) != role_num:
                raise ValueError(f"required {role_num} {role_name}, but get {len(participant_names)}")

        self._logs: List[LogBody] = []
        self._message_pool: MessagePool = MessagePool()
        self._state = EngineState.PENDING

    @property
    def state(self):
        return self._state

    @abstractmethod
    def _assign_roles(self, agents: List[Agent]) -> List[Agent]:
        raise NotImplementedError()

    @abstractmethod
    async def _run(self):
        raise NotImplementedError()

    async def _stream_logs(
        self,
        template: str = "[{time}] :: EVENT - {event} :: MESSAGE - {message}",
        fields: Set[str] = {"time", "event", "message"},
        displayer: Callable[[str], None] = partial(print, flush=True)
    ):
        cur = 0
        while True:
            if cur >= len(self._logs):
                await asyncio.sleep(0.1)
            else:
                displayer(self._logs[cur].format(template, fields))
                cur += 1
                await asyncio.sleep(0.1)
            if self._state == EngineState.FINISHED:
                for log in self._logs[cur:]:
                    displayer(log.format(template, fields))
                break

    def export_logs(self, file: str):
        with open(file, "w", encoding="utf-8") as f:
            for log in self._logs:
                f.write(log.model_dump_json(by_alias=True) + "\n")

    def start(self):
        async def _run_wrapper():
            self._state = EngineState.RUNNING
            await self._run()
            self._state = EngineState.FINISHED

        async def _start():
            await asyncio.gather(_run_wrapper(), self._stream_logs())

        asyncio.get_event_loop().run_until_complete(_start())

    @classmethod
    def from_config(cls, config: config_obj) -> "Engine":
        return cls(config=config)
