import json
from abc import abstractmethod
from collections import defaultdict
from typing import Dict, List, Optional, Any
from threading import Thread

from pydantic import Field, FilePath

from .._config import _Config, _Configurable
from ..agent import Agent
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

    @abstractmethod
    def _assign_roles(self, agents: List[Agent]) -> List[Agent]:
        raise NotImplementedError()

    @abstractmethod
    def _run(self):
        raise NotImplementedError()

    def start(self):
        self._run()
        Thread(target=self.scene.stream_logs,).start()

    def export_logs(self, file: str):
        with open(file, "w", encoding="utf-8") as f:
            for log in self.scene.logs:
                f.write(log.model_dump_json(by_alias=True) + "\n")

    @classmethod
    def from_config(cls, config: config_obj) -> "Engine":
        return cls(config=config)
