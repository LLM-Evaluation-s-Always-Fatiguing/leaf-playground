import random
from abc import abstractmethod
from typing import Dict, List
from uuid import UUID

from .agent import Agent
from .scene.base import Scene, SceneLogBody, SceneSubClsBuildingConfig
from .data.environment import EnvironmentVariable
from .data.profile import Role


class SimulationStep:
    def __init__(self, scene: Scene, agent: Agent):
        self.scene = scene
        self.agent = agent

    @abstractmethod
    def execute(self) -> SceneLogBody:
        raise NotImplementedError()


class Engine:
    def __init__(
        self,
        scene_base_cls: Scene,
        scene_cls_building_config: SceneSubClsBuildingConfig,
        scene_init_kwargs: dict,
        agents: List[Agent]
    ):
        assert len(agents) == len([agent.id for agent in agents]), (
            "id of each agent in a group of agents must unique."
        )

        self._scene: Scene = scene_base_cls.build_sub_cls(scene_cls_building_config)(**scene_init_kwargs)
        self._agents: Dict[UUID, Agent] = {agent.id: agent for agent in agents}
        self._participants: Dict[UUID, str] = self._assign_roles()

    def _assign_roles(self) -> Dict[UUID, str]:
        # TODO: find a better solution, especially a productive way

        num_participants = self._scene.schema.role_schema.num_participants
        assert num_participants == len(self._agents), (
            f"requires {num_participants}, but has {len(self._agents)} agents."
        )

        agents = list(self._agents.keys())
        random.shuffle(agents)
        role2num = {
            role.name: self._scene.schema.role_schema.get_definition(role.name).role_num
            for role in self._scene.roles
        }

        participants = dict()
        for role, num in role2num.items():
            participants.update({agents[i]: role for i in range(num)})
            agents = agents[num:]

        return participants

    def _execute_one_turn(self):
        for step in self._steps:
            log, *_ = step.execute()
            self._scene.append_log(log)

    def run(self):
        while True:
            self._execute_one_turn()

            if self._scene.is_terminal():
                break
