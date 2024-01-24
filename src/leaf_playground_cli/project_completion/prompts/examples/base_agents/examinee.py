from abc import abstractmethod, ABC

from leaf_playground.core.scene_agent import SceneAIAgentConfig, SceneAIAgent
from leaf_playground.data.profile import Profile

from ..scene_definition import ExamineeAnswer, ExaminerSample, SCENE_DEFINITION


ROLE_DEFINITION = SCENE_DEFINITION.get_role_definition("examinee")


class ExamineeConfig(SceneAIAgentConfig):
    pass


class Examinee(SceneAIAgent, ABC, role_definition=ROLE_DEFINITION):
    config_cls = ExamineeConfig
    config: config_cls

    @abstractmethod
    async def answer(self, sample: ExaminerSample, examiner: Profile) -> ExamineeAnswer:
        pass


__all__ = [
    "ExamineeConfig",
    "Examinee"
]
