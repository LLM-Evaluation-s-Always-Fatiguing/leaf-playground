from abc import abstractmethod, ABC

from leaf_playground.core.scene_agent import SceneAIAgentConfig, SceneAIAgent
from leaf_playground.data.profile import Profile

from ..scene_definition import ExamineeAnswer, ExaminerQuestion, SCENE_DEFINITION


ROLE_DEFINITION = SCENE_DEFINITION.get_role_definition("examinee")


class AIBaseExamineeConfig(SceneAIAgentConfig):
    pass


class AIBaseExaminee(SceneAIAgent, ABC, role_definition=ROLE_DEFINITION):
    config_cls = AIBaseExamineeConfig
    config: config_cls

    @abstractmethod
    async def answer_question(self, question: ExaminerQuestion, examiner: Profile) -> ExamineeAnswer:
        pass


__all__ = [
    "AIBaseExamineeConfig",
    "AIBaseExaminee"
]
