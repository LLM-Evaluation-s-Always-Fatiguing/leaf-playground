from abc import abstractmethod
from inspect import Signature, Parameter
from typing import Dict, List

from pydantic import Field

from leaf_playground.core.scene_agent import SceneAIAgent, SceneAIAgentConfig, SceneStaticAgent, SceneStaticAgentConfig
from leaf_playground.data.message import TextMessage
from leaf_playground.data.media import Text
from leaf_playground.data.profile import Profile
from leaf_playground.utils.import_util import DynamicObject
from leaf_playground.zoo.general_mcq_examine.dataset_utils import prepare_dataset, DatasetConfig


class ExaminerQuestion(TextMessage):
    question_id: int = Field(default=...)


class ExamineeAnswer(TextMessage):
    question_id: int = Field(default=...)


class ExaminerConfig(SceneStaticAgentConfig):
    profile: Profile = Field(default=Profile(name="Jane"))


class Examiner(SceneStaticAgent):
    config_obj = ExaminerConfig
    config: config_obj

    _actions: Dict[str, Signature] = {
        "prepare_questions": Signature(
            parameters=[
                Parameter(name="dataset_config", kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=DatasetConfig),
            ],
            return_annotation=None
        ),
        "send_question": Signature(
            parameters=[
                Parameter(name="receivers", kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=List[Profile])
            ],
            return_annotation=ExaminerQuestion
        ),
        "check_examine_finish": Signature(
            parameters=None,
            return_annotation=bool
        )
    }

    description: str = "An agent who minitor the examine"
    obj_for_import: DynamicObject = DynamicObject(
        obj="Examiner",
        module="leaf_playground.zoo.general_examine.scene_agent"
    )

    def __init__(self, config: config_obj):
        super().__init__(config=config)

        self._cur = 0
        self._questions = []
        self._dataset_config: DatasetConfig = None

    def prepare_questions(
        self,
        dataset_config: DatasetConfig,
    ) -> None:
        self._cur = 0
        self._questions = prepare_dataset(dataset_config)
        self._dataset_config = dataset_config

    def send_question(self, receivers: List[Profile]) -> ExaminerQuestion:
        question = ExaminerQuestion(
            sender=self.profile,
            receivers=receivers,
            content=Text(text=self._questions[self._cur][self._dataset_config.question_column]),
            question_id=self._cur
        )
        self._cur += 1
        return question

    def check_examine_finish(self) -> bool:
        return self._cur >= len(self._questions)


class AIBaseExamineeConfig(SceneAIAgentConfig):
    pass


class AIBaseExaminee(SceneAIAgent):
    config_obj = AIBaseExamineeConfig
    config: config_obj

    _actions: Dict[str, Signature] = {
        "answer_question": Signature(
            parameters=[
                Parameter(name="question", kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=ExaminerQuestion),
                Parameter(name="examiner", kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=Profile)
            ],
            return_annotation=ExamineeAnswer
        )
    }

    description: str = (
        "An agent with an AI backend participants in a MCQ examine to answer questions, "
        "this is a base class of this type of agent."
    )
    obj_for_import: DynamicObject = DynamicObject(
        obj="AIBaseExaminee",
        module="leaf_playground.zoo.general_examine.scene_agent"
    )

    @abstractmethod
    async def answer_question(self, question: ExaminerQuestion, examiner: Profile) -> ExamineeAnswer:
        pass


__all__ = [
    "ExaminerQuestion",
    "ExamineeAnswer",
    "ExaminerConfig",
    "Examiner",
    "AIBaseExamineeConfig",
    "AIBaseExaminee"
]
