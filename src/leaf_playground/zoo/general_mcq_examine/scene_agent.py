from abc import abstractmethod
from collections import defaultdict
from inspect import Signature, Parameter
from typing import Dict, List, Optional
from uuid import UUID

from pydantic import Field

from leaf_playground.core.scene_agent import SceneAIAgent, SceneAIAgentConfig, SceneStaticAgent, SceneStaticAgentConfig
from leaf_playground.data.base import Data
from leaf_playground.data.message import TextMessage, MessagePool
from leaf_playground.data.media import Text
from leaf_playground.data.profile import Profile
from leaf_playground.utils.import_util import DynamicObject
from leaf_playground.zoo.general_mcq_examine.dataset_utils import prepare_dataset, DatasetConfig


class ExaminerQuestion(TextMessage):
    question_id: int = Field(default=...)


class ExaminerJudgeResult(Data):
    examinee: Profile = Field(default=...)
    accuracy: float = Field(default=...)


class ExaminerJudgeResults(Data):
    judge_results: List[ExaminerJudgeResult] = Field(default=...)


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
        "judge_answers": Signature(
            parameters=[
                Parameter(
                    name="message_pool",
                    kind=Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=MessagePool
                )
            ],
            return_annotation=Optional[ExaminerJudgeResults]
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

    def judge_answers(
        self,
        message_pool: MessagePool
    ) -> Optional[ExaminerJudgeResults]:
        golden_answer_column = self._dataset_config.golden_answer_column
        if golden_answer_column is None:
            return

        total_questions = len(self._questions)
        golden_answers = [data[golden_answer_column] for data in self._questions]

        messages = [msg for msg in message_pool.get_messages(self.profile) if isinstance(msg, ExamineeAnswer)]

        id2profile = {}
        id2answers: Dict[UUID, List[ExamineeAnswer]] = defaultdict(list)
        for msg in messages:
            sender = msg.sender
            id2profile[sender.id] = sender
            id2answers[sender.id].append(msg)

        judge_results = ExaminerJudgeResults(judge_results=[])
        for examinee_id, answers in id2answers.items():
            acc_num = 0
            for answer in answers:
                question_id = answer.question_id
                pred = answer.content.text
                golden = golden_answers[question_id]
                if pred.startswith(golden):
                    acc_num += 1
            acc = acc_num / total_questions
            judge_results.judge_results.append(
                ExaminerJudgeResult(
                    examinee=id2profile[examinee_id],
                    accuracy=acc
                )
            )

        return judge_results

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
    "ExaminerJudgeResult",
    "ExaminerJudgeResults",
    "ExamineeAnswer",
    "ExaminerConfig",
    "Examiner",
    "AIBaseExamineeConfig",
    "AIBaseExaminee"
]
