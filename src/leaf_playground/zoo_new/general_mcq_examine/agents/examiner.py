from typing import List, Optional

from leaf_playground.core.scene_agent import SceneStaticAgentConfig, SceneStaticAgent
from leaf_playground.data.profile import Profile
from leaf_playground.data.media import Text

from ..scene_definition import ExaminerQuestion, SCENE_DEFINITION
from ..dataset_utils import prepare_dataset, DatasetConfig


ROLE_DEFINITION = SCENE_DEFINITION.get_role_definition("examiner")


ExaminerConfig = SceneStaticAgentConfig.create_config_model(ROLE_DEFINITION)


class Examiner(SceneStaticAgent, role_definition=ROLE_DEFINITION, cls_description="An agent who minitor the examine"):
    config_cls = ExaminerConfig
    config: config_cls

    def __init__(self, config: config_cls):
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

    def get_golden_answer(self, question_id: int) -> Optional[str]:
        if self._dataset_config.golden_answer_column:
            return self._questions[question_id][self._dataset_config.golden_answer_column]
        return None


__all__ = [
    "ExaminerConfig",
    "Examiner"
]
