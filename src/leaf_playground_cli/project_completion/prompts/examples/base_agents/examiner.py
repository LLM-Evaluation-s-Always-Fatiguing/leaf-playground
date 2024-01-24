from typing import List, Optional

from leaf_playground.core.scene_agent import SceneStaticAgentConfig, SceneStaticAgent
from leaf_playground.data.profile import Profile
from leaf_playground.data.media import Text

from ..scene_definition import ExaminerSample, SCENE_DEFINITION
from ..dataset_util import *

ROLE_DEFINITION = SCENE_DEFINITION.get_role_definition("examiner")

ExaminerConfig = SceneStaticAgentConfig.create_config_model(ROLE_DEFINITION)


class Examiner(
    SceneStaticAgent,
    role_definition=ROLE_DEFINITION,
    cls_description=ROLE_DEFINITION.description
):
    config_cls = ExaminerConfig
    config: config_cls

    def __init__(self, config: config_cls):
        super().__init__(config=config)

        self._cur = 0
        self._questions = []
        self._ds_config: DatasetConfig = None

    async def prepare_samples(
            self,
            ds_config: DatasetConfig,
    ) -> None:
        self._cur = 0
        self._questions = prepare_samples(ds_config)
        self._ds_config = ds_config

    async def send_sample(self, receivers: List[Profile]) -> ExaminerSample:
        sample = ExaminerSample(
            sender=self.profile,
            receivers=receivers,
            content=Text(text=self._questions[self._cur][QUESTION_COL],
                         display_text=self._questions[self._cur][QUESTION_COL]),
            sample_id=self._cur
        )
        self._cur += 1
        return sample

    def check_examine_finish(self) -> bool:
        return self._cur >= len(self._questions)

    def get_golden_answer(self, sample_id: int) -> Optional[str]:
        return self._questions[sample_id][ANSWER_COL]


__all__ = [
    "ExaminerConfig",
    "Examiner"
]
