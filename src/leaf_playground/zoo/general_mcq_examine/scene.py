import asyncio
from typing import List, Optional

from pydantic import Field

from leaf_playground.core.scene import Scene, SceneConfig, SceneInfoObjConfig
from leaf_playground.data.log_body import LogBody
from leaf_playground.data.media import MediaType
from leaf_playground.utils.import_util import DynamicObject
from leaf_playground.zoo.general_mcq_examine.dataset_utils import prepare_dataset, DatasetConfig
from leaf_playground.zoo.general_mcq_examine.scene_agent import (
    Examiner,
    AIBaseExaminee,
    ExaminerQuestion,
    ExamineeAnswer,
    ExaminerJudgeResults
)
from leaf_playground.zoo.general_mcq_examine.scene_info import (
    general_mcq_examine_scene_metadata,
)


class GeneralMCQExamineSceneInfoObjConfig(SceneInfoObjConfig):
    scene_info_obj: DynamicObject = Field(
        DynamicObject(
            obj="GeneralMCQExamineSceneInfo",
            module="leaf_playground.zoo.general_mcq_examine.scene_info"
        )
    )


class GeneralMCQExamineSceneConfig(SceneConfig):
    scene_info: GeneralMCQExamineSceneInfoObjConfig = Field(default=...)
    dataset_config: DatasetConfig = Field(default=...)


class GeneralMCQExamineScene(Scene):
    config_obj = GeneralMCQExamineSceneConfig
    config: config_obj

    metadata = general_mcq_examine_scene_metadata
    dynamic_agent_base_classes = [AIBaseExaminee]

    def __init__(self, config: config_obj):
        super().__init__(config=config)

        self.dataset = prepare_dataset(config.dataset_config)
        self.examiner: Examiner = self.static_agents[0]
        self.examinees: List[AIBaseExaminee] = self.agents

        self.judge_results: Optional[ExaminerJudgeResults] = None

    async def _run(self):
        async def examinee_answer(examinee: AIBaseExaminee, q: ExaminerQuestion) -> None:
            answer: ExamineeAnswer = await examinee.answer_question(question=q, examiner=self.examiner.profile)
            self.message_pool.put_message(answer)
            self.logs.append(
                LogBody(
                    index=len(self.logs),  # not thread safe
                    references=[q],
                    response=answer,
                    media_type=MediaType.TEXT,
                    ground_truth=None,
                    eval_result=None,
                    narrator=f"examinee [{examinee.name}] answered question [{q.question_id}]"
                )
            )

        self.examiner.prepare_questions(self.config.dataset_config)
        while not self.examiner.check_examine_finish():
            question: ExaminerQuestion = self.examiner.send_question(
                receivers=[examinee.profile for examinee in self.examinees]
            )
            self.message_pool.put_message(question)
            self.logs.append(
                LogBody(
                    index=len(self.logs),  # not thread safe
                    references=None,
                    response=question,
                    media_type=MediaType.TEXT,
                    ground_truth=None,
                    eval_result=None,
                    narrator=f"examiner sent question [{question.question_id}] to all examinees"
                )
            )

            await asyncio.gather(
                *[examinee_answer(examinee, question) for examinee in self.examinees]
            )

        self.judge_results = self.examiner.judge_answers(self.message_pool)


__all__ = [
    "GeneralMCQExamineSceneInfoObjConfig",
    "GeneralMCQExamineSceneConfig",
    "GeneralMCQExamineScene"
]
