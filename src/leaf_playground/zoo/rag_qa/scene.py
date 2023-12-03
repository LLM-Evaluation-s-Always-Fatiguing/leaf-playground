import asyncio
from typing import List, Optional, Union, Literal

from pydantic import Field

from leaf_playground.core.scene import Scene, SceneConfig
from leaf_playground.data.log_body import LogBody
from leaf_playground.data.media import MediaType, Text
from leaf_playground.data.socket_data import SocketData, SocketDataType
from leaf_playground.zoo.rag_qa.dataset_utils import DatasetConfig
from leaf_playground.zoo.rag_qa.scene_agent import (
    Examiner,
    AIBaseExaminee,
    ExaminerQuestion,
    ExamineeAnswer
)
from leaf_playground.zoo.rag_qa.scene_evaluator import RagasEvaluator, MetricType
from leaf_playground.zoo.rag_qa.scene_info import (
    rag_qa_scene_metadata,
    RagQaSceneInfo
)


class RagQaSceneLogBody(LogBody):
    references: Optional[List[Union[ExaminerQuestion, ExamineeAnswer]]] = Field(default=None)
    response: Union[ExaminerQuestion, ExamineeAnswer] = Field(default=...)


class RagQaSceneConfig(SceneConfig):
    dataset_config: DatasetConfig = Field(default=...)
    activate_metrics: List[MetricType] = Field(default=["answer_correctness"])


class RagQaScene(Scene):
    config_obj = RagQaSceneConfig
    config: config_obj

    metadata = rag_qa_scene_metadata
    dynamic_agent_base_classes = [AIBaseExaminee]
    evaluator_classes = [RagasEvaluator]
    scene_info_class = RagQaSceneInfo
    log_body_class = RagQaSceneLogBody

    def __init__(self, config: config_obj):
        super().__init__(config=config)

        self.examiner: Examiner = self.static_agents[0]
        self.examinees: List[AIBaseExaminee] = self.agents

    async def _run(self):
        evaluator: Optional[RagasEvaluator] = None if not self.evaluators else self.evaluators[0]

        async def examinee_answer(examinee: AIBaseExaminee, q: ExaminerQuestion) -> None:
            try:
                answer: ExamineeAnswer = await examinee.answer_question(question=q, examiner=self.examiner.profile)
            except:
                if self.config.debug_mode:
                    raise
                answer: ExamineeAnswer = ExamineeAnswer(
                    sender=examinee.profile,
                    receivers=[self.examiner.profile],
                    content=Text(text=""),
                    question_id=q.question_id,
                    contexts=['nothing found']
                )
            self.message_pool.put_message(answer)
            self.socket_cache.append(
                SocketData(
                    type=SocketDataType.LOG,
                    data=self.log_body_class(
                        index=len(self.socket_cache),  # not thread safe
                        references=[q],
                        response=answer,
                        media_type=MediaType.TEXT,
                        ground_truth=None,
                        eval_record=None if not evaluator else (await evaluator.nested_record(answer)).model_dump(
                            mode="json"),
                        narrator=f"examinee [{examinee.name}] answered question [{q.question_id}]"
                    ).model_dump(mode="json")
                )
            )

        self.examiner.prepare_questions(self.config.dataset_config)
        if self.evaluators:
            evaluator.post_init(self.examiner._questions, self.examiner._dataset_config)
        while not self.examiner.check_examine_finish():
            question: ExaminerQuestion = self.examiner.send_question(
                receivers=[examinee.profile for examinee in self.examinees]
            )
            self.message_pool.put_message(question)
            self.socket_cache.append(
                SocketData(
                    type=SocketDataType.LOG,
                    data=self.log_body_class(
                        index=len(self.socket_cache),  # not thread safe
                        references=None,
                        response=question,
                        media_type=MediaType.TEXT,
                        ground_truth=None,
                        eval_record=None,
                        narrator=f"examiner sent question [{question.question_id}] to all examinees"
                    ).model_dump(mode="json")
                )
            )

            await asyncio.gather(
                *[examinee_answer(examinee, question) for examinee in self.examinees]
            )

        if evaluator:
            report = await evaluator.report()
            if report:
                self.socket_cache.append(
                    SocketData(
                        type=SocketDataType.METRIC,
                        data=report.model_dump(mode="json")
                    )
                )


__all__ = [
    "RagQaSceneConfig",
    "RagQaScene"
]
