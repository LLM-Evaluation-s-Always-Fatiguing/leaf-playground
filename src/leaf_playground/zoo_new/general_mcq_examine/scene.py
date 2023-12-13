import asyncio
from contextlib import contextmanager
from typing import List, Optional, Union

from pydantic import Field

from leaf_playground.core.scene import Scene
from leaf_playground.core.scene_definition import SceneConfig
from leaf_playground.data.log_body import LogBody
from leaf_playground.data.media import Text
from leaf_playground.data.socket_data import SocketData, SocketDataType

from .agents.examiner import Examiner
from .agents.base_examinee import AIBaseExaminee
from .dataset_utils import DatasetConfig
from .scene_definition import ExamineeAnswer, ExaminerQuestion, MessageType, SCENE_DEFINITION


class GeneralMCQExamineSceneLogBody(LogBody):
    references: Optional[List[MessageType]] = Field(default=None)
    response: MessageType = Field(default=...)
    ground_truth: Optional[Text] = Field(default=None)


GeneralMCQExamineSceneConfig = SceneConfig.create_config_model(
    SCENE_DEFINITION,
    additional_config_fields={"dataset_config": (DatasetConfig, Field(default=...))}
)


class GeneralMCQExamineScene(Scene, scene_definition=SCENE_DEFINITION, log_body_class=GeneralMCQExamineSceneLogBody):
    config_cls = GeneralMCQExamineSceneConfig
    config: config_cls

    def __init__(self, config: config_cls):
        super().__init__(config=config)

        self.examiner: Examiner = self.static_agents["examiner"][0]
        self.examinees: List[AIBaseExaminee] = self.agents["examinee"]

    async def _run(self):
        cur_log_idx = 0

        @contextmanager
        def increase_log_index():
            nonlocal cur_log_idx

            yield
            cur_log_idx += 1

        async def examinee_answer(examinee: AIBaseExaminee, q: ExaminerQuestion) -> None:
            nonlocal cur_log_idx

            try:
                answer: ExamineeAnswer = await examinee.answer_question(question=q, examiner=self.examiner.profile)
            except:
                if self.config.debug_mode:
                    raise
                answer: ExamineeAnswer = ExamineeAnswer(
                    sender=examinee.profile,
                    receivers=[self.examiner.profile],
                    content=Text(text=""),
                    question_id=q.question_id
                )
            self.message_pool.put_message(answer)
            ground_truth = self.examiner.get_golden_answer(q.question_id)
            log = self.log_body_class(
                index=cur_log_idx,
                references=[q],
                response=answer,
                ground_truth=Text(text=ground_truth) if ground_truth else None,
                narrator=f"examinee [{examinee.name}] answered question [{q.question_id}]"
            )
            with increase_log_index():
                self.socket_cache.append(
                    SocketData(
                        type=SocketDataType.LOG,
                        data=log.model_dump(mode="json")
                    )
                )
            self.notify_evaluators_record(log)

        self.examiner.prepare_questions(self.config.dataset_config)
        while not self.examiner.check_examine_finish():
            question: ExaminerQuestion = self.examiner.send_question(
                receivers=[examinee.profile for examinee in self.examinees]
            )
            self.message_pool.put_message(question)
            with increase_log_index():
                self.socket_cache.append(
                    SocketData(
                        type=SocketDataType.LOG,
                        data=self.log_body_class(
                            index=cur_log_idx,
                            references=None,
                            response=question,
                            ground_truth=None,
                            narrator=f"examiner sent question [{question.question_id}] to all examinees"
                        ).model_dump(mode="json")
                    )
                )

            await asyncio.gather(
                *[examinee_answer(examinee, question) for examinee in self.examinees]
            )


__all__ = [
    "GeneralMCQExamineSceneConfig",
    "GeneralMCQExamineScene"
]
