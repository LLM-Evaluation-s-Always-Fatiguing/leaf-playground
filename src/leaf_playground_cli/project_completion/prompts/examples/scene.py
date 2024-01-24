import asyncio
from typing import List, Optional

from leaf_playground.core.workers import Logger
from leaf_playground.core.scene import Scene
from leaf_playground.core.scene_definition import SceneConfig
from leaf_playground.data.log_body import ActionLogBody
from leaf_playground.data.media import Text
from pydantic import Field

from .agents.examiner import Examiner
from .agents.base_examinee import AIBaseExaminee
from .dataset_util import *
from .scene_definition import *


class MmluSceneLogBody(ActionLogBody):
    ground_truth: Optional[Text] = Field(default=None)


MmluSceneConfig = SceneConfig.create_config_model(
    SCENE_DEFINITION,
    additional_config_fields={
        "dataset_config": (DatasetConfig, Field(default=...))
    }
)


class MmluScene(
    Scene,
    scene_definition=SCENE_DEFINITION,
    log_body_class=MmluSceneLogBody
):
    config_cls = MmluSceneConfig
    config: config_cls

    def __init__(self, config: config_cls):
        super().__init__(config=config)

        self.examiner: Examiner = self.static_agents["examiner"][0]
        self.examinees: List[AIBaseExaminee] = self.agents["examinee"]

    async def _run(self):
        async def examinee_answer(examinee: AIBaseExaminee, s: ExaminerSample) -> None:
            try:
                answer: ExamineeAnswer = await examinee.answer(sample=s, examiner=self.examiner.profile)
            except:
                if self.config.debug_mode:
                    raise
                answer: ExamineeAnswer = ExamineeAnswer(
                    sender=examinee.profile,
                    receivers=[self.examiner.profile],
                    content=Text(text=""),
                    sample_id=s.sample_id
                )
            self.message_pool.put_message(answer)
            ground_truth = self.examiner.get_golden_answer(s.sample_id)
            log = self.log_body_class(
                references=[s.id],
                response=answer.id,
                ground_truth=Text(text=ground_truth) if ground_truth else None,
                log_msg=f"examinee [{examinee.name}] answers to sample [{s.sample_id}]",
                action_belonged_chain=examinee.role_definition.get_action_definition("answer").belonged_chain
            )
            self.logger.add_log(log)
            self.notify_evaluators_record(log)

        await self.examiner.prepare_samples(self.config.dataset_config)
        while not self.examiner.check_examine_finish():
            sample: ExaminerSample = await self.examiner.send_sample(
                receivers=[examinee.profile for examinee in self.examinees]
            )
            self.message_pool.put_message(sample)
            self.logger.add_log(
                self.log_body_class(
                    references=None,
                    response=sample.id,
                    ground_truth=None,
                    log_msg=f"examiner sends sample [{sample.sample_id}] to all examinees",
                    action_belonged_chain=self.examiner.role_definition.get_action_definition(
                        "send_sample"
                    ).belonged_chain
                )
            )

            await asyncio.gather(
                *[examinee_answer(examinee, sample) for examinee in self.examinees]
            )


__all__ = [
    "MmluSceneConfig",
    "MmluScene"
]
