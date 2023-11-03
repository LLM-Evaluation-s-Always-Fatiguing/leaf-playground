import asyncio
import json
from typing import List, Any

from .agent import Examiner, ExaminerConfig, Examinee
from .scene import RoleName, DSQA
from ...data.log_body import LogBody
from ...engine.base import Engine, EngineConfig
from ...utils.import_util import dynamically_import_obj


class DSQAEngineConfig(EngineConfig):

    def model_post_init(self, __context: Any) -> None:
        for agent_obj in self.agents_obj:
            obj = dynamically_import_obj(agent_obj.obj)
            if not obj is Examinee and not obj is Examiner:
                raise TypeError(
                    f"required {Examinee.__name__} or {Examiner.__name__} type agent, but get {obj.__name__}"
                )
        obj = dynamically_import_obj(self.scene_obj.obj)
        if not obj is DSQA:
            raise ValueError(f"required {DSQA.__name__} type scene, but get {obj.__name__}")


class DSQAEngine(Engine):
    config_obj = DSQAEngineConfig

    def __init__(self, config: config_obj):
        super().__init__(config=config)
        self.examiner: Examiner = [agent for agent in self.participants if isinstance(agent, Examiner)][0]
        self.examinees: List[Examinee] = [agent for agent in self.participants if isinstance(agent, Examinee)]
        self.examine_results = []

    def _assign_roles(self, agents: List[Examinee]) -> List[Examinee]:
        for agent in agents:
            if isinstance(agent, Examinee):
                agent.profile.role = self.scene.get_role(RoleName.EXAMINEE.value)
            else:
                agent.profile.role = self.scene.get_role(RoleName.EXAMINER.value)
        return agents

    async def _run(self):
        async def questioning_phase(data: dict, examiner: Examiner):
            question = data[self.scene.config.dataset_config.question_column]

            self._logs.append(
                LogBody(
                    event={
                        "content": f"{examiner.name}({examiner.role_name}) sending a question..."
                    }
                )
            )
            message = examiner.act(question)
            self._message_pool.put_message(message)
            self._logs.append(
                LogBody(
                    message=message.model_dump(
                        include={"content", "sender_name", "sender_role_name", "receiver_role_names"}
                    )
                )
            )

            return question

        async def answering_phase(examinee: Examinee):
            self._logs.append(
                LogBody(
                    event={
                        "content": f"{examinee.name}({examinee.role_name}) answering the question..."
                    }
                )
            )
            histories = self._message_pool.get_messages(examinee)
            message = examinee.act(
                histories,
                receiver_role_names=[RoleName.EXAMINER.value],
                response_prefix=examinee.config.answer_prefix
            )
            self._message_pool.put_message(message)
            self._logs.append(
                LogBody(
                    message=message.model_dump(
                        include={"content", "sender_name", "sender_role_name", "receiver_role_names"}
                    )
                )
            )

            return {"name": examinee.name, "id": str(examinee.id), "answer": message.content}

        self._logs.append(LogBody(event={"content": "Examine Start"}))
        while not self.scene.is_terminal():
            data = self.scene.get_data()
            label = None
            if self.scene.config.dataset_config.golden_answer_column:
                label = data[self.scene.config.dataset_config.golden_answer_column]
            question = await questioning_phase(data, self.examiner)
            answers = await asyncio.gather(*[answering_phase(student) for student in self.examinees])
            self.examine_results.append({"question": question, "answers": answers, "label": label})
        self._logs.append(LogBody(event={"content": "Examine End"}))

    def export_examine_results(self, file: str):
        with open(file, "w", encoding="utf-8") as f:
            for exam_res in self.examine_results:
                f.write(json.dumps(exam_res, ensure_ascii=False) + "\n")
