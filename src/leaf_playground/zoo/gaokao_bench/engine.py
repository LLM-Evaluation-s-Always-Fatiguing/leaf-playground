import asyncio
from typing import List, Any

from .agent import Student, Teacher
from .dataset import format_student_prompt, format_teacher_prompt
from .scene import GaoKaoScene
from ...data.log_body import LogBody
from ...engine.base import Engine, EngineConfig
from ...utils.import_util import dynamically_import_obj


class GaoKaoBenchConfig(EngineConfig):

    def model_post_init(self, __context: Any) -> None:
        for agent_obj in self.agents_obj:
            obj = dynamically_import_obj(agent_obj.obj)
            if not obj is Student and not obj is Teacher:
                raise TypeError(
                    f"required {Student.__name__} or {Teacher.__name__} type agent, but get {obj.__name__}"
                )
        obj = dynamically_import_obj(self.scene_obj.obj)
        if not obj is GaoKaoScene:
            raise ValueError(f"required {GaoKaoScene.__name__} type scene, but get {obj.__name__}")


class GaoKaoBench(Engine):
    config_obj = GaoKaoBenchConfig

    def __init__(self, config: config_obj):
        super().__init__(config=config)
        self.teacher: Teacher = [each for each in self.participants if each.role_name == "考官"][0]
        self.students: List[Student] = [each for each in self.participants if each.role_name == "考生"]

    def _assign_roles(self, agents: List[Student]) -> List[Student]:
        for agent in agents:
            if isinstance(agent, Student):
                agent.profile.role = self.scene.get_role("考生")
            else:
                agent.profile.role = self.scene.get_role("考官")
        return agents

    async def _run(self):
        async def questioning_phase(data: dict, teacher: Teacher):
            prompt = format_teacher_prompt(data)

            self._logs.append(
                LogBody(
                    event={
                        "content": f"{teacher.name}({teacher.role_name}) sending a question..."
                    }
                )
            )
            message = teacher.act(prompt)
            self._message_pool.put_message(message)
            self._logs.append(
                LogBody(
                    message=message.model_dump(
                        include={"content", "sender_name", "sender_role_name", "receiver_role_names"}
                    )
                )
            )

        async def answering_phase(data: dict, student: Student):
            prompt = format_student_prompt(data)

            self._logs.append(
                LogBody(
                    event={
                        "content": f"{student.name}({student.role_name}) answering the question..."
                    }
                )
            )
            histories = self._message_pool.get_messages(student)
            message = student.act(histories, receiver_role_names=["考官"], response_prefix=prompt)
            self._message_pool.put_message(message)
            self._logs.append(
                LogBody(
                    message=message.model_dump(
                        include={"content", "sender_name", "sender_role_name", "receiver_role_names"}
                    )
                )
            )

        self._logs.append(LogBody(event={"content": "Examine Start"}))
        while not self.scene.is_terminal():
            data = self.scene.get_data()
            await questioning_phase(data, self.teacher)
            await asyncio.gather(*[answering_phase(data, student) for student in self.students])
        self._logs.append(LogBody(event={"content": "Examine End"}))
