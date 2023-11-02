import asyncio
from typing import List, Any
from threading import Thread

from .agent import Student, Teacher, TeacherConfig
from .dataset import format_data
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
        async def stu_act(stu, message, prefix):
            self._logs.append(
                LogBody(
                    event={
                        "content": f"{stu.name}({stu.role_name}) answering the question..."
                    }
                )
            )
            resp = await stu.a_act([message], response_prefix=prefix)
            self._logs.append(
                LogBody(message=resp.model_dump(include={"content", "sender_name", "sender_role_name"}))
            )

        async def batch_stu_act(message, prefix):
            await asyncio.gather(
                *[
                    stu_act(stu, message, prefix) for stu in self.students
                ]
            )

        self._logs.append(LogBody(event={"content": "Examine Start"}))
        while not self.scene.is_terminal():
            teacher_msg, student_prefix = format_data(self.scene.get_data())
            self._logs.append(LogBody(event={"content": "Session Start"}))
            self._logs.append(
                LogBody(
                    event={
                        "content": f"{self.teacher.name}({self.teacher.role_name}) sending a question..."
                    }
                )
            )
            teacher_message = self.teacher.act(teacher_msg)
            self._logs.append(
                LogBody(
                    message=teacher_message.model_dump(include={"content", "sender_name", "sender_role_name"})
                )
            )
            await batch_stu_act(teacher_message, student_prefix)
            self._logs.append(LogBody(event={"content": "Session End"}))
        self._logs.append(LogBody(event={"content": "Examine End"}))
