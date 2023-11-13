from typing import List, Set
from uuid import uuid4

from pydantic import Field

from ...agent import Agent, AgentConfig, TextCompletionAgent, TextCompletionAgentConfig
from ...data.message import TextMessage
from ...utils.import_util import DynamicFn, DynamicObject


class TeacherConfig(AgentConfig):
    profile_data: dict = Field(default={"id": uuid4(), "name": "Jane"})


class Teacher(Agent):

    def act(self, msg: str) -> TextMessage:

        return TextMessage(
            sender_id=self.id,
            sender_name=self.name,
            sender_role_name=self.role_name,
            content=msg
        )

    async def a_act(self, msg: str) -> TextMessage:
        return self.act(msg)


def student_prompt_constructor(profile: str, histories: List[str], response_prefix: str = "") -> str:
    his_str = "\n".join(histories + [response_prefix.strip()])
    return f"{profile}\n\n{his_str}"


class StudentConfig(TextCompletionAgentConfig):
    profile_data: dict = Field(default={"id": uuid4(), "name": "Tom"})
    profile_format_template: str = Field(
        default="你的名字叫{name}, 你是一名{role_name}, {role_description}"
    )
    profile_format_fields: Set[str] = Field(
        default={"name", "role.name", "role.description"}
    )
    message_format_template: str = Field(
        default="{sender_name}({sender_role_name}): {content}"
    )
    message_format_fields: Set[str] = Field(
        default={"sender_name", "sender_role_name", "content"}
    )
    prompt_constructor_obj: DynamicFn = Field(
        default=DynamicFn(
            fn=DynamicObject(
                obj="student_prompt_constructor",
                module="leaf_playground.zoo.gaokao_bench.agent"
            ),
            default_kwargs=None
        ),
        exclude=True
    )
    history_window_size: int = Field(default=1, exclude=True)


class Student(TextCompletionAgent):
    config_obj = StudentConfig
