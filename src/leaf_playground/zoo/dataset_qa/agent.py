from typing import List, Set
from uuid import uuid4

from pydantic import Field

from ...agent import Agent, AgentConfig, TextCompletionAgent, TextCompletionAgentConfig
from ...data.message import TextMessage
from ...utils.import_util import DynamicFn, DynamicObject


class ExaminerConfig(AgentConfig):
    profile_data: dict = Field(default={"id": uuid4(), "name": "Jane"})


class Examiner(Agent):
    config_obj = ExaminerConfig

    def act(self, question: str) -> TextMessage:

        return TextMessage(
            sender_id=self.id,
            sender_name=self.name,
            sender_role_name=self.role_name,
            content=question
        )

    async def a_act(self, question: str) -> TextMessage:
        return self.act(question)


def default_prompt_constructor(profile: str, histories: List[str], response_prefix: str = "") -> str:
    his_str = "\n".join(histories + [response_prefix.strip()])
    return f"{profile}\n\n{his_str}"


class ExamineeConfig(TextCompletionAgentConfig):
    profile_data: dict = Field(default={"id": uuid4(), "name": "Tom"})
    profile_format_template: str = Field(
        default="Your name is {name}, a(n) {role_name}, {role_description}"
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
                obj="default_prompt_constructor",
                module="leaf_playground.zoo.dataset_qa.agent"
            ),
            default_kwargs=None
        ),
        frozen=True
    )
    history_window_size: int = Field(default=1, frozen=True)
    answer_prefix: str = Field(default="")


class Examinee(TextCompletionAgent):
    config_obj = ExamineeConfig
