import json

from openai import AsyncOpenAI
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import Field

from leaf_playground.ai_backend.openai import OpenAIBackendConfig, CHAT_MODELS
from leaf_playground.data.media import Text, Json
from leaf_playground.data.profile import Profile
from leaf_playground.utils.import_util import DynamicObject
from leaf_playground.zoo.rag_qa.scene_agent import (
    AIBaseExaminee,
    AIBaseExamineeConfig,
    ExamineeAnswer,
    ExaminerQuestion
)


class OpenAIBasicExamineeConfig(AIBaseExamineeConfig):
    ai_backend_config: OpenAIBackendConfig = Field(default=...)
    ai_backend_obj: DynamicObject = Field(
        default=DynamicObject(obj="OpenAIBackend", module="leaf_playground.ai_backend.openai"),
        exclude=True
    )


class OpenAIBasicExaminee(AIBaseExaminee):
    config_obj = OpenAIBasicExamineeConfig
    config: config_obj

    description: str = "Examinee agent using OpenAI API to answer questions"
    obj_for_import: DynamicObject = DynamicObject(
        obj="OpenAIBasicExaminee", module="leaf_playground.zoo.rag_qa.agents.openai_basic_examinee"
    )

    def __init__(self, config: config_obj):
        super().__init__(config=config)

    async def answer_question(self, question: ExaminerQuestion, examiner: Profile) -> ExamineeAnswer:
        client: AsyncOpenAI = self.backend.async_client
        model = self.config.ai_backend_config.model
        is_chat_model = model in CHAT_MODELS

        system_msg = (
            f"You are a meticulous scholar who, when faced with users' questions, not only accurately answers their "
            f"queries but also provides the references used. You always reply to all questions from users in the "
            f"following JSON format:\n\n"
            f"{{\n"
            f"    \"answer\": \"The answer to the original question\",\n"
            f"    \"contexts\": [\"Relevant citations\", \"Typically three to five\", \"Mainly based on objective facts and data\",...]\n"
            f"}}\n\n"
        )
        examiner_msg = question.content.text

        if is_chat_model:
            resp_format = {"type": "json_object"} if model.find("gpt-4") >= 0 else {"type": "text"}
            try:
                resp = await client.chat.completions.create(
                    messages=[
                        ChatCompletionSystemMessageParam(role="system", content=system_msg),
                        ChatCompletionUserMessageParam(role="user", content=examiner_msg),
                    ],
                    model=model,
                    response_format=resp_format,
                    max_tokens=2048
                )
            except Exception as e:
                resp = None

            try:
                obj = json.loads(resp.choices[0].message.content) if resp else {}
            except Exception as e:
                print(f'Response Not JSON: {e}')
                obj = {
                    "answer": resp.choices[0].message.content,
                    "contexts": ['nothing found']  # default for ragas data type validation
                }

            contexts = obj['contexts'] if resp else ['nothing found']
            answer = obj['answer'] if resp else ""

            json_data = {"answer": answer, "contexts": contexts}

            return ExamineeAnswer(
                question_id=question.question_id,
                content=Json(data=json_data, display_text=answer),
                sender=self.profile,
                receivers=[examiner]
            )
        else:
            prompt = (
                f"{system_msg}\n\n"
                f"{examiner.name}({examiner.role.name}): {examiner_msg}\n"
                f"{self.name}({self.role_name}): the answer is "
            )
            contexts = ['nothing found']
            answer = ""

            try:
                resp = await client.completions.create(
                    prompt=prompt,
                    model=model,
                    max_tokens=2048
                )
                answer = resp.choices[0].text if resp else ""

            except Exception as e:
                pass
            return ExamineeAnswer(
                question_id=question.question_id,
                content=Json(data={"answer": answer, "contexts": contexts}, display_text=answer),
                sender=self.profile,
                receivers=[examiner]
            )


__all__ = [
    "OpenAIBasicExamineeConfig",
    "OpenAIBasicExaminee"
]
