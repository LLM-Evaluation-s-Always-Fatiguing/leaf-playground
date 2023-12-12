from pydantic import Field

from leaf_playground.core.scene_definition import *
from leaf_playground.data.message import TextMessage
from leaf_playground.data.profile import Profile


class ExaminerQuestion(TextMessage):
    question_id: int = Field(default=...)


class ExamineeAnswer(TextMessage):
    question_id: int = Field(default=...)


SCENE_DEFINITION = SceneDefinition(
    name="GeneralMCQExamine",
    description="A general multiple choices questioning examine scene that uses dataset "
                "from huggingface hub to test agents.",
    roles=[
        RoleDefinition(
            name="examiner",
            description="the one that participants in an multiple choices question examine to monitor the examinees",
            num_agents_range=(1, 1),
            is_static=True,
            actions=[]
        ),
        RoleDefinition(
            name="examinee",
            description="the one that participants in an multiple choices question examine to answer questions",
            num_agents_range=(1, -1),
            is_static=False,
            actions=[
                ActionDefinition(
                    name="answer_question",
                    description="answering the question sent by examiner",
                    signature=ActionSignatureDefinition(
                        parameters=[
                            ActionSignatureParameterDefinition(
                                name="question",
                                annotation=ExaminerQuestion
                            ),
                            ActionSignatureParameterDefinition(
                                name="examiner",
                                annotation=Profile
                            )
                        ],
                        return_annotation=ExamineeAnswer,
                        is_static_method=False
                    ),
                    metrics=[
                        MetricDefinition(
                            name="accurate",
                            description="accuracy of examinee's answer",
                            metric_dtype=MetricType.SCALAR,
                            expect_resp_msg_type=ExamineeAnswer,
                            is_comparison=False
                        )
                    ],
                )
            ]
        )
    ],
    env_vars=[]
)


__all__ = [
    "ExaminerQuestion",
    "ExamineeAnswer",
    "SCENE_DEFINITION"
]
