from typing import List, Literal, Union
from typing_extensions import Annotated

from pydantic import Field

from leaf_playground.core.scene_definition import *
from leaf_playground.core.scene_definition.definitions.metric import _RecordData, AggregationMethodOutput
from leaf_playground.data.message import TextMessage
from leaf_playground.data.profile import Profile


def accuracy_fn(records: List[_RecordData]) -> AggregationMethodOutput:
    num_records = len(records)
    num_accurate = len([record for record in records if bool(record.value)])
    accuracy = round(num_accurate / num_records, 8)
    return AggregationMethodOutput(value=accuracy)


class ExaminerQuestion(TextMessage):
    question_id: int = Field(default=...)
    msg_type: Literal["question"] = Field(default="question")


class ExamineeAnswer(TextMessage):
    question_id: int = Field(default=...)
    msg_type: Literal["answer"] = Field(default="answer")


MessageType = Annotated[Union[ExaminerQuestion, ExamineeAnswer], Field(discriminator="msg_type")]


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
                            record_value_dtype=ValueDType.SCALAR,
                            expect_resp_msg_type=ExamineeAnswer,
                            agg_method_when_not_compare=DynamicAggregationFn.create_dynamic_fn(fn=accuracy_fn),
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
    "MessageType",
    "SCENE_DEFINITION"
]
