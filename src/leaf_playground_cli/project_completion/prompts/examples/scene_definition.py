from typing import List, Literal, Union
from typing_extensions import Annotated

from leaf_playground.core.scene_definition import *
from leaf_playground.core.scene_definition.definitions.metric import _RecordData
from leaf_playground.data.message import TextMessage
from leaf_playground.data.profile import Profile
from pydantic import Field

from .dataset_util import DatasetConfig


def accuracy_fn(records: List[_RecordData]) -> AggregationMethodOutput:
    num_records = len(records)
    num_accurate = len([record for record in records if bool(record.value)])
    accuracy = round(num_accurate / num_records, 8)
    return AggregationMethodOutput(value=accuracy)


class ExaminerSample(TextMessage):
    sample_id: int = Field(default=...)
    msg_type: Literal["sample"] = Field(default="sample")


class ExamineeAnswer(TextMessage):
    sample_id: int = Field(default=...)
    msg_type: Literal["answer"] = Field(default="answer")


MessageType = Annotated[Union[ExaminerSample, ExamineeAnswer], Field(discriminator="msg_type")]

SCENE_DEFINITION = SceneDefinition(
    name="Mmlu",
    description="using MMLU dataset to test and evaluate agents.",
    roles=[
        RoleDefinition(
            name="examiner",
            description="the one who prepares samples and sends to examinees.",
            num_agents_range=(1, 1),
            is_static=True,
            actions=[
                ActionDefinition(
                    name="prepare_samples",
                    description="prepare samples based on dataset config.",
                    signature=ActionSignatureDefinition(
                        parameters=[
                            ActionSignatureParameterDefinition(
                                name="ds_config",
                                annotation=DatasetConfig
                            )
                        ],
                        return_annotation=None
                    )
                ),
                ActionDefinition(
                    name="send_sample",
                    description="pick one sample and broadcast to all examinees.",
                    signature=ActionSignatureDefinition(
                        parameters=[
                            ActionSignatureParameterDefinition(
                                name="receivers",
                                annotation=List[Profile]
                            )
                        ],
                        return_annotation=ExaminerSample
                    )
                )
            ]
        ),
        RoleDefinition(
            name="examinee",
            description="the one who receives samples sent by examiner and answer to those samples.",
            num_agents_range=(1, -1),
            is_static=False,
            actions=[
                ActionDefinition(
                    name="answer",
                    description="answering the question sent by examiner",
                    signature=ActionSignatureDefinition(
                        parameters=[
                            ActionSignatureParameterDefinition(
                                name="sample",
                                annotation=ExaminerSample
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
                            record_value_dtype=ValueDType.BOOLEAN,
                            record_display_type=DisplayType.BOOLEAN_RADIO,
                            expect_resp_msg_type=ExamineeAnswer,
                            agg_method=DynamicAggregationFn.create_dynamic_fn(fn=accuracy_fn),
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
    "ExaminerSample",
    "ExamineeAnswer",
    "MessageType",
    "SCENE_DEFINITION"
]
