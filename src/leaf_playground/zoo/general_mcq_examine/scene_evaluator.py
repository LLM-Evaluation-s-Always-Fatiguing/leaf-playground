from typing import List, Optional

from pydantic import Field

from leaf_playground.metric.base import ComparisonConfig, MetricRecord, Metric, MetricConfig
from leaf_playground.chart.bar import HorizontalBarChart
from leaf_playground.core.scene_evaluator import *
from leaf_playground.utils.import_util import DynamicObject
from leaf_playground.zoo.general_mcq_examine.dataset_utils import DatasetConfig
from leaf_playground.zoo.general_mcq_examine.scene_agent import ExamineeAnswer


class IsCorrect(MetricRecord):
    value: bool = Field(default=...)

    @classmethod
    def calculate(
        cls,
        target: ExamineeAnswer,
        evaluator: "GeneralMCQSceneEvaluator"
    ) -> "IsCorrect":
        data = evaluator.dataset[target.question_id]
        question = data[evaluator.dataset_config.question_column]
        agent_answer = target.content.text.strip()
        golden_answer = data[evaluator.dataset_config.golden_answer_column]
        return cls(
            value=agent_answer.lower().startswith(golden_answer.lower()),
            target_agent=target.sender_id,
            misc={"question": question, "agent_answer": agent_answer, "golden_answer": golden_answer}
        )


class Accuracy(Metric):
    value: float = Field(default=..., ge=0.0, le=1.0)

    @classmethod
    def calculate(
        cls,
        records: List[MetricRecord],
        evaluator: "GeneralMCQSceneEvaluator"
    ) -> "Accuracy":
        return cls(
            value=sum(record.value for record in records) / len(records),
            records=records
        )


class GeneralMCQSceneEvaluatorConfig(SceneEvaluatorConfig):
    pass


class GeneralMCQSceneEvaluator(SceneEvaluator):
    config_obj = GeneralMCQSceneEvaluatorConfig
    config: GeneralMCQSceneEvaluatorConfig

    description = "General MCQ Scene Evaluator that calculates each examinee's accuracy."
    obj_for_import: DynamicObject = DynamicObject(
        obj="GeneralMCQSceneEvaluator",
        module="leaf_playground.zoo.general_mcq_examine.scene_evaluator"
    )

    _metric_configs: Optional[List[MetricConfig]] = [
        MetricConfig(
            metric_name="accuracy",
            metric_description="Accuracy of an examinee's answer.",
            metric_type=Accuracy,
            metric_record_type=IsCorrect,
            chart_type=HorizontalBarChart,
        )
    ]
    _comparison_configs: Optional[List[ComparisonConfig]] = None
    _target_type = ExamineeAnswer

    def __init__(self, config: config_obj):
        super().__init__(config=config)

    def post_init(self, dataset: List[dict], dataset_config: DatasetConfig) -> None:
        self.dataset = dataset
        self.dataset_config = dataset_config
