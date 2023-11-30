from typing import Dict, List, Optional, Type

from pydantic import Field

from leaf_playground.core.scene_evaluator import *
from leaf_playground.utils.import_util import DynamicObject
from leaf_playground.zoo.general_mcq_examine.dataset_utils import DatasetConfig
from leaf_playground.zoo.general_mcq_examine.scene_agent import ExamineeAnswer


class IsCorrect(NestedMetricRecord):
    @classmethod
    def calculate(
        cls,
        target: ExamineeAnswer,
        evaluator: "GeneralMCQSceneEvaluator",
        metric_record_types: Optional[Dict[str, Type[MetricRecord]]] = None,
    ) -> "IsCorrect":
        data = evaluator.dataset[target.question_id]
        question = data[evaluator.dataset_config.question_column]
        agent_answer = target.content.text.strip()
        golden_answer = data[evaluator.dataset_config.golden_answer_column]
        return cls(
            value={"is_correct": agent_answer.lower().startswith(golden_answer.lower())},
            target_agent=target.sender_id,
            misc={"question": question, "agent_answer": agent_answer, "golden_answer": golden_answer}
        )


class AccurateInfo(NestedMetric):
    @classmethod
    def calculate(
        cls,
        records: List[NestedMetricRecord],
        evaluator: "GeneralMCQSceneEvaluator"
    ) -> "AccurateInfo":
        acc_num = sum(record.value["is_correct"] for record in records)
        return cls(
            value={"accuracy": acc_num / len(records), "acc_num": acc_num},
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

    _metric_configs: Optional[List[MetricConfig]] = None
    _nested_metric_configs: Optional[List[NestedMetricConfig]] = [
        NestedMetricConfig(
            metric_name="AccurateInfo",
            metric_description="Accurate information of each examinee.",
            metric_type=AccurateInfo,
            metric_record_type=IsCorrect,
            metric_sub_record_types=None,
            chart_type=NestedNightingaleRoseChart
        )
    ]
    _comparison_configs: Optional[List[ComparisonConfig]] = None
    _target_type = ExamineeAnswer

    def post_init(self, dataset: List[dict], dataset_config: DatasetConfig) -> None:
        self.dataset = dataset
        self.dataset_config = dataset_config
