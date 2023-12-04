from typing import Dict, List, Optional, Type, Literal, get_args

from datasets import Dataset, Features, Value, Sequence
from pydantic import Field
from ragas import evaluate

from leaf_playground.core.scene_agent import SceneAgent
from leaf_playground.core.scene_evaluator import *
from leaf_playground.utils.import_util import DynamicObject
from leaf_playground.zoo.rag_qa.charts.bar import RagasBarChart
from leaf_playground.zoo.rag_qa.dataset_utils import DatasetConfig
from leaf_playground.zoo.rag_qa.scene_agent import ExamineeAnswer

from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    answer_similarity,
    context_precision,
    context_recall,
    context_relevancy,
    faithfulness
)

MetricType = Literal[
    "answer_correctness",
    "answer_relevancy",
    "answer_similarity",
    "context_precision",
    "context_recall",
    "context_relevancy",
    "faithfulness"
]

ragas_metrics_map = {
    "answer_correctness": answer_correctness,
    "answer_relevancy": answer_relevancy,
    "answer_similarity": answer_similarity,
    "context_precision": context_precision,
    "context_recall": context_recall,
    "context_relevancy": context_relevancy,
    "faithfulness": faithfulness
}


class RagasMetricRecord(NestedMetricRecord):
    @classmethod
    def calculate(
            cls,
            target: ExamineeAnswer,
            evaluator: "RagasEvaluator",
            metric_record_types: Optional[Dict[str, Type[MetricRecord]]] = None,
    ) -> "RagasMetricRecord":
        data = evaluator.dataset[target.question_id]
        question = data[evaluator.dataset_config.question_column]
        ground_truth = data[evaluator.dataset_config.ground_truth_column]
        golden_answer = data[evaluator.dataset_config.golden_answer_column]
        agent_answer = target.content.text.strip()
        contexts = target.contexts

        misc = {
            'question': question,
            'answer': agent_answer,
            'contexts': contexts,
            'ground_truths': ground_truth,
            'golden_answer': golden_answer
            # Actually, it’s not used here. The original answer from ragas is the evaluated answer, and it is placed here for future reference when displaying in the log.
        }

        features = Features({
            'question': Value('string'),
            'answer': Value('string'),
            'contexts': Sequence(Value('string')),
            'ground_truths': Sequence(Value('string')),
            'golden_answer': Value('string')
        })

        def gen():
            yield misc

        dataset = Dataset.from_generator(gen, features=features)

        result = evaluate(dataset, metrics=[ragas_metrics_map[m] for m in evaluator.config.activate_metrics])

        return cls(
            value=result,
            target_agent=target.sender_id,
            misc=misc
        )


class RagasMetrics(NestedMetric):
    @classmethod
    def calculate(
            cls,
            records: List[NestedMetricRecord],
            evaluator: "RagasEvaluator"
    ) -> "RagasMetrics":

        total_metrics = {}
        for record in records:
            for metric, value in record.value.items():
                if metric not in total_metrics:
                    total_metrics[metric] = value
                else:
                    total_metrics[metric] += value

        average_metrics = {metric: total / len(records) for metric, total in total_metrics.items()}

        return cls(
            value=average_metrics,
            records=records
        )


class RagasEvaluatorConfig(SceneEvaluatorConfig):
    activate_metrics: List[MetricType] = Field(default=["answer_correctness"])


class RagasEvaluator(SceneEvaluator):
    config_obj = RagasEvaluatorConfig
    config: config_obj

    description = "Retrieval augmented generation question answering examine scene evaluator powered by ragas."
    obj_for_import: DynamicObject = DynamicObject(
        obj="RagasEvaluator",
        module="leaf_playground.zoo.rag_qa.scene_evaluator"
    )

    _metric_configs: Optional[List[MetricConfig]] = None
    _nested_metric_configs: Optional[List[NestedMetricConfig]] = None
    _comparison_configs: Optional[List[ComparisonConfig]] = None
    _target_type = ExamineeAnswer

    def __init__(self, config: config_obj, agents: List[SceneAgent]):
        super().__init__(config, agents)
        self._nested_metric_configs: Optional[List[NestedMetricConfig]] = [
            NestedMetricConfig(
                metric_name='ragas metrics',
                metric_description=f'ragas metrics',
                metric_type=RagasMetrics,
                metric_record_type=RagasMetricRecord,
                chart_type=RagasBarChart
            )
        ]

    def post_init(self, dataset: List[dict], dataset_config: DatasetConfig) -> None:
        self.dataset = dataset
        self.dataset_config = dataset_config
