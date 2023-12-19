from typing import Any, Dict, List

from leaf_playground.core.workers import MetricEvaluatorConfig, MetricEvaluatorProxy, MetricEvaluator
from leaf_playground.core.workers.evaluator import _MetricName, CompareOutput, RecordOutput
from leaf_playground.data.log_body import ActionLogBody

from ..scene_definition import ExamineeAnswer, SCENE_DEFINITION


ROLE_DEFINITION = SCENE_DEFINITION.get_role_definition("examinee")


class ExamineeAnswerEvaluatorConfig(MetricEvaluatorConfig):
    pass


class ExamineeAnswerEvaluatorProxy(MetricEvaluatorProxy):

    def _init_evaluator(self, config: ExamineeAnswerEvaluatorConfig) -> Any:
        return

    async def _record(self, log: ActionLogBody, evaluator: Any) -> Dict[_MetricName, RecordOutput]:
        result = {}
        if isinstance(log.response, ExamineeAnswer) and log.ground_truth:
            answer = log.response.content.text
            ground_truth = log.ground_truth.text
            result["accurate"] = RecordOutput(
                record_value=answer.lower().startswith(ground_truth.lower()),
                misc={
                    "question": log.references[0].content.text,
                    "agent_answer": answer,
                    "ground_truth": ground_truth
                }
            )
        return result

    async def _compare(self, logs: List[ActionLogBody], evaluator: Any) -> Dict[_MetricName, CompareOutput]:
        return {}


class ExamineeAnswerEvaluator(
    MetricEvaluator,
    metric_definitions=ROLE_DEFINITION.get_action_definition("answer_question").metrics,
    cls_description="An evaluator that evaluate examinee's answers",
    evaluator_proxy_class=ExamineeAnswerEvaluatorProxy
):
    config_cls = ExamineeAnswerEvaluatorConfig
    config: config_cls


__all__ = [
    "ExamineeAnswerEvaluatorConfig",
    "ExamineeAnswerEvaluator"
]
