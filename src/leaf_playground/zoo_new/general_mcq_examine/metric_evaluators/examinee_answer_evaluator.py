from typing import Any, Dict, List

from leaf_playground.core.scene_observer import MetricEvaluatorConfig, MetricEvaluatorProxy, MetricEvaluator
from leaf_playground.core.scene_observer.metric_evaluator import (
    _MetricName, CompareOutput, RecordOutput
)
from leaf_playground.data.log_body import LogBody

from ..scene_definition import ExamineeAnswer, SCENE_DEFINITION


ROLE_DEFINITION = SCENE_DEFINITION.get_role_definition("examinee")


class ExamineeAnswerEvaluatorConfig(MetricEvaluatorConfig):
    pass


class ExamineeAnswerEvaluatorProxy(MetricEvaluatorProxy):

    def _init_evaluator(self, config: ExamineeAnswerEvaluatorConfig) -> Any:
        return

    async def _record(self, log: LogBody, evaluator: Any) -> Dict[_MetricName, RecordOutput]:
        result = {}
        if isinstance(log.response, ExamineeAnswer) and log.ground_truth:
            answer = log.response.content.text
            result["accurate"] = RecordOutput(record_value=answer.lower().startswith(log.ground_truth.lower()))
        return result

    async def _compare(self, logs: List[LogBody], evaluator: Any) -> Dict[_MetricName, CompareOutput]:
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
