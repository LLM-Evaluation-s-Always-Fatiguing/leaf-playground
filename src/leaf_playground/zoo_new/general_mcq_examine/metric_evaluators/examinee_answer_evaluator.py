from typing import Any, Dict, List, Optional

from leaf_playground.core.scene_observer import MetricEvaluatorConfig, MetricEvaluatorProxy, MetricEvaluator
from leaf_playground.core.scene_observer.metric_evaluator import _MetricName, _MetricRecordValue
from leaf_playground.data.log_body import LogBody

from ..scene_definition import ExamineeAnswer, ExaminerQuestion, SCENE_DEFINITION


ROLE_DEFINITION = SCENE_DEFINITION.get_role_definition("examinee")


class ExamineeAnswerEvaluatorConfig(MetricEvaluatorConfig):
    pass


class ExamineeAnswerEvaluatorProxy(MetricEvaluatorProxy):

    def _init_evaluator(self, config: ExamineeAnswerEvaluatorConfig) -> Any:
        return

    async def _cal_record_value(self, log: LogBody, evaluator: Any) -> Dict[_MetricName, _MetricRecordValue]:
        result = {}
        if isinstance(log.response, ExamineeAnswer) and log.ground_truth:
            answer = log.response.content.text
            result["accurate"] = answer.lower().startswith(log.ground_truth.lower())
        return result

    async def _comment_record(self, log: LogBody, evaluator: Any) -> Optional[Dict[_MetricName, str]]:
        return None

    async def _collect_record_misc(self, log: LogBody, evaluator: Any) -> Optional[Dict[_MetricName, dict]]:
        return None

    async def _cal_compare_value(self, logs: List[LogBody], evaluator: Any) -> Dict[_MetricName, _MetricRecordValue]:
        return {}

    async def _comment_compare(self, logs: List[LogBody], evaluator: Any) -> Optional[Dict[_MetricName, str]]:
        return None

    async def _collect_compare_misc(self, logs: List[LogBody], evaluator: Any) -> Optional[Dict[_MetricName, dict]]:
        return None


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
