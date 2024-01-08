from collections import defaultdict
from uuid import UUID
from typing import Any, Dict, List, Type

from .chart import Chart
from ..scene_definition import CombinedMetricsData, MetricDefinition, SceneConfig, SceneDefinition
from ..scene_definition.definitions.metric import (
    _RecordData,
    _MetricData,
    DynamicAggregationMethod,
    DEFAULT_AGG_METHODS,
)
from ...data.log_body import LogBody
from ...utils.import_util import dynamically_import_fn


def _agg(records: List[_RecordData], agg_method: DynamicAggregationMethod) -> Any:
    if isinstance(agg_method, str):
        agg_method_ = DEFAULT_AGG_METHODS[agg_method]
    else:
        agg_method_ = dynamically_import_fn(agg_method)
    return agg_method_(records).value


def _win_ratio(ranks: List[List[str]], top_n: int):
    counter = {agent: 0 for agent in ranks[0]}
    for rank in ranks:
        for agent in rank[:top_n]:
            counter[agent] += 1
    return {k: v / len(ranks) for k, v in counter.items()}


class MetricReporter:
    def __init__(self, scene_definition: SceneDefinition, chart_classes: List[Type[Chart]]):
        metric_definitions = {}
        for role in scene_definition.roles:
            for action in role.actions:
                for metric_def in action.metrics or []:
                    metric_definitions[metric_def.belonged_chain] = metric_def
        self.metric_definitions: Dict[str, MetricDefinition] = metric_definitions
        self.records: Dict[str, List[_RecordData]] = defaultdict(list)
        self.charts = [chart_cls() for chart_cls in chart_classes]
        self.human_records: Dict[str, Dict[UUID, _RecordData]] = defaultdict(dict)

    def put_record(self, record: _RecordData, metric_belonged_chain: str):
        self.records[metric_belonged_chain].append(record)

    def put_human_record(self, record: _RecordData, metric_belonged_chain: str, log_id: UUID):
        self.human_records[metric_belonged_chain][log_id] = record

    def _cal_record_metric(self, records: List[_RecordData], metric_def: MetricDefinition) -> List[_MetricData]:
        agg_method = metric_def.agg_method
        metric_data_model, _ = metric_def.create_data_models()

        agent2records = defaultdict(list)
        for record in records:
            agent2records[record.target_agent].append(record)

        metrics = []
        for agent_id, agent_records in agent2records.items():
            metrics.append(
                metric_data_model(value=_agg(agent_records, agg_method), records=agent_records, target_agent=agent_id)
            )

        return metrics

    def _cal_compare_metric(self, records: List[_RecordData], metric_def: MetricDefinition) -> _MetricData:
        metric_data_model, _ = metric_def.create_data_models()
        record_values: List[List[str]] = [record.value for record in records]

        num_agents = len(record_values[0])

        result = {"win_ratio": _win_ratio(record_values, top_n=1)}
        if num_agents >= 10:
            result["top3_ratio"] = _win_ratio(record_values, top_n=3)
        if num_agents >= 30:
            result["top10_ratio"] = _win_ratio(record_values, top_n=10)

        return metric_data_model(value=result, records=records)

    def _cal_metrics(self) -> CombinedMetricsData:
        metrics = {}
        human_metrics = {}

        for metric_belonged_chain, records in self.records.items():
            metric_def = self.metric_definitions[metric_belonged_chain]
            is_compare = metric_def.is_comparison

            if not is_compare:
                metrics[metric_belonged_chain] = self._cal_record_metric(records, metric_def)
            else:
                metrics[metric_belonged_chain] = self._cal_compare_metric(records, metric_def)

        for metric_belonged_chain, records in self.human_records.items():
            metric_def = self.metric_definitions[metric_belonged_chain]
            is_compare = metric_def.is_comparison

            if not is_compare:
                human_metrics[metric_belonged_chain] = self._cal_record_metric(list(records.values()), metric_def)
            else:
                human_metrics[metric_belonged_chain] = self._cal_compare_metric(list(records.values()), metric_def)

        return {"metrics": metrics, "human_metrics": human_metrics}

    @property
    def metrics_data(self) -> CombinedMetricsData:
        return self._cal_metrics()

    def generate_reports(
        self,
        scene_config: SceneConfig,
        evaluator_configs: List["leaf_playground.core.workers.MetricEvaluatorConfig"],
        logs: List[LogBody],
    ):
        metrics = self.metrics_data

        charts = {
            chart.chart_name: chart.generate(metrics, scene_config, evaluator_configs, logs) for chart in self.charts
        }
        charts = {chart_name: chart for chart_name, chart in charts.items() if chart is not None}

        return metrics, charts


__all__ = ["MetricReporter"]
