from collections import defaultdict
from typing import Dict, List

from ..scene_definition import SceneDefinition, MetricDefinition
from ..scene_definition.definitions.metric import _RecordData


class MetricReporter:
    def __init__(self, scene_definition: SceneDefinition):
        metric_definitions = {}
        for role in scene_definition.roles:
            for action in role.actions:
                for metric_def in action.metrics:
                    metric_definitions[metric_def.belonged_chain] = metric_def
        self.metric_definitions: Dict[str, MetricDefinition] = metric_definitions
        self.records: Dict[str, List[_RecordData]] = defaultdict(list)

    def put_record(self, record: _RecordData, metric_belonged_chain: str):
        self.records[metric_belonged_chain].append(record)
