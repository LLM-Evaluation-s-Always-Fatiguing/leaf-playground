import random
from abc import abstractmethod
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Union
from uuid import UUID

from pyecharts.charts.chart import Chart as EChart

from ..metric import ComparisonMetric, Metric, NestedMetric, MetricTypes


class Chart:
    def __init__(
        self,
        name: str,
        reports: List[Dict[str, Union[Dict[UUID, Union[Metric, NestedMetric]], ComparisonMetric]]],
        metric_name: str,
        metric_type: MetricTypes,
        aggregate_method: Callable[[List[Union[float, Dict[str, float]]]], Union[float, Dict[str, float]]],
        agents_name: Optional[Dict[UUID, str]] = None,
        agents_color: Optional[Dict[UUID, str]] = None,
    ):
        agents = set()
        for report in reports:
            if metric_type == MetricTypes.COMPARISON:
                agents.update(report[metric_name].value.keys())
            else:
                agents.update(report[metric_name].keys())

        agent2values: Dict[UUID, List[Union[float, Dict[str, float]]]] = defaultdict(list)
        if metric_type == MetricTypes.METRIC:
            for report in reports:
                for agent in agents:
                    metric: Metric = report[metric_name][agent]
                    agent2values[agent].append(metric.value)
        elif metric_type == MetricTypes.NESTED_METRIC:
            for report in reports:
                for agent in agents:
                    metric: NestedMetric = report[metric_name][agent]
                    agent2values[agent].append(metric.value)
        else:
            for report in reports:
                for agent in agents:
                    metric: ComparisonMetric = report[metric_name]
                    agent2values[agent].append(metric.value[agent])

        self.name = name
        self.metric_name = metric_name
        self.metric_type = metric_type

        def _build_agent_name(agent_id: UUID) -> str:
            if not agents_name:
                return agent_id.hex
            return f"{agents_name[agent_id]}({agent_id.hex[:8]})"

        self.agents_color = (
            {_build_agent_name(agent): agent_color for agent, agent_color in agents_color.items()}
            if (agents_color and all(agents_color.values())) else None
        )
        self.data = [(_build_agent_name(agent), aggregate_method(agent2values[agent])) for agent in agents]

        self.chart = self._build_chart()

    @staticmethod
    def _gen_random_colors(num_colors: int) -> List[str]:
        """Generate random bright colors"""

        col_arr = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f"]
        colors = []
        while len(colors) != num_colors:
            combination = ["90"]
            combination.insert(random.randint(0, 1), "ff")
            third_color = "{0}{1}".format(random.choice(col_arr), random.choice(col_arr))
            combination.insert(random.randint(0, 2), third_color)
            color = "#" + "".join(combination)
            if color not in colors:
                colors.append(color)
        return colors

    @abstractmethod
    def _build_chart(self) -> EChart:
        pass

    def render_chart(self, save_path: str) -> None:
        self.chart.render(save_path)

    def dump_chart_options(self, save_path: str) -> None:
        with open(save_path, "w") as f:
            f.write(self.chart.dump_options())


__all__ = [
    "Chart"
]
