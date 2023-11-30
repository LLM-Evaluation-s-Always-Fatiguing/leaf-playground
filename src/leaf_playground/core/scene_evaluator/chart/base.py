import random
from abc import abstractmethod
from collections import defaultdict
from typing import Callable, Dict, List, Union
from uuid import UUID

from pyecharts.charts.chart import Chart as EChart

from ..metric import ComparisonMetric, Metric, MetricTypes


class Chart:
    def __init__(
        self,
        name: str,
        reports: List[Dict[str, Union[Dict[UUID, Metric], ComparisonMetric]]],
        metric_name: str,
        metric_type: MetricTypes,
        aggregate_method: Callable[[List[float]], float]
    ):
        agents = set()
        for report in reports:
            if metric_type == MetricTypes.METRIC:
                agents.update(report[metric_name].keys())
            else:
                agents.update(report[metric_name].value.keys())

        agent2values = defaultdict(list)
        if metric_type == MetricTypes.METRIC:
            for report in reports:
                for agent in agents:
                    metric = report[metric_name][agent]
                    agent2values[agent].append(metric.value)
        else:
            for report in reports:
                for agent in agents:
                    metric = report[metric_name]
                    agent2values[agent].append(metric.value[agent])

        self.name = name
        self.metric_name = metric_name
        self.data = [(str(agent), aggregate_method(agent2values[agent])) for agent in agents]

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
        chart = self._build_chart()
        chart.render(save_path)

    def dump_chart_options(self, save_path: str) -> None:
        chart = self._build_chart()
        with open(save_path, "w") as f:
            f.write(chart.dump_options())


__all__ = [
    "Chart"
]
