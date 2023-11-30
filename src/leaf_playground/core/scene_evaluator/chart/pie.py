from typing import List, Optional

from pyecharts import options as opts
from pyecharts.charts import Pie

from .base import Chart
from ..metric import MetricTypes


class PieChart(Chart):
    radius: Optional[List[str]] = None
    center: Optional[List[str]] = None
    rosetype: Optional[str] = None
    nested: bool = False

    def _build_chart(self) -> Pie:
        n_series = 1 if self.metric_type != MetricTypes.NESTED_METRIC else len(self.data[0][1])
        colors = self._gen_random_colors(n_series)

        if self.metric_type != MetricTypes.NESTED_METRIC:
            pie = (
                Pie()
                .add(
                    self.metric_name,
                    self.data,
                    radius=self.radius,
                    center=self.center,
                    rosetype=self.rosetype,
                )
            )
        else:
            pie = Pie()
            metric_names = list(self.data[0][1].keys())
            num_metrics = len(metric_names)
            for i, metric_name in enumerate(metric_names):
                pie.add(
                    metric_name,
                    [item[1][metric_name] for item in self.data],
                    radius=self.radius if not self.nested
                    else [f"{(1 / num_metrics) * i * 100}%", f"{(1 / num_metrics) * (i + 1) * 100}%"],
                    center=self.center if not self.nested else None,
                    rosetype=self.rosetype,
                )

        return (
            pie
            .set_series_opts(label_opts=opts.LabelOpts(formatter="{c}"))
            .set_global_opts(
                title_opts=opts.TitleOpts(title=self.name, pos_left="center"),
                legend_opts=opts.LegendOpts(is_show=False, pos_bottom=0),
            )
            .set_colors(colors)
        )


class NestedPieChart(PieChart):
    nested: bool = True


class NightingaleRoseChart(PieChart):
    radius: Optional[List[str]] = ["5%", "100%"]
    rosetype: Optional[str] = "area"


class NestedNightingaleRoseChart(NightingaleRoseChart):
    nested: bool = True


__all__ = [
    "PieChart",
    "NestedPieChart",
    "NightingaleRoseChart",
    "NestedNightingaleRoseChart"
]
