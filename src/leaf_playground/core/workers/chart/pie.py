import math
from typing import List, Optional

from pyecharts import options as opts
from pyecharts.charts import Pie

from .base import Chart
from ..metric import MetricTypes


class PieChart(Chart):
    rosetype: Optional[str] = None
    nested: bool = False

    def _build_chart(self) -> Pie:
        if self.metric_type != MetricTypes.NESTED_METRIC:
            pie = (
                Pie()
                .add(
                    self.metric_name,
                    self.data,
                    radius=["5%", "85%"],
                    rosetype=self.rosetype,
                )
            )
        else:
            pie = Pie()
            metric_names = list(self.data[0][1].keys())
            num_metrics = len(metric_names)

            num_columns = math.floor(math.sqrt(num_metrics)) + 1
            num_row = math.ceil(num_metrics / num_columns)
            colors = self._gen_random_colors(num_metrics)

            for i, metric_name in enumerate(metric_names):
                col_idx, row_idx = i % num_columns + 1, i // num_columns + 1
                col_step = 50 / num_columns
                radius = ["0%", f"{min(35, 100 // max(num_columns, num_row))}%"]
                center = [f"{col_step * (col_idx * 2 - 1)}%", f"{col_step * (row_idx * 2 - 1)}%"]
                if self.nested:
                    radius = [
                        f"{(85 / num_metrics) * i}%",
                        f"{(85 / num_metrics) * (i + 1) - 1}%"
                    ]
                    center = None

                pie.add(
                    metric_name,
                    [(item[0], item[1][metric_name]) for item in self.data],
                    color=colors[i],
                    radius=radius,
                    center=center,
                    rosetype=self.rosetype,
                )

        return (
            pie
            .set_series_opts(label_opts=opts.LabelOpts(formatter="{c}"))
            .set_global_opts(
                title_opts=opts.TitleOpts(title=self.name, pos_left="center"),
                legend_opts=opts.LegendOpts(is_show=False, pos_bottom=0),
            )
            .set_colors([self.agents_color[data[0]] for data in self.data])
        )


class NestedPieChart(PieChart):
    nested: bool = True


class NightingaleRoseChart(PieChart):
    rosetype: Optional[str] = "area"


class NestedNightingaleRoseChart(NightingaleRoseChart):
    nested: bool = True


__all__ = [
    "PieChart",
    "NestedPieChart",
    "NightingaleRoseChart",
    "NestedNightingaleRoseChart"
]
