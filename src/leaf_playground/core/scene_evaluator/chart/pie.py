from typing import List, Optional

from pyecharts import options as opts
from pyecharts.charts import Pie

from .base import Chart
from ..metric import MetricTypes


class PieChart(Chart):
    radius: Optional[List[str]] = None
    center: Optional[List[str]] = None
    rosetype: Optional[str] = None

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
            for metric_name in self.data[0][1].keys():
                pie.add(
                    metric_name,
                    [item[1][metric_name] for item in self.data],
                    radius=self.radius,
                    center=self.center,
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


class NightingaleRoseChart(PieChart):
    radius: Optional[List[str]] = ["5%", "100%"]
    center: Optional[List[str]] = ["50%", "65%"]
    rosetype: Optional[str] = "area"


__all__ = [
    "PieChart",
    "NightingaleRoseChart"
]
