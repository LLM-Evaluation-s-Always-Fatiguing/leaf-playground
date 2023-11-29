from pyecharts import options as opts
from pyecharts.charts import Pie

from .base import Chart


class PieChart(Chart):
    def _build_chart(self) -> Pie:
        return (
            Pie()
            .add(
                self.metric_name,
                self.data
            )
            .set_series_opts(label_opts=opts.LabelOpts(formatter="{c}"))
            .set_global_opts(
                title_opts=opts.TitleOpts(title=self.name, pos_left="center"),
                legend_opts=opts.LegendOpts(is_show=False, pos_bottom=0),
            )
            .set_colors(self._gen_random_colors(len(self.data)))
        )


class NightingaleRoseChart(PieChart):
    def _build_chart(self) -> Pie:
        return (
            Pie()
            .add(
                self.metric_name,
                sorted(self.data, key=lambda x: x[1]),
                radius=["5%", "100%"],
                center=["50%", "65%"],
                rosetype="area",
            )
            .set_series_opts(label_opts=opts.LabelOpts(formatter="{c}"))
            .set_global_opts(
                title_opts=opts.TitleOpts(title=self.name, pos_left="center"),
                legend_opts=opts.LegendOpts(is_show=False, pos_bottom=0),
            )
            .set_colors(self._gen_random_colors(len(self.data)))
        )


__all__ = [
    "PieChart",
    "NightingaleRoseChart"
]
