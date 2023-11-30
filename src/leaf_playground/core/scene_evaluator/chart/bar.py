from pyecharts import options as opts
from pyecharts.charts import Bar

from .base import Chart


class BarChart(Chart):
    def _build_chart(self) -> Bar:
        colors = self._gen_random_colors(1)[0]
        return (
            Bar()
            .add_xaxis([item[0] for item in self.data])
            .add_yaxis(
                self.metric_name,
                [item[1] for item in self.data],
                category_gap="50%",
            )
            .set_series_opts(label_opts=opts.LabelOpts(formatter="{c}"))
            .set_global_opts(
                title_opts=opts.TitleOpts(title=self.name, pos_left="center"),
                legend_opts=opts.LegendOpts(is_show=False, pos_bottom=0),
                yaxis_opts=opts.AxisOpts(
                    axislabel_opts=opts.LabelOpts(formatter="{value}%")
                ),
            )
            .set_colors(colors)
        )


class HorizontalBarChart(BarChart):
    def _build_chart(self) -> Bar:
        return super()._build_chart().reversal_axis()


__all__ = [
    "BarChart",
    "HorizontalBarChart"
]
