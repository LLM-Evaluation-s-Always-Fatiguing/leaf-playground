from pyecharts import options as opts
from pyecharts.charts import Bar

from .base import Chart
from ..metric import MetricTypes


class BarChart(Chart):
    stacked: bool = False

    def _build_chart(self) -> Bar:
        n_series = 1 if self.metric_type != MetricTypes.NESTED_METRIC else len(self.data[0][1])
        colors = self._gen_random_colors(n_series)

        if self.metric_type != MetricTypes.NESTED_METRIC:
            bar = (
                Bar()
                .add_xaxis([item[0] for item in self.data])
                .add_yaxis(
                    self.metric_name,
                    [item[1] for item in self.data],
                    category_gap="50%",
                )
            )
        else:
            bar = Bar().add_xaxis([item[0] for item in self.data])
            for metric_name in self.data[0][1].keys():
                bar.add_yaxis(
                    metric_name,
                    [item[1][metric_name] for item in self.data],
                    category_gap="50%",
                    stack=None if not self.stacked else "stack_1"
                )
            bar = (
                bar
                .set_series_opts(label_opts=opts.LabelOpts(formatter="{c}"))
                .set_global_opts(
                    title_opts=opts.TitleOpts(title=self.name, pos_left="center"),
                    legend_opts=opts.LegendOpts(is_show=False, pos_bottom=0),
                    xaxis_opts=opts.AxisOpts(name="Agent"),
                    yaxis_opts=opts.AxisOpts(
                        name=self.metric_name,
                        axislabel_opts=opts.LabelOpts(formatter="{value}%")
                    ),
                )
                .set_colors(colors)
            )

        return (
            bar
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


class StackedBarChart(BarChart):
    stacked: bool = True


class HorizontalBarChart(BarChart):
    def _build_chart(self) -> Bar:
        return super()._build_chart().reversal_axis()


class HorizontalStackedBarChart(HorizontalBarChart):
    stacked: bool = True


__all__ = [
    "BarChart",
    "StackedBarChart",
    "HorizontalBarChart",
    "HorizontalStackedBarChart"
]
