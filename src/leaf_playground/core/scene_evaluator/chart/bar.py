from pyecharts import options as opts
from pyecharts.charts import Bar
from pyecharts.commons.utils import JsCode

from .base import Chart
from ..metric import MetricTypes


class BarChart(Chart):
    stacked: bool = False

    def _build_chart(self) -> Bar:
        if self.metric_type != MetricTypes.NESTED_METRIC:
            itemstyle_opts = None
            if self.agents_color:
                color_function = f"""
                function (params) {{
                    var colors = {self.agents_color};
                    return colors[params.name];
                }}
                """
                itemstyle_opts = opts.ItemStyleOpts(color=JsCode(color_function))

            bar = (
                Bar()
                .add_xaxis([item[0] for item in self.data])
                .add_yaxis(
                    self.metric_name,
                    [item[1] for item in self.data],
                    category_gap="50%",
                    itemstyle_opts=itemstyle_opts,
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
                .set_colors(self._gen_random_colors(len(self.data[0][1])))
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
        )


class StackedBarChart(BarChart):
    stacked: bool = True


class HorizontalBarChart(BarChart):
    def _build_chart(self) -> Bar:
        return super()._build_chart().reversal_axis()


class StackedHorizontalBarChart(HorizontalBarChart):
    stacked: bool = True


__all__ = [
    "BarChart",
    "StackedBarChart",
    "HorizontalBarChart",
    "StackedHorizontalBarChart"
]
