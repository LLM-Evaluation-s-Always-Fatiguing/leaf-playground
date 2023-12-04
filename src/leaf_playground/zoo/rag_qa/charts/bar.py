from pyecharts import options as opts
from pyecharts.charts import Bar
from pyecharts.commons.utils import JsCode

from leaf_playground.core.scene_evaluator.chart.base import Chart


class RagasBarChart(Chart):
    stacked: bool = False

    def _build_chart(self) -> Bar:
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
            .set_series_opts(
                label_opts=opts.LabelOpts(formatter=JsCode("function(c){return (c.data * 100).toFixed(2) + '%';}")))
            .set_global_opts(
                title_opts=opts.TitleOpts(title=self.name, pos_left="center"),
                legend_opts=opts.LegendOpts(is_show=False, pos_bottom=0),
                xaxis_opts=opts.AxisOpts(name="Agent"),
                yaxis_opts=opts.AxisOpts(
                    name=self.metric_name,
                    axislabel_opts=opts.LabelOpts(
                        formatter=JsCode("function(value){return (value * 100).toFixed(2) + '%';}"))
                ),
            )
            .set_colors(self._gen_random_colors(len(self.data[0][1])))
        )

        return bar


__all__ = [
    "RagasBarChart",
]
