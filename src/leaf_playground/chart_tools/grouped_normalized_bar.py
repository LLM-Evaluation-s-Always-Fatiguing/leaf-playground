from typing import Literal

from pandas import DataFrame
import altair as alt

from .base_vega_chart import BaseVegaChart


class GroupedNormalizedBar(BaseVegaChart):

    def __init__(self, mode: Literal["percent", "value"] = "value"):
        self.mode = mode

    def generate(self, data: DataFrame) -> dict:
        # use the data format [{agent: gpt4, metric: accuracy, value: 0.81}, ...]

        fmt = ".2f" if self.mode == "value" else ".1%"

        chart = alt.Chart(data).mark_bar(cornerRadius=5, height={"band": 0.6}).encode(
            y=alt.X("metric:N").axis(labelAngle=0, title=None),
            x=alt.Y("value:Q").axis(title="value", format=fmt).scale(domain=(0, 1)),
            color=alt.Color("metric:N"),
        )

        text = chart.mark_text(align='left', dx=3).encode(
            text=alt.Text("value:Q", format=fmt)
        )

        return (chart + text).facet(row=alt.Row("agent:N", header=alt.Header(title=None))).to_dict()
