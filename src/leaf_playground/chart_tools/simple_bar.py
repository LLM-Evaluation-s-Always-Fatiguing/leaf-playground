from typing import Literal

import altair as alt
from pandas import DataFrame

from .base_vega_chart import BaseVegaChart


class SimpleBar(BaseVegaChart):
    def __init__(self, mode: Literal["percent", "value"] = "value", max_value=1.0):
        self.mode = mode
        self.max_value = max_value

    def generate(self, data: DataFrame) -> dict:
        # use the data format [{agent: gpt4, color: "#7D3C98", value: 4}, ...]

        fmt = ".2f" if self.mode == "value" else ".1%"

        chart = (
            alt.Chart(data)
            .mark_bar(cornerRadius=5, height={"band": 0.6})
            .encode(
                y=alt.X("agent:N").axis(labelAngle=0, title=None),
                x=alt.Y("value:Q").axis(title=None, format=fmt).scale(domain=(0, self.max_value)),
                color=alt.Color("color:N").scale(None).legend(None),
            )
        )

        text = chart.mark_text(align="left", dx=3).encode(text=alt.Text("value:Q", format=fmt))

        return (chart + text).to_dict()


if __name__ == "__main__":
    # This is a test
    from pandas import DataFrame
    import json

    data = DataFrame([
        {"agent": "gpt4", "color": "#7D3C98", "value": 4},
        {"agent": "gpt3", "color": "#F39C12", "value": 3},
        {"agent": "gpt2", "color": "#3498DB", "value": 2},
        {"agent": "gpt1", "color": "#E74C3C", "value": 1},
    ])

    chart = SimpleBar(mode="value", max_value=5)
    print(json.dumps(chart.generate(data)))
