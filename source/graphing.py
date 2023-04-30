import _dataset as dataset

import plotly.graph_objects as go


class Graphing:
    def __init__(self, dataset: dataset.Dataset):
        self.dataset = dataset

    def get_monthly_graph(self, column: dataset.id.Column, label: str) -> go.Figure:
        # reshape array into a desired form, and fill missing values by using previous values
        value_by_date = dataframe.pivot(index="date", columns="id", values=values)
        value_by_date.interpolate(method="pad", inplace=True)

        # create a period array in the same form as value_by_date array, and set period to 0,
        #   after asset is sold, then fill missing values in the same fashion as value_by_date array
        period = dataframe.pivot(index="date", columns="id", values="period")
        sold = dataframe.pivot(index="date", columns="id", values="sold").shift(1)
        for c in period.columns:
            period[c] = np.where((sold[c] == True) & np.isnan(period[c]), 0, period[c])
        period.interpolate(method="pad", inplace=True)

        # set value to 0, whereever period is 0
        value_by_date *= period.applymap(lambda p: 1 if p > 0 else 0)

        str_value_by_date = value_by_date.applymap(currency_str)
        value_by_date_sum = value_by_date.sum(axis=1, skipna=True)
        str_value_by_date_sum = value_by_date_sum.apply(currency_str)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=value_by_date_sum.index,
                y=value_by_date_sum.values,
                mode="lines+markers",
                name="Total",
                marker=dict(color=cyan),
                customdata=np.transpose(str_value_by_date_sum.values),
                hovertemplate=(
                    f"%{{x|%B %Y}}<br>"
                    f"<b>Total {label_text.lower()}:</b> %{{customdata}}<extra></extra>"
                ),
            )
        )

        for a in value_by_date.columns:
            if contains_non_zero_values(value_by_date[a]):
                name = asset_properties.loc[a, "name"]
                fig.add_trace(
                    go.Bar(
                        x=value_by_date.index,
                        y=value_by_date[a],
                        name=name,
                        marker=dict(color=asset_properties.loc[a, "color"]),
                        customdata=np.transpose(str_value_by_date[a]),
                        hovertemplate=(
                            f"%{{x|%B %Y}}<br>"
                            f"<b>{name}</b><br>"
                            f"{label_text}: %{{customdata}}<extra></extra>"
                        ),
                    )
                )

        fig.update_layout(barmode="relative")
        six_months = [
            value_by_date.index[-1] - datetime.timedelta(days=(365 / 2 - 15)),
            value_by_date.index[-1] + datetime.timedelta(days=15),
        ]
        fig.update_xaxes(range=six_months)
        fig.update_yaxes(
            ticksuffix=currency_tick_suffix(), tickprefix=currency_tick_prefix()
        )
        configure_historical_dataview(fig, latest_date - value_by_date.index[0])

        return fig
