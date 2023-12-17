from source import dataset
from source import dataset_identification as id
from source import locale
from source import settings

import datetime
import plotly.graph_objects as go
import numpy as np
import pandas as pd


class Graphing:
    def __init__(
        self, dataset: dataset.Dataset, settings: settings.Settings, theme_colors: dict
    ):
        self._dataset = dataset
        self._settings = settings
        self._locale = locale.Locale(
            locale=self._settings.locale, currency=self._settings.currency
        )
        self._theme_colors = theme_colors

    def get_monthly_graph(self, column: id.Column, label: str) -> go.Figure:
        # reshape array into a desired form, and fill missing values by using previous values
        value_by_date = dataframe.pivot(index="date", columns="id", values=values)
        value_by_date.interpolate(method="pad", inplace=True)

        # create a period array in the same form as value_by_date array, and set period to 0,
        #   after asset is sold, then fill missing values in the same fashion as value_by_date array
        period = dataframe.pivot(index="date", columns="id", values=id.Column.PERIOD)
        sold = dataframe.pivot(index="date", columns="id", values="sold").shift(1)
        for c in period.columns:
            period[c] = np.where((sold[c] == True) & np.isnan(period[c]), 0, period[c])
        period.interpolate(method="pad", inplace=True)

        # set value to 0, whereever period is 0
        value_by_date *= period.applymap(lambda p: 1 if p > 0 else 0)

        str_value_by_date = value_by_date.applymap(self._locale.currency_str)
        value_by_date_sum = value_by_date.sum(axis=1, skipna=True)
        str_value_by_date_sum = value_by_date_sum.apply(self._locale.currency_str)

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
            ticksuffix=self._locale.currency_tick_suffix(),
            tickprefix=self._locale.currency_tick_prefix(),
        )
        configure_historical_dataview(fig, latest_date - value_by_date.index[0])

        return fig

    def get_yearly_asset_data_plot(self, data: pd.DataFrame) -> go.Figure:
        fig = go.Figure()

        # filter price-only data, where period is 0
        # take earliest value of each year, and append overall latest value
        #yearly_data = data[data[id.Column.PERIOD] != 0].groupby(data.index.year).head(1)
        yearly_data = data.groupby(data.index.year).head(1)
        yearly_data = yearly_data._append(data.iloc[-1])

        yearly_data["value_change"] = (
            yearly_data[id.Column.NET_PROFIT] - yearly_data[id.Column.NET_RETURN]
        )
        yearly_data.loc[yearly_data.index[0], "value_change"] = 0
        yearly_data["value_change"] = yearly_data["value_change"].diff()
        yearly_data.loc[yearly_data.index[0], "total_return_received"] = 0
        yearly_data["total_return_received"] = yearly_data[
            "total_return_received"
        ].diff()

        # subtract one year from each date, since these lines are going to represent value change,
        # which occured during previous year
        yearly_data["date"] = yearly_data.index.year - 1
        if yearly_data.index[-1] != yearly_data.index[-2]:
            yearly_data.loc[
                yearly_data.index[-1], "date"
            ] += 1  # set back year of last row
        yearly_data.drop(yearly_data.head(1).index, inplace=True)  # remove first row
        yearly_data.drop_duplicates(subset=["date"], inplace=True)

        yearly_data["relative_value_change"] = abs(
            yearly_data["value_change"] / yearly_data[id.Column.NET_INVESTMENT_MAX]
        )
        yearly_data["relative_return_received"] = (
            yearly_data["total_return_received"]
            / yearly_data[id.Column.NET_INVESTMENT_MAX]
        )

        yearly_data["value_change_positive"] = np.where(
            yearly_data["value_change"] > 0, yearly_data["value_change"], 0
        )
        yearly_data["value_change_negative"] = np.where(
            yearly_data["value_change"] < 0, yearly_data["value_change"], 0
        )
        yearly_data["str_total_return_received"] = (
            yearly_data["total_return_received"].apply(self._locale.currency_str)
            + " / "
            + yearly_data["relative_return_received"].apply(self._locale.percentage_str)
        )
        yearly_data["str_value_change_positive"] = (
            "+"
            + yearly_data["value_change_positive"].apply(self._locale.currency_str)
            + " / "
            + yearly_data["relative_value_change"].apply(self._locale.percentage_str)
        )

        if any(data[id.Column.VALUE] != 0):
            yearly_data["str_value_change_negative"] = (
                yearly_data["value_change_negative"].apply(self._locale.currency_str)
                + " / "
                + yearly_data["relative_value_change"].apply(
                    self._locale.percentage_str
                )
            )
        else:
            yearly_data["str_value_change_negative"] = abs(
                yearly_data["value_change_negative"]
            ).apply(self._locale.currency_str)

        bar_width = (
            [0.5] if (len(yearly_data) == 1) else None
        )  # single column looks ugly otherwise

        fig.add_trace(
            go.Bar(
                x=yearly_data["date"],
                y=yearly_data["total_return_received"],
                marker=dict(color="rgb(73,200,22)"),
                width=bar_width,
                customdata=np.transpose(yearly_data["str_total_return_received"]),
                hovertemplate=(
                    f"<b>%{{x}}</b><br>"
                    f"Return received:<br>%{{customdata}}<extra></extra>"
                ),
            )
        )

        if any(data[id.Column.VALUE] != 0):
            hovertemplate = (
                f"<b>%{{x}}</b><br>" f"Value change:<br>%{{customdata}}<extra></extra>"
            )
        else:
            hovertemplate = (
                f"<b>%{{x}}</b><br>"
                f"Funds invested:<br>%{{customdata}}<extra></extra>"
            )

        fig.add_trace(
            go.Bar(
                x=yearly_data["date"],
                y=yearly_data["value_change_positive"],
                marker=dict(color="rgba(0, 255, 0, 0.7)"),
                width=bar_width,
                hovertemplate=hovertemplate,
                customdata=np.transpose(yearly_data["str_value_change_positive"]),
            )
        )
        fig.add_trace(
            go.Bar(
                x=yearly_data["date"],
                y=yearly_data["value_change_negative"],
                marker=dict(color="rgba(255, 0, 0, 0.7)"),
                width=bar_width,
                hovertemplate=hovertemplate,
                customdata=np.transpose(yearly_data["str_value_change_negative"]),
            )
        )

        fig.update_layout(barmode="relative", showlegend=False)
        fig.update_xaxes(type="category", fixedrange=True)
        fig.update_yaxes(
            ticksuffix=self._locale.currency_tick_suffix(),
            tickprefix=self._locale.currency_tick_prefix(),
            fixedrange=True,
        )
        fig.update_layout(margin=dict(l=10, r=10, t=35, b=10))

        return fig

    def get_historical_asset_data_plot(self, input: pd.DataFrame) -> go.Figure:
        fig = go.Figure()

        data = input.copy()
        data["value_and_return"] = (
            data[id.Column.NET_INVESTMENT] + data[id.Column.NET_PROFIT]
        )

        data["str_net_investment"] = data[id.Column.NET_INVESTMENT].apply(
            self._locale.currency_str
        )
        data["str_return_received"] = data[id.Column.NET_RETURN].apply(
            self._locale.currency_str
        )
        data["str_net_sale_profit"] = data[id.Column.NET_SALE_PROFIT].apply(
            self._locale.currency_str
        )
        data["str_value"] = data[id.Column.VALUE].apply(self._locale.currency_str)
        if any(data[id.Column.AMOUNT] != 0):
            data["str_net_investment"] += "<br><b>Amount:</b> " + data[
                id.Column.AMOUNT
            ].apply(self._locale.decimal_str)

        data["f_profit"] = (
            data[id.Column.NET_PROFIT].apply(self._locale.currency_str) + " / "
        )
        data["f_relative_profit"] = data[id.Column.RELATIVE_NET_PROFIT].apply(
            self._locale.percentage_str
        )
        data["profit_string"] = data["f_profit"] + data["f_relative_profit"]

        for p in input[id.Column.PERIOD].dropna().unique():
            if p == 0:
                continue
            pdata = data[data[id.Column.PERIOD] == p].copy()

            fig.add_trace(
                go.Scatter(
                    x=pdata.index,
                    y=pdata[id.Column.NET_INVESTMENT],
                    mode="none",
                    customdata=pdata["str_net_investment"],
                    showlegend=False,
                    hovertemplate=f"<b>Net investment:</b> %{{customdata}}<extra></extra>",
                )
            )

            pdata["red_fill"] = np.where(
                pdata[id.Column.NET_INVESTMENT] > pdata[id.Column.NET_RETURN],
                pdata[id.Column.NET_INVESTMENT],
                pdata[id.Column.NET_RETURN],
            )
            fig.add_trace(
                go.Scatter(
                    x=pdata.index,
                    y=pdata["red_fill"],
                    fill="tozeroy",
                    mode="none",
                    fillcolor="rgba(255,0,0,0.7)",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=pdata.index,
                    y=pdata["value_and_return"],
                    fill="tozeroy",
                    mode="none",
                    fillcolor="rgba(0,255,0,0.7)",
                    customdata=pdata["profit_string"],
                    hovertemplate=f"<b>Net profit:</b> %{{customdata}}<extra></extra>",
                    showlegend=False,
                )
            )

            blue_fill_mode = "tozeroy"
            if max(pdata[id.Column.NET_RETURN]) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=pdata.index,
                        y=pdata[id.Column.NET_RETURN],
                        fill="tozeroy",
                        mode="none",
                        fillcolor="rgba(0,0,0,0)",
                        customdata=pdata["str_return_received"],
                        showlegend=False,
                        hovertemplate=f"<b>Return received:</b> %{{customdata}}<extra></extra>",
                    )
                )
                blue_fill_mode = "tonexty"
            if max(pdata[id.Column.NET_SALE_PROFIT]) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=pdata.index,
                        y=pdata[id.Column.NET_RETURN]+pdata[id.Column.NET_SALE_PROFIT],
                        fill="tonexty",
                        mode="none",
                        fillcolor="rgba(0,0,0,0)",
                        customdata=pdata["str_net_sale_profit"],
                        showlegend=False,
                        hovertemplate=f"<b>Sale profit:</b> %{{customdata}}<extra></extra>",
                    )
                )
                blue_fill_mode = "tonexty"

            blue_fill = pdata[["red_fill", "value_and_return"]].copy()
            blue_fill["date"] = blue_fill.index
            blue_fill = blue_fill.reset_index()
            blue_fill.index *= 2
            blue_fill["y"] = np.where(
                blue_fill["red_fill"] < blue_fill["value_and_return"],
                blue_fill["red_fill"],
                pdata["value_and_return"],
            )
            blue_fill["profitable"] = blue_fill["y"] == blue_fill["red_fill"]
            mask = blue_fill.iloc[:-1]["profitable"] ^ blue_fill["profitable"].shift(-1)
            intermediate_values = blue_fill[mask].copy()
            intermediate_values.index += 1
            intermediate_values["y"] = np.nan
            blue_fill = (
                blue_fill._append(intermediate_values)
                .sort_index()
                .reset_index(drop=True)
            )
            blue_fill["slope"] = abs(
                blue_fill["value_and_return"] - blue_fill["red_fill"]
            ) / (
                abs(blue_fill["value_and_return"] - blue_fill["red_fill"])
                + abs(
                    blue_fill.shift(-1)["value_and_return"]
                    - blue_fill.shift(-1)["red_fill"]
                )
            )
            blue_fill["date"] = np.where(
                pd.isna(blue_fill["y"]),
                (
                    (blue_fill.shift(-1)["date"] - blue_fill["date"])
                    * blue_fill["slope"]
                    + blue_fill["date"]
                ),
                blue_fill["date"],
            )
            blue_fill["y"] = np.where(
                pd.isna(blue_fill["y"]),
                (
                    (blue_fill.shift(-1)["red_fill"] - blue_fill["red_fill"])
                    * blue_fill["slope"]
                    + blue_fill["red_fill"]
                ),
                blue_fill["y"],
            )
            fig.add_trace(
                go.Scatter(
                    x=blue_fill["date"],
                    y=blue_fill["y"],
                    fill=blue_fill_mode,
                    mode="none",
                    fillcolor="rgba(0,0,255,0.5)",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

            if max(pdata[id.Column.VALUE]) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=pdata.index,
                        y=pdata[id.Column.VALUE],
                        mode="lines",
                        line=dict(color="yellow"),
                        customdata=pdata["str_value"],
                        hovertemplate=f"<b>Value:</b> %{{customdata}}<extra></extra>",
                        showlegend=False,
                    )
                )

        fig.update_layout(
            hovermode="x",
            showlegend=True,
            legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1.04),
        )
        fig.update_layout(
            hoverlabel=dict(
                bgcolor=self._theme_colors["tab_background_color"],
                font=dict(color=self._theme_colors["text_color"]),
            )
        )
        frequency = Graphing._calculate_frequency(data.index)
        earliest_entry_date = input.index[0]
        latest_entry_date = input.index[-1]
        self._configure_historical_dataview(
            fig, self._dataset.latest_date - earliest_entry_date, frequency
        )
        span = pd.DateOffset(years=1)
        if (frequency in ["H", "BH"]) and (  # hourly data
            self._dataset.latest_date - latest_entry_date < datetime.timedelta(days=3)
        ):
            span = pd.DateOffset(days=5)
        elif (frequency in ["H", "BH", "B", "C", "D"]) and (  # daily data
            self._dataset.latest_date - latest_entry_date < datetime.timedelta(days=14)
        ):
            span = pd.DateOffset(months=1)
        range = [self._dataset.latest_date - span, self._dataset.latest_date]
        fig.update_xaxes(range=range, rangeslider=dict(visible=True))
        max_value = max(
            max(data[id.Column.NET_INVESTMENT]), max(data["value_and_return"])
        )
        fig.update_yaxes(
            ticksuffix=self._locale.currency_tick_suffix(),
            tickprefix=self._locale.currency_tick_prefix(),
            range=[0, max_value * 1.05],
        )

        if any(data[id.Column.PRICE] != 0):  # check if price data is present
            data["price_string"] = data[id.Column.PRICE].apply(
                self._locale.currency_str
            )
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[id.Column.PRICE],
                    mode="lines",
                    name="Price",
                    marker=dict(color="cyan"),
                    yaxis="y2",
                    customdata=data["price_string"],
                    visible="legendonly",
                    hovertemplate=f"<b>Price:</b> %{{customdata}}<extra></extra>",
                )
            )
            margin = (max(data[id.Column.PRICE]) - min(data[id.Column.PRICE])) * 0.05
            fig.update_layout(
                yaxis2=dict(
                    title="",
                    title_standoff=0,
                    titlefont=dict(color="cyan"),
                    tickfont=dict(color="cyan"),
                    overlaying="y",
                    ticksuffix=self._locale.currency_tick_suffix(),
                    tickprefix=self._locale.currency_tick_prefix(),
                    side="right",
                    range=[
                        min(data[id.Column.PRICE]) - margin,
                        max(data[id.Column.PRICE]) + margin,
                    ],
                )
            )
            fig.update_layout(margin=dict(r=2))

        comments = data[pd.notnull(data[id.Column.COMMENT])]

        fig.add_trace(
            go.Scatter(
                x=comments.index,
                y=[max_value * 0.05]
                * len(comments),  # display comment mark at 5% of max y value
                mode="markers",
                marker=dict(
                    line=dict(width=2, color="purple"), size=12, symbol="asterisk"
                ),
                customdata=comments[id.Column.COMMENT],
                hovertemplate=f"<b>*</b> %{{customdata}}<extra></extra>",
                showlegend=False,
            )
        )

        return fig

    @classmethod
    def _calculate_frequency(cls, dates: pd.Series) -> str:
        if len(dates) >= 4:  # calculate frequency from three dates before the last
            return pd.infer_freq(dates[-4:-1])
        if len(dates) >= 2:
            period = dates[-1] - dates[-2]
            if period < datetime.timedelta(hours=2):
                return "H"  # hourly
            elif period < datetime.timedelta(days=2):
                return "D"  # daily

    def _configure_historical_dataview(
        self, figure: go.Figure, timerange: datetime.timedelta, frequency: str = None
    ):
        buttons = []

        if frequency in ["H", "BH"]:  # hourly data
            buttons += [
                dict(count=1, label="1D", step="day", stepmode="backward"),
                dict(count=5, label="5D", step="day", stepmode="backward"),
            ]

        if frequency in ["H", "BH", "B", "C", "D"]:  # daily data
            buttons += [
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
            ]

        # buttons, that are always present
        buttons += [
            dict(count=6, label="6M", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1Y", step="year", stepmode="backward"),
        ]

        # adding buttons depending on input time range
        years = [1, 2, 5]
        for i in range(1, len(years)):
            if timerange.days > years[i - 1] * 365:
                buttons += [
                    dict(
                        count=years[i],
                        label=f"{years[i]}Y",
                        step="year",
                        stepmode="backward",
                    )
                ]

        buttons += [dict(label="ALL", step="all")]  # ALL is also always available

        figure.update_xaxes(
            type="date",
            rangeslider_visible=False,
            rangeselector=dict(
                x=0,
                buttons=buttons,
                font=dict(color=self._theme_colors["text_color"]),
                bgcolor=self._theme_colors["background_color"],
                activecolor=self._theme_colors["hover_tab_indicator_color"],
            ),
        )
        figure.update_layout(margin=dict(l=10, r=10, t=20, b=10))
        figure.update_layout(
            xaxis=dict(title=dict(text="")), yaxis=dict(title=dict(text=""))
        )
