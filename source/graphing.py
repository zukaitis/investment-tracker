from source import dataset
from source import dataset_identification as id
from source import locale
from source import settings

import datetime
import enum
import plotly.graph_objects as go
import plotly.express as px
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

    def get_sunburst(self, attribute: id.Attribute, label_text: str) -> str:
        class Attribute(enum.Enum):
            LABEL = enum.auto()
            ID = enum.auto()
            PARENT = enum.auto()
            PROFITABILITY = enum.auto()
            PROFITABILITY_SUM = enum.auto()
            GROUP_SUM = enum.auto()
            DISPLAY_STRING = enum.auto()
            DISPLAYED_FRACTION = enum.auto()
            FRACTION_IN_GROUP = enum.auto()
            FRACTION_IN_PROFITABILITY = enum.auto()
            PERCENTAGE_IN_PROFITABILITY_STRING = enum.auto()
            OF_PROFITABILITY_STRING = enum.auto()
            OF_GROUP_STRING = enum.auto()
            SEPARATOR_STRING = enum.auto()

        data = self._dataset.assets[
            self._dataset.assets[attribute] != 0
        ].copy()  # filter 0 values
        data[Attribute.PROFITABILITY] = np.where(
            data[attribute] > 0, "Profitable", "Unprofitable"
        )

        # move name in place of account to avoid empty rings in graph
        data.loc[
            data[id.Attribute.ACCOUNT] == dataset.unassigned,
            [id.Attribute.ACCOUNT, id.Attribute.NAME],
        ] = np.array(
            [
                data.loc[
                    data[id.Attribute.ACCOUNT] == dataset.unassigned, id.Attribute.NAME
                ],
                None,
            ],
            dtype="object",
        )

        data[[Attribute.PROFITABILITY_SUM, Attribute.GROUP_SUM]] = 0.0
        for p in data[Attribute.PROFITABILITY].unique():
            p_subset = data[Attribute.PROFITABILITY] == p
            data.loc[p_subset, Attribute.PROFITABILITY_SUM] = data.loc[
                p_subset, attribute
            ].sum()
            for g in data.loc[p_subset, id.Attribute.GROUP].unique():
                g_subset = p_subset & (data[id.Attribute.GROUP] == g)
                data.loc[g_subset, Attribute.GROUP_SUM] = data.loc[
                    g_subset, attribute
                ].sum()

        graph_data = pd.DataFrame(
            columns=[
                Attribute.LABEL,
                Attribute.ID,
                Attribute.PARENT,
                id.Attribute.VALUE,
                id.Attribute.COLOR,
            ]
        )
        graph_data[[Attribute.LABEL, Attribute.ID, Attribute.PARENT]] = ""
        tree = [
            Attribute.PROFITABILITY,
            id.Attribute.GROUP,
            id.Attribute.ACCOUNT,
            id.Attribute.NAME,
        ]
        if attribute in [
            id.Attribute.VALUE,
            id.Attribute.NET_INVESTMENT,
            id.Attribute.NET_RETURN,
        ]:
            tree.remove(
                Attribute.PROFITABILITY
            )  # don't separate these graphs by profitability

        for t in range(len(tree)):
            d = data.groupby(tree[: t + 1]).first().reset_index()
            d[attribute] = data.groupby(tree[: t + 1]).sum().reset_index()[attribute]
            if attribute == id.Attribute.RELATIVE_NET_PROFIT:
                d[[id.Attribute.NET_PROFIT, id.Attribute.NET_INVESTMENT_MAX]] = (
                    data.groupby(tree[: t + 1])
                    .sum()
                    .reset_index()[
                        [id.Attribute.NET_PROFIT, id.Attribute.NET_INVESTMENT_MAX]
                    ]
                )
                d[Attribute.DISPLAYED_FRACTION] = (
                    d[id.Attribute.NET_PROFIT] / d[id.Attribute.NET_INVESTMENT_MAX]
                )
                d[Attribute.DISPLAY_STRING] = d[Attribute.DISPLAYED_FRACTION].apply(
                    self._locale.percentage_str
                )
            else:
                d[Attribute.FRACTION_IN_GROUP] = d[attribute] / d[Attribute.GROUP_SUM]
                d[Attribute.FRACTION_IN_PROFITABILITY] = (
                    d[attribute] / d[Attribute.PROFITABILITY_SUM]
                )
                d[Attribute.PERCENTAGE_IN_PROFITABILITY_STRING] = np.where(
                    d[Attribute.FRACTION_IN_PROFITABILITY] < 1,
                    "<br>"
                    + d[Attribute.FRACTION_IN_PROFITABILITY].apply(
                        self._locale.percentage_str
                    ),
                    "",
                )
                if (
                    d[Attribute.PROFITABILITY].nunique() > 1
                ):  # if values are separated by profitability
                    d[Attribute.OF_PROFITABILITY_STRING] = np.where(
                        d[Attribute.FRACTION_IN_PROFITABILITY] < 1,
                        " of " + d[Attribute.PROFITABILITY],
                        "",
                    )
                    d[Attribute.SEPARATOR_STRING] = "<br>"
                else:
                    d[Attribute.OF_PROFITABILITY_STRING] = ""
                    d[Attribute.SEPARATOR_STRING] = " / "
                d[Attribute.SEPARATOR_STRING] = np.where(
                    d[Attribute.FRACTION_IN_GROUP] < 1,
                    d[Attribute.SEPARATOR_STRING],
                    "",
                )
                d[Attribute.OF_GROUP_STRING] = np.where(
                    d[Attribute.FRACTION_IN_GROUP] < 1,
                    d[Attribute.FRACTION_IN_GROUP].apply(self._locale.percentage_str)
                    + " of "
                    + d[id.Attribute.GROUP],
                    "",
                )
                # don't display fraction in group, when it matches fraction in profitability
                d.loc[
                    d[Attribute.FRACTION_IN_PROFITABILITY]
                    == d[Attribute.FRACTION_IN_GROUP],
                    [Attribute.SEPARATOR_STRING, Attribute.OF_GROUP_STRING],
                ] = [["", ""]]
                d[Attribute.DISPLAY_STRING] = (
                    d[attribute].apply(self._locale.currency_str)
                    + d[Attribute.PERCENTAGE_IN_PROFITABILITY_STRING]
                    + d[Attribute.OF_PROFITABILITY_STRING]
                    + d[Attribute.SEPARATOR_STRING]
                    + d[Attribute.OF_GROUP_STRING]
                )
            d[attribute] = abs(d[attribute])
            d[Attribute.PARENT] = ""
            for i in range(t):
                d[Attribute.PARENT] += d[tree[i]]
            d[Attribute.ID] = d[Attribute.PARENT] + d[tree[t]]
            d = d[
                [
                    tree[t],
                    Attribute.ID,
                    Attribute.PARENT,
                    attribute,
                    Attribute.DISPLAY_STRING,
                    id.Attribute.COLOR,
                ]
            ]
            d.columns = [
                Attribute.LABEL,
                Attribute.ID,
                Attribute.PARENT,
                id.Attribute.VALUE,
                Attribute.DISPLAY_STRING,
                id.Attribute.COLOR,
            ]
            graph_data = pd.concat([graph_data, d])

        # Set colors of Profitable and Unprofitable labels if they exist
        graph_data.loc[
            graph_data[Attribute.LABEL] == "Profitable", id.Attribute.COLOR
        ] = "green"
        graph_data.loc[
            graph_data[Attribute.LABEL] == "Unprofitable", id.Attribute.COLOR
        ] = "red"

        sunburst = go.Figure(
            go.Sunburst(
                labels=graph_data[Attribute.LABEL],
                ids=graph_data[Attribute.ID],
                parents=graph_data[Attribute.PARENT],
                values=graph_data[id.Attribute.VALUE],
                sort=False,
                customdata=graph_data[Attribute.DISPLAY_STRING],
                marker=dict(colors=graph_data[id.Attribute.COLOR]),
                branchvalues="total",
                hovertemplate=f"<b>%{{label}}</b><br>{label_text}: %{{customdata}}<extra></extra>",
            )
        )
        sunburst.update_traces(insidetextorientation="radial")
        sunburst.update_layout(margin=dict(l=10, r=10, t=10, b=10))

        return sunburst.to_html(full_html=False, include_plotlyjs=True)

    _cyan = px.colors.qualitative.Plotly[5]

    def get_monthly_graph(self, column: id.Column, label_text: str) -> str:
        data = self._dataset.get_monthly_data(column)
        data_str = (
            data.map(self._locale.percentage_str)
            if (column == id.Column.RELATIVE_NET_PROFIT)
            else data.map(self._locale.currency_str)
        )

        data_sum = self._dataset.get_monthly_data_sum(column)
        data_sum_str = (
            data_sum.map(self._locale.percentage_str)
            if (column == id.Column.RELATIVE_NET_PROFIT)
            else data_sum.map(self._locale.currency_str)
        )

        if column == id.Column.RELATIVE_NET_PROFIT:
            data *= 100
            data_sum *= 100

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=data_sum.index,
                y=data_sum.values,
                mode="lines+markers",
                name="Total",
                marker=dict(color=Graphing._cyan),
                customdata=np.transpose(data_sum_str.values),
                hovertemplate=(
                    f"%{{x|%B %Y}}<br>"
                    f"<b>Total {label_text.lower()}:</b> %{{customdata}}<extra></extra>"
                ),
            )
        )

        for a in data.columns:
            if any(data[a] != 0.0):
                name = self._dataset.assets.loc[a, id.Attribute.NAME]
                fig.add_trace(
                    go.Bar(
                        x=data.index,
                        y=data[a],
                        name=name,
                        marker=dict(
                            color=self._dataset.assets.loc[a, id.Attribute.COLOR]
                        ),
                        customdata=np.transpose(data_str[a]),
                        hovertemplate=(
                            f"%{{x|%B %Y}}<br>"
                            f"<b>{name}</b><br>"
                            f"{label_text}: %{{customdata}}<extra></extra>"
                        ),
                    )
                )

        bar_mode = (
            "group"
            if (
                (column == id.Column.RETURN)
                or (column == id.Column.RELATIVE_NET_PROFIT)
            )
            else "relative"
        )
        fig.update_layout(barmode=bar_mode)
        six_months = [
            data.index[-1] - datetime.timedelta(days=(365 / 2 - 15)),
            data.index[-1] + datetime.timedelta(days=15),
        ]
        fig.update_xaxes(range=six_months)
        if column == id.Column.RELATIVE_NET_PROFIT:
            fig.update_yaxes(ticksuffix=self._locale.percentage_tick_suffix())
        else:
            fig.update_yaxes(
                ticksuffix=self._locale.currency_tick_suffix(),
                tickprefix=self._locale.currency_tick_prefix(),
            )
        self._configure_historical_dataview(
            fig, self._dataset.latest_date - data.index[0]
        )

        return fig.to_html(full_html=False, include_plotlyjs=True)

    def get_yearly_asset_data_plot(self, input_data: pd.DataFrame) -> go.Figure:
        class Column(enum.Enum):
            DATE = enum.auto()
            VALUE_CHANGE = enum.auto()
            VALUE_CHANGE_POSITIVE = enum.auto()
            VALUE_CHANGE_NEGATIVE = enum.auto()
            VALUE_CHANGE_POSITIVE_STRING = enum.auto()
            VALUE_CHANGE_NEGATIVE_STRING = enum.auto()
            SAME_PERIOD_AS_LAST_YEAR = enum.auto()
            RELATIVE_VALUE_CHANGE = enum.auto()
            RELATIVE_NET_RETURN = enum.auto()
            NET_RETURN_STRING = enum.auto()
            RELATIVE_NET_SALE_PROFIT = enum.auto()
            NET_SALE_PROFIT_STRING = enum.auto()

        fig = go.Figure()

        # filter out price-only data, where period is 0
        data = input_data[input_data[id.Column.PERIOD] != 0]
        # take latest value of each year, and prepend overall earliest value
        yearly_data = data.groupby(data.index.year).tail(1)
        yearly_data = pd.concat([pd.DataFrame([data.iloc[0]]), yearly_data])

        yearly_data.loc[yearly_data.index[0], id.Column.PERIOD] = (
            -99
        )  # First diff shouldn't be 0
        yearly_data[Column.SAME_PERIOD_AS_LAST_YEAR] = (
            0 == yearly_data[id.Column.PERIOD].diff()
        )

        yearly_data[Column.VALUE_CHANGE] = (
            yearly_data[id.Column.NET_PROFIT]
            - yearly_data[id.Column.NET_RETURN]
            - yearly_data[id.Column.NET_SALE_PROFIT]
        )
        yearly_data[Column.VALUE_CHANGE] = np.where(
            yearly_data[Column.SAME_PERIOD_AS_LAST_YEAR],
            yearly_data[Column.VALUE_CHANGE].diff(),
            yearly_data[Column.VALUE_CHANGE],
        )
        yearly_data[id.Column.NET_RETURN] = np.where(
            yearly_data[Column.SAME_PERIOD_AS_LAST_YEAR],
            yearly_data[id.Column.NET_RETURN].diff(),
            yearly_data[id.Column.NET_RETURN],
        )
        yearly_data[id.Column.NET_SALE_PROFIT] = np.where(
            yearly_data[Column.SAME_PERIOD_AS_LAST_YEAR],
            yearly_data[id.Column.NET_SALE_PROFIT].diff(),
            yearly_data[id.Column.NET_SALE_PROFIT],
        )

        yearly_data[Column.DATE] = yearly_data.index.year
        yearly_data.drop(yearly_data.head(1).index, inplace=True)  # remove first row
        yearly_data.drop_duplicates(subset=[Column.DATE], inplace=True)

        yearly_data[Column.RELATIVE_VALUE_CHANGE] = abs(
            yearly_data[Column.VALUE_CHANGE] / yearly_data[id.Column.NET_INVESTMENT_MAX]
        )
        yearly_data[Column.RELATIVE_NET_RETURN] = (
            yearly_data[id.Column.NET_RETURN]
            / yearly_data[id.Column.NET_INVESTMENT_MAX]
        )
        yearly_data[Column.RELATIVE_NET_SALE_PROFIT] = (
            yearly_data[id.Column.NET_SALE_PROFIT]
            / yearly_data[id.Column.NET_INVESTMENT_MAX]
        )

        yearly_data[Column.VALUE_CHANGE_POSITIVE] = np.where(
            yearly_data[Column.VALUE_CHANGE] > 0, yearly_data[Column.VALUE_CHANGE], 0
        )
        yearly_data[Column.VALUE_CHANGE_NEGATIVE] = np.where(
            yearly_data[Column.VALUE_CHANGE] < 0, yearly_data[Column.VALUE_CHANGE], 0
        )
        yearly_data[Column.NET_RETURN_STRING] = (
            yearly_data[id.Column.NET_RETURN].apply(self._locale.currency_str)
            + " / "
            + yearly_data[Column.RELATIVE_NET_RETURN].apply(self._locale.percentage_str)
        )
        yearly_data[Column.NET_SALE_PROFIT_STRING] = (
            yearly_data[id.Column.NET_SALE_PROFIT].apply(self._locale.currency_str)
            + " / "
            + yearly_data[Column.RELATIVE_NET_SALE_PROFIT].apply(
                self._locale.percentage_str
            )
        )
        yearly_data[Column.VALUE_CHANGE_POSITIVE_STRING] = (
            "+"
            + yearly_data[Column.VALUE_CHANGE_POSITIVE].apply(self._locale.currency_str)
            + " / "
            + yearly_data[Column.RELATIVE_VALUE_CHANGE].apply(
                self._locale.percentage_str
            )
        )

        if any(data[id.Column.VALUE] != 0):
            yearly_data[Column.VALUE_CHANGE_NEGATIVE_STRING] = (
                yearly_data[Column.VALUE_CHANGE_NEGATIVE].apply(
                    self._locale.currency_str
                )
                + " / "
                + yearly_data[Column.RELATIVE_VALUE_CHANGE].apply(
                    self._locale.percentage_str
                )
            )
        else:
            yearly_data[Column.VALUE_CHANGE_NEGATIVE_STRING] = abs(
                yearly_data[Column.VALUE_CHANGE_NEGATIVE]
            ).apply(self._locale.currency_str)

        bar_width = (
            [0.5] if (len(yearly_data) == 1) else None
        )  # single column looks ugly otherwise

        fig.add_trace(
            go.Bar(
                x=yearly_data[Column.DATE],
                y=yearly_data[id.Column.NET_RETURN],
                marker=dict(color="rgb(73,200,22)"),
                width=bar_width,
                customdata=np.transpose(yearly_data[Column.NET_RETURN_STRING]),
                hovertemplate=(
                    f"<b>%{{x}}</b><br>"
                    f"Return received:<br>%{{customdata}}<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Bar(
                x=yearly_data[Column.DATE],
                y=yearly_data[id.Column.NET_SALE_PROFIT],
                marker=dict(color="rgb(73,200,22)"),
                width=bar_width,
                customdata=np.transpose(yearly_data[Column.NET_SALE_PROFIT_STRING]),
                hovertemplate=(
                    f"<b>%{{x}}</b><br>"
                    f"Sale Profit:<br>%{{customdata}}<extra></extra>"
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
                x=yearly_data[Column.DATE],
                y=yearly_data[Column.VALUE_CHANGE_POSITIVE],
                marker=dict(color="rgba(0, 255, 0, 0.7)"),
                width=bar_width,
                hovertemplate=hovertemplate,
                customdata=np.transpose(
                    yearly_data[Column.VALUE_CHANGE_POSITIVE_STRING]
                ),
            )
        )
        fig.add_trace(
            go.Bar(
                x=yearly_data[Column.DATE],
                y=yearly_data[Column.VALUE_CHANGE_NEGATIVE],
                marker=dict(color="rgba(255, 0, 0, 0.7)"),
                width=bar_width,
                hovertemplate=hovertemplate,
                customdata=np.transpose(
                    yearly_data[Column.VALUE_CHANGE_NEGATIVE_STRING]
                ),
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
                pdata[id.Column.NET_INVESTMENT]
                > (pdata[id.Column.NET_RETURN] + pdata[id.Column.NET_SALE_PROFIT]),
                pdata[id.Column.NET_INVESTMENT],
                pdata[id.Column.NET_RETURN] + pdata[id.Column.NET_SALE_PROFIT],
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
                        y=pdata[id.Column.NET_RETURN]
                        + pdata[id.Column.NET_SALE_PROFIT],
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
                pd.concat([blue_fill, intermediate_values])
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
