import datetime
import logging
import numpy as np
import pandas as pd

from source import dataset
from source import dataset_identification as id
from source import graphing
from source import html
from source import locale
from source import log
from source import settings

_colors_light = dict(
    background_color="#e5ecf6",
    text_color="#555555",
    tab_background_color="#ffffff",
    tab_shadow_color="#f0f0f0",
    checked_tab_indicator_color="#00ccee",
    hover_tab_indicator_color="#c6fbff",
)

_colors_dark = dict(
    background_color="#283442",
    text_color="#999999",
    tab_background_color="#111111",
    tab_shadow_color="#202020",
    checked_tab_indicator_color="#00ccee",
    hover_tab_indicator_color="#11363c",
)


class Report:
    def __init__(self, dataset: dataset.Dataset, settings: settings.Settings):
        self._dataset = dataset
        self._settings = settings
        self._locale = locale.Locale(
            locale=self._settings.locale, currency=self._settings.currency
        )
        theme_colors = _colors_dark if settings.theme == "dark" else _colors_light
        self._graphing = graphing.Graphing(self._dataset, self._settings, theme_colors)

        self._title = f"{settings.owner} investment portfolio"
        self._report = html.Document(title=self._title, css_variables=theme_colors)

        self._append_header()
        self._append_overall_data_tabs()
        self._append_historical_data_tabs()
        self._append_log()
        self._append_footer()

    def write_to_file(self, filename: str):
        with open(filename, "w") as f:
            print(self._report, file=f)

    def _append_header(self):
        self._report.append(f"<h1>{self._title}</h1>")
        self._report.append(
            f"<h3>Data from {self._locale.date_str(self._dataset.earliest_date)} to "
            f"{self._locale.date_str(self._dataset.latest_date)}</h3>"
        )
        self._report.append(
            html.Button(
                image_initial="calendar_day.svg",
                image_alternate="calendar_month.svg",
                identifier="value_change_button",
            )
        )

    def _append_overall_data_tabs(self):
        tabs = [self._get_overall_data_tab(id.Column.VALUE, "Value").check()]
        tabs.append(
            self._get_overall_data_tab(id.Column.NET_INVESTMENT, "Funds invested")
        )
        if any(self._dataset.assets[id.Attribute.NET_RETURN] > 0):
            tabs.append(
                self._get_overall_data_tab(id.Column.NET_RETURN, "Return received")
            )
        tabs.append(self._get_overall_data_tab(id.Column.NET_PROFIT, "Net profit"))
        tabs.append(
            self._get_overall_data_tab(
                id.Column.RELATIVE_NET_PROFIT, "Relative net profit"
            )
        )
        self._report.append(html.TabContainer(tabs))

    def _get_overall_data_tab(self, column: id.Column, label_text: str) -> html.Tab:
        historical_data = self._dataset.get_historical_data_sum(self._dataset.assets)
        last_row = historical_data.iloc[-1]
        if column == id.Column.VALUE:
            # Buying/selling assets shouldn't affect value change indicators
            value_change = self._dataset.get_value_change(
                historical_data[id.Column.VALUE]
                - historical_data[id.Column.NET_INVESTMENT]
            )
        else:
            value_change = self._dataset.get_value_change(historical_data[column])

        if column == id.Column.RELATIVE_NET_PROFIT:
            value = html.Value(
                self._locale.percentage_str(last_row[id.Column.RELATIVE_NET_PROFIT]),
                valuechange=html.ValueChange(
                    self._locale.percentage_str(value_change.daily),
                    self._locale.percentage_str(value_change.monthly),
                ),
            ).color()
        else:
            value = html.Value(
                self._locale.currency_str(last_row[column]),
                valuechange=html.ValueChange(
                    self._locale.currency_str(value_change.daily),
                    self._locale.currency_str(value_change.monthly),
                ),
            )
            if column == id.Column.NET_PROFIT:
                value = value.color()

        label = html.Label(label_text, value)
        content = html.Columns(
            [
                html.Column(
                    width=30,
                    content=self._graphing.get_sunburst(
                        attribute=id.get_corresponding_attribute(column),
                        label_text=label_text,
                    ),
                ),
                html.Column(
                    content=self._graphing.get_monthly_graph(
                        column=(
                            id.Column.RETURN
                            if (column == id.Column.NET_RETURN)
                            else column
                        ),
                        label_text=label_text,
                    )
                ),
            ]
        )
        return html.Tab(label=label, content=content)

    def _append_historical_data_tabs(self):
        tabs = []

        groups = self._list_by_value(self._dataset.assets, id.Attribute.GROUP)
        for group in groups:
            content = ""
            dividers_to_add = 0

            group_assets = self._dataset.assets[
                self._dataset.assets[id.Attribute.GROUP] == group
            ]
            group_accounts = self._list_by_value(group_assets, id.Attribute.ACCOUNT)

            # Display group total if there is more than one account, or only "mixed" accounts
            if (len(group_accounts) > 1) or (
                (len(group_accounts) == 1)
                and (group_accounts[0] == dataset.unassigned)
                and (len(group_assets) > 1)
            ):
                open_by_default = any(group_assets[id.Attribute.ACTIVE])
                content += self._create_historical_data_view(
                    assets=group_assets,
                    name=f"{group} Total",
                    open_by_default=open_by_default,
                )
                dividers_to_add = 2  # double divider after Total

            for account in group_accounts:
                content += html.Divider() * dividers_to_add
                account_assets = group_assets[
                    group_assets[id.Attribute.ACCOUNT] == account
                ]
                account_asset_names = self._list_by_value(
                    account_assets, id.Attribute.NAME
                )

                if (account != dataset.unassigned) and (len(account_asset_names) > 1):
                    open_by_default = any(account_assets[id.Attribute.ACTIVE])
                    content += self._create_historical_data_view(
                        assets=account_assets,
                        name=f"{account} Total",
                        open_by_default=open_by_default,
                    )

                for asset in account_asset_names:
                    asset_data = account_assets[
                        account_assets[id.Attribute.NAME] == asset
                    ]
                    open_by_default = any(asset_data[id.Attribute.ACTIVE])
                    content += self._create_historical_data_view(
                        assets=asset_data, open_by_default=open_by_default
                    )
                dividers_to_add = 1

            tabs.append(html.Tab(html.Label(group), content))

        if len(groups) > 1:
            content = self._create_historical_data_view(
                assets=self._dataset.assets, name="Total"
            )
            tabs.append(html.Tab(html.Label("<i>Total</i>"), content, checked=True))

        self._report.append(html.TabContainer(tabs))

    def _append_log(self):
        if len(log.get().records) > 0:
            label = "Log"
            if log.get().error_count > 0:
                label += "&nbsp;&nbsp;" + html.TextBox(
                    f"{log.get().error_count}", "red"
                )
            if log.get().warning_count > 0:
                label += "&nbsp;&nbsp;" + html.TextBox(
                    f"{log.get().warning_count}", "orange"
                )
            content = "<br>".join(log.get().records)
            self._report.append(
                html.Container(html.Accordion(html.Heading2(label), content))
            )

    def _append_footer(self):
        self._report.append(
            f'<p class="footer">Report generated on {self._locale.date_str(datetime.date.today())}, '
            "using open source script: "
            '<a href="https://github.com/zukaitis/investment-tracker/">Investment Tracker</a>'
            "<br>All charts are displayed using "
            '<a href="https://plotly.com/python/">Plotly</p>'
        )

    def _list_by_value(self, assets: pd.DataFrame, group_by: id.Attribute) -> list:
        groups = assets.groupby(group_by)
        groups = groups.agg(
            {id.Attribute.VALUE: "sum", id.Attribute.ACTIVE: "any"}
        ).reset_index()
        groups.sort_values(
            by=[id.Attribute.VALUE, id.Attribute.ACTIVE], inplace=True, ascending=False
        )
        return groups[group_by].unique()

    def _get_successful_fetch_indicator(self, symbol: str) -> str:
        # TODO: Add a tooltip
        return f'<span style="color:green;">{symbol}</span>'

    def _create_historical_data_view(
        self, assets: pd.DataFrame, name: str = None, open_by_default: bool = True
    ) -> str:

        label = self._create_historical_data_view_header(assets, name)
        content = self._create_historical_data_view_statistics(assets)
        content += self._create_historical_data_figures(assets)

        output = html.Accordion(label, content, open=open_by_default)

        return output

    def _create_historical_data_view_header(
        self, assets: pd.DataFrame, name: str = None
    ) -> str:
        first_row = assets.iloc[0]
        title = name if (name is not None) else first_row[id.Attribute.NAME]

        if len(assets) == 1:
            if first_row[id.Attribute.YFINANCE_FETCH_SUCCESSFUL]:
                title += f" - {self._get_successful_fetch_indicator(first_row[id.Attribute.SYMBOL])}"
            elif first_row[id.Attribute.SYMBOL] != dataset.unassigned:
                title += f" - {first_row[id.Attribute.SYMBOL]}"

            if first_row[id.Attribute.ACCOUNT] != dataset.unassigned:
                title += f"<br>{first_row[id.Attribute.ACCOUNT]}"

            info = (
                first_row[id.Attribute.INFO]
                if (first_row[id.Attribute.INFO] != dataset.unassigned)
                else ""
            )
            return html.Columns(
                [
                    html.Column(width=50, content=html.Heading2(title)),
                    html.Column(content=html.Paragraph(info)),
                ]
            )

        return html.Heading2(title)

    def _create_historical_data_view_statistics(self, assets: pd.DataFrame) -> str:
        historical_data = self._dataset.get_historical_data_sum(assets)
        last_row = historical_data.iloc[-1]
        last_nonzero_row = historical_data[historical_data[id.Column.PERIOD] != 0].iloc[
            -1
        ]
        statistics = []

        if any(historical_data[id.Column.VALUE] != 0):
            # Buying/selling assets shouldn't affect value change indicators
            value_change = self._dataset.get_value_change(
                historical_data[id.Column.VALUE]
                - historical_data[id.Column.NET_INVESTMENT]
            )
            statistics.append(
                html.Label(
                    "Value",
                    html.Value(
                        self._locale.currency_str(last_row[id.Column.VALUE]),
                        valuechange=html.ValueChange(
                            self._locale.currency_str(value_change.daily),
                            self._locale.currency_str(value_change.monthly),
                        ),
                    ),
                )
            )

        # don't display Funds invested, if asset was sold
        if not (
            any(historical_data[id.Column.VALUE] != 0)
            and (last_row[id.Column.VALUE] == 0)
        ):
            value_change = self._dataset.get_value_change(
                historical_data[id.Column.NET_INVESTMENT]
            )
            statistics.append(
                html.Label(
                    "Funds invested",
                    html.Value(
                        self._locale.currency_str(last_row[id.Column.NET_INVESTMENT]),
                        valuechange=html.ValueChange(
                            self._locale.currency_str(value_change.daily),
                            self._locale.currency_str(value_change.monthly),
                        ),
                    ),
                )
            )

        if (
            (len(assets) == 1)
            and assets.iloc[0][id.Attribute.DISPLAY_PRICE]
            and any(historical_data[id.Column.PRICE] != 0)
        ):
            value_change = self._dataset.get_value_change(
                historical_data[id.Column.PRICE]
            )
            statistics.append(
                html.Label(
                    "Price",
                    html.Value(
                        self._locale.currency_str(last_row[id.Column.PRICE]),
                        valuechange=html.ValueChange(
                            self._locale.currency_str(value_change.daily),
                            self._locale.currency_str(value_change.monthly),
                        ),
                    ),
                )
            )

        if last_nonzero_row[id.Column.NET_RETURN] != 0:
            value_change = self._dataset.get_value_change(
                historical_data[id.Column.NET_RETURN]
            )
            statistics.append(
                html.Label(
                    "Return received",
                    html.Value(
                        self._locale.currency_str(
                            last_nonzero_row[id.Column.NET_RETURN]
                        ),
                        valuechange=html.ValueChange(
                            self._locale.currency_str(value_change.daily),
                            self._locale.currency_str(value_change.monthly),
                        ),
                    ),
                )
            )

        value_change = self._dataset.get_value_change(
            historical_data[id.Column.NET_PROFIT]
        )
        statistics.append(
            html.Label(
                "Net profit",
                html.Value(
                    self._locale.currency_str(last_nonzero_row[id.Column.NET_PROFIT]),
                    valuechange=html.ValueChange(
                        self._locale.currency_str(value_change.daily),
                        self._locale.currency_str(value_change.monthly),
                    ),
                ).color(),
            )
        )

        value_change = self._dataset.get_value_change(
            historical_data[id.Column.RELATIVE_NET_PROFIT]
        )
        statistics.append(
            html.Label(
                "Relative net profit",
                html.Value(
                    self._locale.percentage_str(
                        last_nonzero_row[id.Column.RELATIVE_NET_PROFIT]
                    ),
                    valuechange=html.ValueChange(
                        self._locale.percentage_str(value_change.daily),
                        self._locale.percentage_str(value_change.monthly),
                    ),
                ).color(),
            )
        )

        return f"{html.Columns(statistics)}"

    def _create_historical_data_figures(self, assets: pd.DataFrame) -> str:
        historical_data = self._dataset.get_historical_data_sum(assets)
        if len(historical_data) > 1:
            yearly_figure = self._graphing.get_yearly_asset_data_plot(
                historical_data
            ).to_html(full_html=False, include_plotlyjs=True)
            historical_figure = self._graphing.get_historical_asset_data_plot(
                historical_data
            ).to_html(full_html=False, include_plotlyjs=True)
            figures = html.Columns(
                [
                    html.Column(width=30, content=yearly_figure),
                    html.Column(content=historical_figure),
                ]
            )
            return f"{figures}"

        return ""
