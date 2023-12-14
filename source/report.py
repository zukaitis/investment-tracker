import datetime
import numpy as np
import pandas as pd

from source import dataset
from source import dataset_identification as id
from source import html
from source import locale
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

        self._title = f"{settings.owner} investment portfolio"
        theme_colors = _colors_dark if settings.theme == "dark" else _colors_light
        self._report = html.Document(title=self._title, css_variables=theme_colors)

        self._append_header()
        # self._append_overall_data_tabs()
        self._append_historical_data_tabs()
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

    def _append_historical_data_tabs(self):
        tabs = []

        groups = self._list_by_value(self._dataset.assets, id.Attribute.GROUP)
        print(groups)
        for group in groups:
            content = ""

            group_assets = self._dataset.assets[
                self._dataset.assets[id.Attribute.GROUP] == group
            ]
            group_accounts = self._list_by_value(group_assets, id.Attribute.ACCOUNT)

            # Display group total if there is more than one account, or only "mixed" account
            if (len(group_accounts) > 1) or (
                (len(group_accounts) == 1)
                and (group_accounts[0] == dataset.unassigned)
                and (len(group_assets) > 1)
            ):
                content += self._create_historical_data_view(
                    group_assets, f"{group} Total"
                )
                content += html.Divider() * 2  # double divider after Total

            print(group_accounts)
            for account in group_accounts:
                account_assets = group_assets[
                    group_assets[id.Attribute.ACCOUNT] == account
                ]
                account_asset_names = self._list_by_value(
                    account_assets, id.Attribute.NAME
                )
                print(account_asset_names)
                for asset in account_asset_names:
                    content += self._create_historical_data_view(
                        account_assets[account_assets[id.Attribute.NAME] == asset]
                    )

            tabs.append(html.Tab(html.Label(group), content))

        if len(groups) > 1:
            content = self._create_historical_data_view(self._dataset.assets)
            tabs.append(html.Tab(html.Label("<i>Total</i>"), content, checked=True))

        self._report.append(html.TabContainer(tabs))

    def _append_footer(self):
        self._report.append(
            f'<p class="footer">Report generated on {self._locale.date_str(datetime.date.today())}, '
            f"using open source script: "
            f'<a href="https://github.com/zukaitis/investment-tracker/">Investment Tracker</a>'
            f"<br>All charts are displayed using "
            f'<a href="https://plotly.com/python/">Plotly</p>'
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
        self, assets: pd.DataFrame, name: str = None
    ) -> str:
        first_row = assets.iloc[0]

        output = self._create_historical_data_view_header(assets, name)
        output += self._create_historical_data_view_statistics(assets)

        return output

        if len(data) > 1:
            yearly_figure = plot_yearly_asset_data(data).to_html(
                full_html=False, include_plotlyjs=True
            )
            historical_figure = plot_historical_asset_data(data).to_html(
                full_html=False, include_plotlyjs=True
            )
            figures = html.Columns(
                [
                    html.Column(width=30, content=yearly_figure),
                    html.Column(content=historical_figure),
                ]
            )
            output += f"{figures}"

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
                if pd.notna(first_row[id.Attribute.INFO])
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
        statistics = []

        if any(historical_data[id.Column.VALUE] != 0):
            statistics.append(
                html.Label(
                    "Value",
                    html.Value(
                        self._locale.currency_str(
                            historical_data.iloc[-1][id.Column.VALUE]
                        )
                    ),
                )
            )
        if (
            (len(assets) == 1)
            and assets.iloc[0][id.Attribute.DISPLAY_PRICE]
            and any(historical_data[id.Column.PRICE] != 0)
        ):
            value_change = self._dataset.get_value_change(historical_data[id.Column.PRICE])
            print(value_change)
            statistics.append(
                html.Label(
                    "Price",
                    html.Value(
                        self._locale.currency_str(
                            historical_data.iloc[-1][id.Column.PRICE]
                        )
                    ),
                )
            )
        # don't display Funds invested, if asset was sold
        if not (
            any(historical_data[id.Column.VALUE] != 0)
            and (historical_data.iloc[-1][id.Column.VALUE] == 0)
        ):
            statistics.append(
                html.Label(
                    "Funds invested",
                    html.Value(
                        self._locale.currency_str(
                            historical_data.iloc[-1][id.Column.NET_INVESTMENT]
                        )
                    ),
                )
            )
        if historical_data.iloc[-1][id.Column.NET_RETURN] != 0:
            statistics.append(
                html.Label(
                    "Return received",
                    html.Value(
                        self._locale.currency_str(
                            historical_data.iloc[-1][id.Column.NET_RETURN]
                        )
                    ),
                )
            )
        statistics.append(
            html.Label(
                "Net profit",
                html.Value(
                    self._locale.currency_str(
                        historical_data.iloc[-1][id.Column.NET_PROFIT]
                    )
                ).color(),
            )
        )
        statistics.append(
            html.Label(
                "Relative net profit",
                html.Value(
                    self._locale.percentage_str(
                        historical_data.iloc[-1][id.Column.RELATIVE_NET_PROFIT]
                    )
                ).color(),
            )
        )

        return f"{html.Columns(statistics)}"
