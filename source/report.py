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
        for group in groups:
            content = ""

            group_assets = self._dataset.assets[
                self._dataset.assets[id.Attribute.GROUP] == group
            ]
            group_accounts = self._list_by_value(group_assets, id.Attribute.ACCOUNT)

            # Display group total if there is more than one account, or only "mixed" account
            if (len(group_accounts) > 1) or (
                (len(group_accounts) == 1) and (group_accounts[0] == np.nan)
            ):
                content += self._create_historical_data_view(
                    group_assets, f"{group} Total"
                )
                content += html.Divider() * 2  # double divider after Total

            for account in group_accounts:
                account_assets = group_assets[
                    group_assets[id.Attribute.ACCOUNT] == account
                ]
                account_asset_names = self._list_by_value(
                    account_assets, id.Attribute.NAME
                )
                for asset in account_asset_names:
                    content += self._create_historical_data_view(
                        account_assets[account_assets[id.Attribute.NAME] == asset]
                    )

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

    def _create_historical_data_view(self, assets: pd.DataFrame, name: str = None):
        first_row = data.iloc[0]
        title = first_row[id.Attribute.NAME]
        if name is not None:
            title = name

        if len(assets) == 1:
            if ("symbol" in first_row) and (pd.notnull(first_row["symbol"])):
                title += f" ({first_row['symbol']})"
            if ("account" in first_row) and (pd.notnull(first_row["account"])):
                title += f"<br>{first_row['account']}"

            if ("info" not in first_row) or (type(first_row["info"]) is not str):
                first_row["info"] = ""
        output = html.Columns(
            [
                html.Column(width=50, content=html.Heading2(title)),
                html.Column(content=html.Paragraph(last_row["info"])),
            ]
        )
        statistics = []

        if contains_non_zero_values(data["value"]):
            data["value_change"] = data["profit"] - data["return_received"]
            c = calculate_value_change(
                data.set_index("date")["value_change"], iscurrency=True
            )
            statistics.append(
                html.Label(
                    "Value", html.Value(currency_str(last_row["value"]), valuechange=c)
                )
            )

        # don't display Funds invested, if asset was sold
        if not ((contains_non_zero_values(data["value"])) and (last_row["value"] == 0)):
            c = calculate_value_change(
                data.set_index("date")["net_investment"], iscurrency=True
            )
            statistics.append(
                html.Label(
                    "Funds invested",
                    html.Value(currency_str(last_row["net_investment"]), valuechange=c),
                )
            )

        last_row["return_received"] = round(last_row["return_received"], 2)
        if last_row["return_received"] != 0:
            c = calculate_value_change(
                data.set_index("date")["return_received"], iscurrency=True
            )
            statistics.append(
                html.Label(
                    "Return received",
                    html.Value(
                        currency_str(last_row["return_received"]), valuechange=c
                    ),
                )
            )

        c = calculate_value_change(data.set_index("date")["profit"], iscurrency=True)
        statistics.append(
            html.Label(
                "Net profit",
                html.Value(currency_str(last_row["profit"]), valuechange=c).color(),
            )
        )

        c = calculate_value_change(
            data.set_index("date")["relative_profit"], ispercentage=True
        )
        statistics.append(
            html.Label(
                "Relative net profit",
                html.Value(
                    percentage_str(last_row["relative_profit"]), valuechange=c
                ).color(),
            )
        )

        output += f"{html.Columns(statistics)}"

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
