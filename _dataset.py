import datetime
import enum
import dataclasses
import numpy as np
import pandas as pd
import plotly.express as px
import typing

import _dataset_identification as id
import _report as report
import _settings as settings
import _yfinance_wrapper as yfinance_wrapper


class Dataset:
    def __init__(self, settings: settings.Settings):
        self._settings = settings
        self._yfinance = yfinance_wrapper.YfinanceWrapper(self._settings)
        self._assets = pd.DataFrame(columns=list(id.Attribute))
        self._attribute_data_calculated = False
        self.historical_data = {}
        self.latest_date = None
        self.earliest_date = None

    @property
    def assets(self) -> pd.DataFrame:
        if not self._attribute_data_calculated:
            self._calculate_attribute_data()
            self._attribute_data_calculated = True
        return self._assets

    def append(self, filedict: dict):
        self._append_asset_assets(filedict)

        last_id = self._assets.index[-1]
        # try:
        self._append_historical_data(filedict["data"], identifier=last_id)
        # except Exception as error:
        #     # remove attribute if something went wrong
        #     self._assets.drop(last_id, inplace=True)
        #     raise ValueError(error) from error

    def get_historical_data_sum(
        self, assets: typing.Union[list, pd.DataFrame]
    ) -> pd.DataFrame:
        if isinstance(assets, pd.DataFrame):
            assets = list(assets.index)
        if not isinstance(assets, list):
            raise TypeError("Parameter should be either a list or DataFrame")

        all_dates = pd.Index([])
        for a in assets:
            all_dates = (
                all_dates.append(self.historical_data[a].index).unique().sort_values()
            )

        result = pd.DataFrame(columns=id.Column, index=all_dates)
        result[[id.Column.VALUE, id.Column.INVESTMENT, id.Column.RETURN]] = 0
        for a in assets:
            historical = self.historical_data[a].reindex(all_dates)
            historical[[id.Column.PRICE, id.Column.AMOUNT, id.Column.COMMENT]] = np.nan
            historical = self._interpolate_historical_data(historical)
            result = result.add(historical)

        return self._interpolate_historical_data(result)

    def get_monthly_data(self, column: id.Column) -> pd.DataFrame:
        print("yo")

    def _calculate_attribute_data(self):
        self.latest_date = max([max(a.index) for _, a in self.historical_data.items()])
        self.earliest_date = min(
            [min(a.index) for _, a in self.historical_data.items()]
        )

        for identifier in self._assets.index:
            self._assets.at[
                identifier, id.Attribute.CURRENT_VALUE
            ] = self.historical_data[identifier][id.Column.VALUE].iloc[-1]

            if any(self.historical_data[identifier][id.Column.VALUE] > 0):
                self._assets.at[identifier, id.Attribute.IS_RELEVANT] = (
                    self._assets.at[identifier, id.Attribute.CURRENT_VALUE] != 0
                )
            else:
                # taxes are considered irrelevant,
                # if they weren't updated for a certain amount of time
                self._assets.at[identifier, id.Attribute.IS_RELEVANT] = max(
                    self.historical_data[identifier].index
                ) < (
                    self.latest_date
                    - pd.tseries.frequencies.to_offset(self._settings.relevance_period)
                )

        self._assets[id.Attribute.GROUP].fillna("Ungrouped", inplace=True)
        self._reassign_colors()

    def _reassign_colors(self):
        self._assets.sort_values(
            by=[id.Attribute.CURRENT_VALUE, id.Attribute.IS_RELEVANT],
            inplace=True,
            ascending=False,
        )

        bright_colors = px.colors.qualitative.Set1
        pastel_colors = px.colors.qualitative.Pastel1
        groups = self._assets.groupby(id.Attribute.GROUP).agg(
            {id.Attribute.CURRENT_VALUE: "sum"}
        )
        groups.sort_values(by=id.Attribute.CURRENT_VALUE, ascending=False, inplace=True)
        if len(groups) > len(bright_colors):
            report.error(
                "Input data contains too many different groups. "
                "Because of that, group colors will be reused"
            )

        color_index = 0
        for group in groups.index:
            asset_count = len(self._assets[self._assets[id.Attribute.GROUP] == group])
            bright_color = bright_colors[color_index]
            pastel_color = pastel_colors[color_index]
            colors = px.colors.n_colors(
                bright_color, pastel_color, max(asset_count, 4), colortype="rgb"
            )
            colors = colors[:asset_count]
            self._assets.loc[
                self._assets[id.Attribute.GROUP] == group, id.Attribute.COLOR
            ] = colors
            color_index += 1
            if color_index == len(bright_colors):
                color_index = 0

    def _append_asset_assets(self, filedict: dict):
        expected_assets = {
            id.Attribute.NAME: "name",
            id.Attribute.SYMBOL: "symbol",
            id.Attribute.GROUP: "group",
            id.Attribute.ACCOUNT: "account",
            id.Attribute.INFO: "info",
            id.Attribute.FILENAME: "filename",
        }

        for a in expected_assets.values():
            if (a in filedict) and (not isinstance(filedict[a], str)):
                report.warn(
                    f'Attribute "{a}" is of wrong type. Attribute type should be string'
                )
                filedict.pop(a)

        identifier = ">".join(
            [
                filedict[a]
                for a in ["name", "symbol", "account", "group"]
                if a in filedict
            ]
        )
        new_entry = pd.DataFrame(
            {
                key: filedict[val]
                for key, val in expected_assets.items()
                if val in filedict
            },
            index=[identifier],
        )
        try:
            # check for duplicate IDs is enabled with verify_integrity
            self._assets = pd.concat([self._assets, new_entry], verify_integrity=True)
        except ValueError as error:
            raise ValueError(
                "Identical asset _assets found in files "
                f"{report.italic(self._assets.at[identifier, id.Attribute.FILENAME])} and "
                f'{report.italic(filedict["filename"])}. Data from latter file is ignored'
            ) from error

    def _append_historical_data(self, data: list, identifier: str):
        if not isinstance(data, list):
            raise ValueError("Data should be structured as a list")

        new_entry = pd.DataFrame(data)

        if "date" not in new_entry.columns:
            raise ValueError("Not a single date found in file")

        new_entry = self._convert_historical_data(new_entry)
        if len(new_entry) == 0:
            raise ValueError("No data left in file after filtering")

        symbol = self._assets.at[identifier, id.Attribute.SYMBOL]
        if (
            symbol is not np.nan
            and id.Column.VALUE not in new_entry.columns
            and id.Column.PRICE not in new_entry.columns
        ):
            report.report(f"Fetching yfinance data for {report.italic(symbol)}")
            yfdata = self._yfinance.get_historical_data(symbol, min(new_entry.index))
            self._assets.at[identifier, id.Attribute.YFINANCE_FETCH_SUCCESSFUL] = True
            new_entry = pd.concat([new_entry, yfdata], axis=1)

        new_entry = self._interpolate_historical_data(new_entry)
        self.historical_data.update({identifier: new_entry})

    def _contains_non_zero_values(self, column: pd.Series) -> bool:
        return any((column.values != 0) & (pd.notna(column.values)))

    def _convert_historical_data(self, input_data: pd.DataFrame) -> pd.DataFrame:
        @dataclasses.dataclass()
        class InputColumn:
            name: str
            non_negative: bool = False
            max_value: float = None

        expected_columns = {
            id.Column.INVESTMENT: InputColumn("investment"),
            id.Column.RETURN: InputColumn("return", non_negative=True),
            id.Column.RETURN_TAX: InputColumn(
                "return_tax", non_negative=True, max_value=1.0
            ),
            id.Column.PRICE: InputColumn("price", non_negative=True),
            id.Column.AMOUNT: InputColumn("amount", non_negative=True),
            id.Column.VALUE: InputColumn("value", non_negative=True),
        }

        data = input_data.copy()
        data.rename(
            columns={
                val.name: key
                for key, val in expected_columns.items()
                if val.name in data
            },
            inplace=True,
        )
        data.rename(columns={"date": id.Index.DATE}, inplace=True)
        if "comment" in data.columns:
            data.rename(columns={"comment": id.Column.COMMENT}, inplace=True)

        data = data[data[id.Index.DATE].notnull()]  # remove rows without a date
        data[id.Index.DATE] = pd.to_datetime(data[id.Index.DATE], errors="coerce")
        if any(data[id.Index.DATE].isnull()):
            report.warn("Unrecognized dates found in data")
            data = data[
                data[id.Index.DATE].notnull()
            ]  # remove rows where date was not converted
        if any(data[id.Index.DATE] > datetime.datetime.now()):
            report.warn("Future dates found in data, they will be ignored")
            data = data[data[id.Index.DATE] <= datetime.datetime.now()]

        # filter out unrecognized, negative, and too high values
        available_columns = [c for c in expected_columns if c in data.columns]
        original_data = data.copy()
        data[available_columns] = data[available_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        for col in available_columns:
            column_name_italic = report.italic(expected_columns[col].name)
            unrecognized = data[col].isnull() & original_data[col].notnull()
            if any(unrecognized):
                report.warn(f"Unrecognized data found in {column_name_italic} column")
                data = data[~unrecognized]
                original_data = original_data[~unrecognized]
            if expected_columns[col].non_negative:
                negative = data[col] < 0
                if any(negative):
                    report.warn(f"Negative values found in {column_name_italic} column")
                    data = data[~negative]
                    original_data = original_data[~negative]
            if expected_columns[col].max_value is not None:
                too_high = data[col] > expected_columns[col].max_value
                if any(too_high):
                    report.warn(
                        f"Too high values found in {column_name_italic} column. "
                        "Values in this column should not exceed "
                        f"{expected_columns[col].max_value}"
                    )
                    data = data[~too_high]
                    original_data = original_data[~too_high]

        data = data.set_index(id.Index.DATE).sort_index()
        return data

    def _interpolate_historical_data(self, input_data: pd.DataFrame) -> pd.DataFrame:
        interpolated_columns = {
            id.Column.AMOUNT: "pad",
            id.Column.PRICE: "pad",
            id.Column.INVESTMENT: None,
            id.Column.RETURN: None,
            id.Column.RETURN_TAX: "pad",
        }

        data = input_data.copy()
        for col, method in interpolated_columns.items():
            if col not in data.columns:
                data[col] = np.nan
            if method is not None:
                data[col] = data[col].interpolate(method=method)
            data[col] = data[col].fillna(0.0)

        if id.Column.VALUE not in data.columns:
            data[id.Column.VALUE] = np.nan
        if self._contains_non_zero_values(
            data[id.Column.AMOUNT]
        ) and self._contains_non_zero_values(data[id.Column.PRICE]):
            data.loc[pd.isna(data[id.Column.VALUE]), id.Column.VALUE] = (
                data[id.Column.AMOUNT] * data[id.Column.PRICE]
            )
        data[id.Column.VALUE] = data[id.Column.VALUE].interpolate(method="linear")
        data[id.Column.VALUE] = data[id.Column.VALUE].fillna(0.0)

        data[id.Column.RETURN] = data[id.Column.RETURN] * (
            1 - data[id.Column.RETURN_TAX]
        )

        data = self._fill_period_data(data)

        data[id.Column.NET_SALE_PROFIT] = np.where(
            data[id.Column.NET_INVESTMENT] < 0, -data[id.Column.NET_INVESTMENT], 0
        )
        data.loc[data[id.Column.NET_INVESTMENT] < 0, id.Column.NET_INVESTMENT] = 0
        data[id.Column.NET_PROFIT] = (
            data[id.Column.VALUE]
            - data[id.Column.NET_INVESTMENT]
            + data[id.Column.NET_RETURN]
            + data[id.Column.NET_SALE_PROFIT]
        )

        non_zero = data[id.Column.NET_INVESTMENT_MAX] != 0
        data.loc[non_zero, id.Column.RELATIVE_NET_PROFIT] = (
            data.loc[non_zero, id.Column.NET_PROFIT]
            / data.loc[non_zero, id.Column.NET_INVESTMENT_MAX]
        )

        return data

    def _fill_period_data(self, input_data: pd.DataFrame) -> pd.DataFrame:
        data = input_data.copy()

        start_of_period = (data[id.Column.VALUE] != 0) & (
            data[id.Column.VALUE].shift(1) == 0
        )

        data[id.Column.PERIOD] = np.where(start_of_period, 1, np.nan)
        data[id.Column.PERIOD] = (
            data[id.Column.PERIOD].cumsum().interpolate(method="pad")
        )
        data[id.Column.PERIOD] = data[id.Column.PERIOD].fillna(0) + 1

        zero_mask = (
            (data[id.Column.VALUE] == 0)
            & (data[id.Column.INVESTMENT] == 0)
            & (data[id.Column.RETURN] == 0)
        )
        data.loc[zero_mask, id.Column.PERIOD] = 0

        for prd in data[id.Column.PERIOD].unique():
            period_mask = data[id.Column.PERIOD] == prd
            data.loc[period_mask, id.Column.NET_RETURN] = data.loc[
                period_mask, id.Column.RETURN
            ].cumsum()
            data.loc[period_mask, id.Column.NET_INVESTMENT] = data.loc[
                period_mask, id.Column.INVESTMENT
            ].cumsum()
            data.loc[period_mask, id.Column.NET_INVESTMENT_MAX] = data.loc[
                period_mask, id.Column.NET_INVESTMENT
            ].cummax()

        return data
