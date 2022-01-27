import datetime
import enum
import dataclasses
import numpy as np
import pandas as pd
import plotly.express as px

import _dataset_identification as id
import _report as report
import _settings as settings
import _yfinance_wrapper as yfinance_wrapper

_interpolation_method = {
    id.Column.AMOUNT: 'pad',
    id.Column.VALUE: 'linear',
    id.Column.INVESTMENT: 'pad',
    id.Column.RETURN: None,
    id.Column.RETURN_TAX: 'pad',
    id.Column.PRICE: 'pad'
}

class Dataset:
    def __init__(self, settings:settings.Settings):
        self._settings = settings
        self._yfinance = yfinance_wrapper.YfinanceWrapper(self._settings)
        self.attributes = pd.DataFrame(columns=list(id.Attribute))
        self.historical_data = {}
        self.latest_date = None
        self.earliest_date = None

    def append(self, filedict: dict):
        self._append_asset_attributes(filedict)

        last_id = self.attributes.index[-1]
        try:
            self._append_historical_data(filedict['data'], identifier=last_id)
        except Exception as error:
            self.attributes.drop(last_id, inplace=True)  # remove attribute if something went wrong
            raise ValueError(error) from error

        self._recalculate_attribute_data()

    def _recalculate_attribute_data(self):
        self.latest_date = max([max(a.index) for _, a in self.historical_data.items()])
        self.earliest_date = min([min(a.index) for _, a in self.historical_data.items()])

        for identifier in self.attributes.index:
            self.attributes.at[identifier, id.Attribute.CURRENT_VALUE] = (
                self.historical_data[identifier][id.Column.VALUE].iloc[-1])

            if any(self.historical_data[identifier][id.Column.VALUE] > 0):
                self.attributes.at[identifier, id.Attribute.IS_RELEVANT] = (
                    self.attributes.at[identifier, id.Attribute.CURRENT_VALUE] != 0)
            else:
                # taxes are considered irrelevant,
                # if they weren't updated for a certain amount of time
                self.attributes.at[identifier, id.Attribute.IS_RELEVANT] = (
                    max(self.historical_data[identifier].index) < (self.latest_date
                        - pd.tseries.frequencies.to_offset(self._settings.relevance_period)))

        self._reassign_colors()

    def _reassign_colors(self):
        self.attributes.sort_values(by=[id.Attribute.CURRENT_VALUE, id.Attribute.IS_RELEVANT],
            inplace=True, ascending=False)

        group_list = self.attributes[id.Attribute.GROUP].unique()
        if np.nan in group_list:
            group_list.remove(np.nan)
        print(group_list)
        for rrr, group in self.attributes.groupby(id.Attribute.GROUP):
            boundary_colors = [px.colors.qualitative.Set1[index], px.colors.qualitative.Pastel1[index]]
            c = px.colors.n_colors(boundary_colors[0], boundary_colors[1], max(len(group.index), 4), colortype='rgb')
            c = c[:len(group.index)]
            group[id.Attribute.COLOR] = c
            index += 1


    def _append_asset_attributes(self, filedict: dict):
        expected_attributes = {
            id.Attribute.NAME: 'name',
            id.Attribute.SYMBOL: 'symbol',
            id.Attribute.GROUP: 'group',
            id.Attribute.ACCOUNT: 'account',
            id.Attribute.INFO: 'info',
            id.Attribute.FILENAME: 'filename'
        }

        for a in expected_attributes.values():
            if (a in filedict) and (not isinstance(filedict[a], str)):
                report.warn(f'Attribute "{a}" is of wrong type. Attribute type should be string')
                filedict.pop(a)

        identifier = '>'.join([
            filedict[a] for a in ['name', 'symbol', 'account', 'group'] if a in filedict])
        new_entry = pd.DataFrame(
            {key:filedict[val] for key, val in expected_attributes.items() if val in filedict},
            index=[identifier])
        try:
            # check for duplicate IDs is enabled with verify_integrity
            self.attributes = self.attributes.append(new_entry, verify_integrity=True)
        except ValueError as error:
            raise ValueError('Identical asset attributes found in files '
                f'{report.cf.italic(self.attributes.at[identifier, id.Attribute.FILENAME])} and '
                f'{report.cf.italic(filedict["filename"])}. Data from latter file is ignored'
                ) from error

    def _append_historical_data(self, data: list, identifier: str):
        if not isinstance(data, list):
            raise ValueError('Data should be structured as a list')

        new_entry = pd.DataFrame(data)

        if 'date' not in new_entry.columns:
            raise ValueError('Not a single date found in file')

        new_entry = self._convert_historical_data(new_entry)
        if len(new_entry) == 0:
            raise ValueError('No data left in file after filtering')

        symbol = self.attributes.at[identifier, id.Attribute.SYMBOL]
        if (symbol is not np.nan
                and id.Column.VALUE not in new_entry.columns
                and id.Column.PRICE not in new_entry.columns):
            report.report(f'Fetching yfinance data for {report.cf.italic(symbol)}')
            yfdata = self._yfinance.get_historical_data(symbol,
                min(new_entry[id.Index.DATE]))
            self.attributes.at[identifier, id.Attribute.YFINANCE_FETCH_SUCCESSFUL] = True
            new_entry = pd.merge(new_entry, yfdata, on=id.Index.DATE, how='outer')

        new_entry = new_entry.set_index(id.Index.DATE).sort_index()
        new_entry = new_entry.reindex(list(id.Column), axis=1)

        new_entry = self._interpolate_historical_data(new_entry)
        
        self.historical_data.update({identifier:new_entry})

    def _convert_historical_data(self, input_data: pd.DataFrame) -> pd.DataFrame:
        @dataclasses.dataclass()
        class InputColumn:
            name: str
            non_negative: bool = False
            max_value: float = None

        expected_columns = {
            id.Column.INVESTMENT: InputColumn('investment'),
            id.Column.RETURN: InputColumn('return', non_negative=True),
            id.Column.RETURN_TAX: InputColumn('return_tax', non_negative=True, max_value=1.0),
            id.Column.PRICE: InputColumn('price', non_negative=True),
            id.Column.AMOUNT: InputColumn('amount', non_negative=True),
            id.Column.VALUE: InputColumn('value', non_negative=True)
        }

        data = input_data.copy()
        data.rename(
            columns={val.name:key for key, val in expected_columns.items() if val.name in data},
            inplace=True)
        data.rename(columns={'date': id.Index.DATE}, inplace=True)
        if 'comment' in data.columns:
            data.rename(columns={'comment': id.Column.COMMENT}, inplace=True)

        data = data[data[id.Index.DATE].notnull()]  # remove rows without a date
        data[id.Index.DATE] = pd.to_datetime(data[id.Index.DATE], errors='coerce')
        if any(data[id.Index.DATE].isnull()):
            report.warn('Unrecognized dates found in data')
            data = data[data[id.Index.DATE].notnull()]  # remove rows where date was not converted
        if any(data[id.Index.DATE] > datetime.datetime.now()):
            report.warn('Future dates found in data, they will be ignored')
            data = data[data[id.Index.DATE] <= datetime.datetime.now()]

        # filter out unrecognized, negative, and too high values
        available_columns = [c for c in expected_columns if c in data.columns]
        original_data = data.copy()
        data[available_columns] = data[available_columns].apply(pd.to_numeric, errors='coerce')
        for col in available_columns:
            column_name_italic = report.cf.italic(expected_columns[col].name)
            unrecognized = data[col].isnull() & original_data[col].notnull()
            if any(unrecognized):
                report.warn(f'Unrecognized data found in {column_name_italic} column')
                data = data[~unrecognized]
                original_data = original_data[~unrecognized]
            if expected_columns[col].non_negative:
                negative = (data[col] < 0)
                if any(negative):
                    report.warn(f'Negative values found in {column_name_italic} column')
                    data = data[~negative]
                    original_data = original_data[~negative]
            if expected_columns[col].max_value is not None:
                too_high = (data[col] > expected_columns[col].max_value)
                if any(too_high):
                    report.warn(f'Too high values found in {column_name_italic} column. '
                        'Values in this column should not exceed '
                        f'{expected_columns[col].max_value}')
                    data = data[~too_high]
                    original_data = original_data[~too_high]

        return data

    def _interpolate_historical_data(self, input_data: pd.DataFrame) -> pd.DataFrame:
        interpolated_columns = {
            id.Column.AMOUNT: 'pad',
            id.Column.PRICE: 'pad',
            id.Column.INVESTMENT: None,
            id.Column.RETURN: None,
            id.Column.RETURN_TAX: 'pad'}

        data = input_data.copy()
        for col, method in interpolated_columns.items():
            if method is not None:
                data[col] = data[col].interpolate(method=method)
            data[col] = data[col].fillna(0.0)

        data.loc[pd.isna(data[id.Column.VALUE]), id.Column.VALUE] = (
            data[id.Column.AMOUNT] * data[id.Column.PRICE])
        
        data = self._fill_period_data(data)

        data[id.Column.NET_SALE_PROFIT] = np.where(data[id.Column.NET_INVESTMENT] < 0,
            -data[id.Column.NET_INVESTMENT], 0)
        data.loc[data[id.Column.NET_INVESTMENT] < 0, id.Column.NET_INVESTMENT] = 0
        data[id.Column.NET_PROFIT] = (data[id.Column.VALUE] - data[id.Column.NET_INVESTMENT]
            + data[id.Column.NET_RETURN] + data[id.Column.NET_SALE_PROFIT])
        data[id.Column.RELATIVE_NET_PROFIT] = (
            data[id.Column.NET_PROFIT] / data[id.Column.NET_INVESTMENT_MAX])

        return data

    def _fill_period_data(self, input_data: pd.DataFrame) -> pd.DataFrame:
        data = input_data.copy()

        start_of_period = ((data[id.Column.VALUE] != 0) & (data[id.Column.VALUE].shift(1) == 0))
        
        data[id.Column.PERIOD] = np.where(start_of_period, 1, np.nan)
        data[id.Column.PERIOD] = data[id.Column.PERIOD].cumsum().interpolate(method='pad')
        data[id.Column.PERIOD] = data[id.Column.PERIOD].fillna(0) + 1

        zero_mask = ((data[id.Column.VALUE] == 0)
            & (data[id.Column.INVESTMENT] == 0)
            & (data[id.Column.RETURN] == 0))
        data.loc[zero_mask, id.Column.PERIOD] = 0

        for prd in data[id.Column.PERIOD].unique():
            period_mask = (data[id.Column.PERIOD] == prd)
            data.loc[period_mask, id.Column.NET_RETURN] = (
                data.loc[period_mask, id.Column.RETURN].cumsum())
            data.loc[period_mask, id.Column.NET_INVESTMENT] = (
                data.loc[period_mask, id.Column.INVESTMENT].cumsum())
            data.loc[period_mask, id.Column.NET_INVESTMENT_MAX] = (
                data.loc[period_mask, id.Column.NET_INVESTMENT].cummax())

        return data
