import datetime
import enum
import dataclasses
import pandas as pd

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
        self._yfinance = yfinance_wrapper.YfinanceWrapper(settings)
        self.attributes = pd.DataFrame(columns=list(id.Attribute))
        index = pd.MultiIndex([[]]*2, [[]]*2, names=list(id.Index))
        self.historical_data = pd.DataFrame(columns=list(id.Column), index=index)

    def append(self, filedict: dict):
        self._append_asset_attributes(filedict)

        last_id = self.attributes.index[-1]
        try:
            self._append_historical_data(filedict['data'], identifier=last_id)
        except Exception as error:
            self.attributes.drop(last_id, inplace=True)  # remove attribute if something went wrong
            raise ValueError(error) from error

        # self.update_attributes()

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
        new_entry[id.Index.ID] = identifier

        if 'date' not in new_entry.columns:
            raise ValueError('Not a single date found in file')

        new_entry = self._convert_historical_data(new_entry)
        if len(new_entry) == 0:
            raise ValueError('No data left in file after filtering')

        #print(new_entry)

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

        data = data.set_index(list(id.Index)).sort_index()
        data = data.reindex(list(id.Column), axis=1)  # add missing columns
        return data
