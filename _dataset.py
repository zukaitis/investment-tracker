import _report as report

import enum
import dataclasses
import typing
import pandas as pd


class Column(enum.Enum):
    AMOUNT = enum.auto()
    PRICE = enum.auto()
    VALUE = enum.auto()
    INVESTMENT = enum.auto()
    PERIOD = enum.auto()
    NET_INVESTMENT = enum.auto()
    MAXIMUM_NET_INVESTMENT = enum.auto()
    RETURN = enum.auto()
    NET_RETURN = enum.auto()
    RETURN_TAX = enum.auto()
    NET_SALE_PROFIT = enum.auto()
    NET_PROFIT = enum.auto()
    RELATIVE_NET_PROFIT = enum.auto()
    COMMENT = enum.auto()

_interpolation_method = {
    Column.AMOUNT: 'pad',
    Column.VALUE: 'linear',
    Column.INVESTMENT: 'pad',
    Column.RETURN: None,
    Column.RETURN_TAX: 'pad',
    Column.PRICE: 'pad'
}

class Attribute(enum.Enum):
    NAME = enum.auto()
    SYMBOL = enum.auto()
    GROUP = enum.auto()
    ACCOUNT = enum.auto()
    COLOR = enum.auto()
    INFO = enum.auto()
    FILENAME = enum.auto()
    CURRENT_VALUE = enum.auto()
    IS_RELEVANT = enum.auto()

class Dataset:
    def __init__(self):
        self.attributes = pd.DataFrame(columns=Attribute)
        index = pd.MultiIndex([[]]*2, [[]]*2, names=['id', 'date'])
        self.historical_data = pd.DataFrame(columns=Column, index=index)

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
            Attribute.NAME: 'name',
            Attribute.SYMBOL: 'symbol',
            Attribute.GROUP: 'group',
            Attribute.ACCOUNT: 'account',
            Attribute.INFO: 'info',
            Attribute.FILENAME: 'filename'
        }

        for a in expected_attributes.values():
            if (a in filedict) and (not isinstance(filedict[a], str)):
                report.warn(f'Attribute "{a}" in a file {filedict["filename"]} is of wrong type. '
                    'Attribute type should be string')
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
                f'{self.attributes.at[identifier, Attribute.FILENAME]} and '
                f'{filedict["filename"]}. Data from latter file is ignored') from error

    def _append_historical_data(self, data: typing.Union[list, dict], identifier: str):
        if (not isinstance(data, (list, dict))) and (data is not None):
            raise ValueError('Wrong type of data in '
                f'{self.attributes.at[identifier, Attribute.FILENAME]}. '
                'It should be either a list or a dictionary')

        new_entry = pd.DataFrame(data)
        new_entry['id'] = identifier

        if 'date' not in new_entry.columns:
            raise ValueError('Not a single date found in '
                f'{self.attributes.at[identifier, Attribute.FILENAME]}. Ignoring this file')

        new_entry = self._convert_historical_data(new_entry)

        print(new_entry)

    def _convert_historical_data(self, data: pd.DataFrame) -> pd.DataFrame:
        @dataclasses.dataclass()
        class InputColumn:
            name: str
            min_value: float = None
            max_value: float = None

        expected_columns = {
            Column.INVESTMENT: InputColumn('investment'),
            Column.RETURN: InputColumn('return', min_value=0),
            Column.RETURN_TAX: InputColumn('return_tax', min_value=0, max_value=1.0),
            Column.PRICE: InputColumn('price', min_value=0),
            Column.AMOUNT: InputColumn('amount', min_value=0),
            Column.VALUE: InputColumn('value', min_value=0)
        }

        data.rename(
            columns={val.name:key for key, val in expected_columns.items() if val.name in data},
            inplace=True)

        data['date'] = pd.to_datetime(data['date'], errors='ignore')
        available_columns = [c for c in expected_columns if c in data.columns]
        data[available_columns] = data[available_columns].astype(float, errors='ignore')

        for index, row in data.iterrows():
            if not isinstance(row['date'], datetime.datetime):
                

        return data