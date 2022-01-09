import pandas as pd
import enum
import dataclasses

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
            self._append_historical_data(filedict['data'], id=last_id)
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

    def _append_historical_data(self, data: dict, id: str):
        expected_columns = {
            Column.INVESTMENT: 'investment',
            Column.RETURN: 'return',
            Column.RETURN_TAX: 'return_tax',
            Column.PRICE: 'price',
            Column.AMOUNT: 'amount',
            Column.VALUE: 'value'
        }

        new_entry = pd.DataFrame(
            {key:date[val] for key, val in expected_columns.items() if val in date},
            index=[id, data['date']])
        print(new_entry)
