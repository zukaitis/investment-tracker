from _common import print_warning

import babel
import yfinance as yf
import pandas as pd

class Settings:
    owner = 'Your'
    currency = 'EUR'
    locale = 'en_US_POSIX'
    autofill_interval = '1d'
    autofill_price_mark = 'Close'
    theme = 'auto'
    value_change_span = '2d'
    relevance_period = '6M'
    timezone = 'UTC'

    def __setattr__(self, name, value):
        if name == 'owner':
            self.__dict__[name] = f"{value}'s"
        elif name == 'currency':
            try:
                yf.Ticker(f'{value}=X').info
            except ValueError:
                print_warning(f'Unknown currency - "{value}"')
            else:
                self.__dict__[name] = value
        elif name == 'locale':
            if babel.localedata.exists(value):
                self.__dict__[name] = value
            else:
                print_warning(f'Unknown locale - "{value}"')
        elif name == 'autofill_interval':
            allowed = ['1d', '5d', '1wk', '1mo', '3mo']
            if value in allowed:
                self.__dict__[name] = value
            else:
                print_warning(f'Unknown interval - "{value}". Allowed intervals: {allowed}')
        elif name == 'autofill_price_mark':
            allowed = ['Open', 'Close', 'High', 'Low']
            if value in allowed:
                self.__dict__[name] = value
            else:
                print_warning(f'Unknown price mark - "{value}". Allowed marks: {allowed}')
        elif name == 'theme':
            allowed = ['light', 'dark', 'auto']
            if value in allowed:
                self.__dict__[name] = value
            else:
                print_warning(f'Unknown theme - "{value}". Allowed themes: {allowed}')
        elif name == 'value_change_span':
            try:
                pd.tseries.frequencies.to_offset(value)
            except ValueError:
                print_warning(f'Unrecognized value change span - {value}')
            else:
                self.__dict__[name] = value
        elif name == 'relevance_period':
            try:
                pd.tseries.frequencies.to_offset(value)
            except ValueError:
                print_warning(f'Unrecognized relevance period - {value}')
            else:
                self.__dict__[name] = value
        elif name not in self:
            print_warning(f'No such setting - "{name}"')
        else:
            self.__dict__[name] = value

    def __iter__(self):
        variables = [d for d in dir(self) if not d.startswith('_')]
        return iter(variables)
