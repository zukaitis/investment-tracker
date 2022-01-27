import datetime
import yfinance as yf
import pandas as pd
import supress

import _report as report
import _settings as settings
import _dataset_identification as id

class YfinanceWrapper:
    def __init__(self, settings: settings.Settings):
        self.settings = settings

    def get_historical_data(self, symbol: str, start_date: datetime.datetime) -> pd.DataFrame:
        ticker = yf.Ticker(symbol)
        with supress.supressed():
            fine = ticker.history(period='5d', interval='60m')
        coarse_end_date = None
        if not fine.empty:
            fine.index = fine.index.tz_convert(self.settings.timezone).tz_localize(None)
            coarse_end_date = fine.index[0].date()
        with supress.supressed():
            coarse = ticker.history(start=start_date, end=coarse_end_date, interval='1d')
        data = pd.concat([coarse, fine])

        if data.empty:
            raise ValueError(f'Symbol "{symbol}" not found')

        if 'currency' in ticker.info:
            if ticker.info['currency'] != self.settings.currency:
                # convert currency, if it differs from the one selected in settings
                currency_ticker = yf.Ticker(f"{ticker.info['currency']}{self.settings.currency}=X")
                currency_rate = currency_ticker.history(start=start_date, interval='1d')
                data[self.settings.autofill_price_mark] *= (
                    currency_rate[self.settings.autofill_price_mark])
                data['Dividends'] *= currency_rate[self.settings.autofill_price_mark]
        else:
            report.warn(
                f'Ticker currency info is missing. '
                f'Assuming, that ticker currency matches input currency ({self.settings.currency})')

        data = data.reset_index().rename(columns={'index':id.Index.DATE, #'Date':id.Index.DATE,
            self.settings.autofill_price_mark:id.Column.PRICE, 'Dividends': id.Column.RETURN})

        return data[[id.Index.DATE, id.Column.PRICE, id.Column.RETURN]]

    def get_info(symbol: str) -> str:
        ticker = yf.Ticker(symbol)
        if 'description' in ticker.info:
            return ticker.info['description']
        if 'longBusinessSummary' in ticker.info:
            return ticker.info['longBusinessSummary']
        return ''
