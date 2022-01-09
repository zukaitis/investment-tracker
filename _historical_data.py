import _report as report
import _settings as settings

import yfinance as yf
import pandas as pd
import supress


def get(symbol: str, start_date: pd.DatetimeIndex, settings: settings.Settings) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    with supress.supressed():
        fine = ticker.history(period='5d', interval='60m').astype(float)
    coarse_end_date = None
    if not fine.empty:
        fine.index = fine.index.tz_convert(settings.timezone).tz_localize(None)
        coarse_end_date = fine.index[0].date()
    with supress.supressed():
        coarse = ticker.history(start=start_date, end=coarse_end_date, interval='1d')
    data = pd.concat([coarse, fine])

    if data.empty:
        raise ValueError(f'Symbol "{symbol}" not found')

    if 'currency' in ticker.info:
        if ticker.info['currency'] != settings.currency:
            # convert currency, if it differs from the one selected in settings
            currency_ticker = yf.Ticker(f"{ticker.info['currency']}{settings.currency}=X")
            currency_rate = currency_ticker.history(start=start_date, interval='1d')
            data[settings.autofill_price_mark] *= currency_rate[settings.autofill_price_mark]
            data['Dividends'] *= currency_rate[settings.autofill_price_mark]
    else:
        report.warn(
            f'Ticker currency info is missing. '
            f'Assuming, that ticker currency matches input currency ({settings.currency})')

    data = data.reset_index().rename(columns={'index':'date', 'Date':'date',
        settings.autofill_price_mark:'price', 'Dividends': 'dividends'})

    if 'description' in ticker.info:
        data['info'] = ticker.info['description']
    elif 'longBusinessSummary' in ticker.info:
        data['info'] = ticker.info['longBusinessSummary']

    data = data[['date', 'price', 'info', 'dividends']]

    return data