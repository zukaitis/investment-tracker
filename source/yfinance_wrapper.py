import datetime
import yfinance as yf
import pandas as pd
import supress

from source import report
from source import settings
from source import dataset


class YfinanceWrapper:
    def __init__(self, settings: settings.Settings):
        self.settings = settings

    def get_historical_data(
        self, symbol: str, start_date: datetime.datetime
    ) -> pd.DataFrame:
        ticker = yf.Ticker(symbol)
        with supress.supressed():
            fine = ticker.history(period="5d", interval="60m")
        coarse_end_date = None
        if not fine.empty:
            fine.index = fine.index.tz_convert(self.settings.timezone).tz_localize(None)
            coarse_end_date = fine.index[0].date()
        with supress.supressed():
            coarse = ticker.history(
                start=start_date, end=coarse_end_date, interval="1d"
            )
            coarse.index = coarse.index.tz_convert(self.settings.timezone).tz_localize(
                None
            )
        data = pd.concat([coarse, fine])

        if data.empty:
            raise ValueError(f'Symbol "{symbol}" not found')

        if "currency" in ticker.fast_info:
            if ticker.fast_info["currency"] != self.settings.currency:
                # convert currency, if it differs from the one selected in settings
                currency_ticker = yf.Ticker(
                    f"{ticker.fast_info['currency']}{self.settings.currency}=X"
                )
                currency_rate = currency_ticker.history(start=start_date, interval="1d")
                data[self.settings.autofill_price_mark] *= currency_rate[
                    self.settings.autofill_price_mark
                ]
                data["Dividends"] *= currency_rate[self.settings.autofill_price_mark]
        else:
            report.warn(
                f"Ticker currency info is missing. "
                f"Assuming, that ticker currency matches input currency ({self.settings.currency})"
            )

        data = data.rename(
            columns={
                self.settings.autofill_price_mark: dataset.id.Column.PRICE,
                "Dividends": dataset.id.Column.RETURN,
            }
        )

        return data[[dataset.id.Column.PRICE, dataset.id.Column.RETURN]]

    def get_info(symbol: str) -> str:
        ticker = yf.Ticker(symbol)
        if "description" in ticker.fast_info:
            return ticker.fast_info["description"]
        if "longBusinessSummary" in ticker.fast_info:
            return ticker.fast_info["longBusinessSummary"]
        return ""
