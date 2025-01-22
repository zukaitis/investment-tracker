import babel.dates
import babel.numbers
import datetime
import numpy as np


class Locale:
    def __init__(self, locale: str, currency: str):
        self._locale = locale
        self._currency = currency

    def currency_str(self, value: float) -> str:
        if np.isnan(value) or np.isinf(value):
            return "?"
        precision = 2
        return babel.numbers.format_currency(
            round(value, precision),
            self._currency,
            locale=self._locale,
            decimal_quantization=False,
        )

    def percentage_str(self, value: float) -> str:
        if np.isnan(value) or np.isinf(value):
            return "?"
        precision = 4
        if abs(value) >= 0.5:  # 50%
            precision = 2
        elif abs(value) >= 0.05:  # 5.0%
            precision = 3
        return babel.numbers.format_percent(
            round(value, precision), locale=self._locale, decimal_quantization=False
        )

    def decimal_str(self, value: float, precision: int = None) -> str:
        if np.isnan(value) or np.isinf(value):
            return "?"
        if precision != None:
            value = round(value, precision)
        return babel.numbers.format_decimal(value, locale=self._locale)

    def date_str(self, date: datetime.date) -> str:
        return babel.dates.format_date(date, locale=self._locale)

    def currency_symbol(self) -> str:
        return babel.numbers.get_currency_symbol(self._currency, locale=self._locale)

    def currency_tick_prefix(self) -> str:
        if (
            babel.numbers.Locale.parse(self._locale)
            .currency_formats['standard']
            .pattern.startswith('¤')
        ):
            if (
                babel.numbers.Locale.parse(self._locale)
                .currency_formats['standard']
                .pattern.startswith('¤\xa0')
            ):
                return f'{self.currency_symbol()} '  # add space after the symbol
            return f'{self.currency_symbol()}'

    def currency_tick_suffix(self) -> str:
        if (
            babel.numbers.Locale.parse(self._locale)
            .currency_formats['standard']
            .pattern.endswith('¤')
        ):
            if (
                babel.numbers.Locale.parse(self._locale)
                .currency_formats['standard']
                .pattern.endswith('\xa0¤')
            ):
                return f' {self.currency_symbol()}'  # add space before the symbol
            return f'{self.currency_symbol()}'

    def percentage_tick_suffix(self) -> str:
        string = self.percentage_str(777)  # checking generated string of random percentage
        if string[-2] == ' ':
            return ' %'
        return '%'
