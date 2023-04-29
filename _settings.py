import babel
import yfinance as yf
import pandas as pd
import inspect
import pytz

import _report as report


class _Setting:
    def __init__(self, default, description: str, allowed: list = None):
        self._value = default
        self.allowed = allowed
        self.description = description

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if self.is_value_allowed(value):
            self._value = value
        else:
            raise ValueError

    def is_value_allowed(self, value) -> bool:
        try:
            # some of the checks throw exceptions instead of returning False
            if not self._is_allowed(value):
                raise ValueError
        except:
            return False
        return True

    def _is_allowed(self, value) -> bool:
        if (self.allowed is not None) and (value not in self.allowed):
            return False
        return True


class _Currency(_Setting):
    def _is_allowed(self, value) -> bool:
        if len(yf.Ticker(f"{value}=X").history()) == 0:
            return False
        return True


class _Locale(_Setting):
    def _is_allowed(self, value) -> bool:
        return babel.localedata.exists(value)


class _Period(_Setting):
    def _is_allowed(self, value) -> bool:
        pd.tseries.frequencies.to_offset(value)
        return True


class _Timezone(_Setting):
    def _is_allowed(self, value) -> bool:
        return value in pytz.all_timezones


class _Float(_Setting):
    def _is_allowed(self, value) -> bool:
        float(value)
        return True


class _Name(_Setting):
    @_Setting.value.setter
    def value(self, value: str):
        self._value = f"{value}'s"


class Settings:
    owner = _Name(
        default="Your",
        description="Name of the portfolio owner, which will be displayed in the title",
    )
    currency = _Currency(
        default="EUR", description="Currency, in which all the input data is specified"
    )
    locale = _Locale(
        default="en_US_POSIX",
        description="Locale, which determines, how numbers are displayed",
    )
    autofill_price_mark = _Setting(
        default="Close",
        description="Selects, which value column to use, when fetching financial data",
        allowed=["Open", "Close", "High", "Low"],
    )
    theme = _Setting(
        default="auto",
        description="Color scheme to be used in the report",
        allowed=["light", "dark", "auto"],
    )
    value_change_span = _Period(
        default="2d",
        description="Selects, how recent and frequent data entries have to be, "
        "to display value change",
    )
    relevance_period = _Period(
        default="6M",
        description="Selects, how recent and frequent data entries have to be, "
        "for data be considered relevant",
    )
    timezone = _Timezone(default="UTC", description="Time zone, used in the report")

    def __init__(self):
        for s in self:  # copy all class variables to instance
            self.__dict__[s] = getattr(Settings, s)

    def __setattr__(self, name, value):
        if name not in self:
            report.warn(f'No such setting: "{name}"')
        else:
            try:
                self.__dict__[name].value = value
            except ValueError:
                message = f'Unrecognized {name.replace("_", " ")}: "{value}"'
                if self.__dict__[name].allowed is not None:
                    message += f" Allowed values: {self.__dict__[name].allowed}"
                report.warn(message)

    def __getattribute__(self, name):
        if ("__dict__" == name) or callable(super().__getattribute__(name)):
            return super().__getattribute__(name)  # call from __iter__()
        return super().__getattribute__(name).value

    def __iter__(self):
        variables = [
            d
            for d in dir(Settings)
            if not (d.startswith("_") or callable(getattr(Settings, d)))
        ]
        return iter(variables)

    def get_description(self, name: str):
        return super().__getattribute__(name).description

    def get_allowed(self, name: str):
        return super().__getattribute__(name).allowed
