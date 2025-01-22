import enum


class Index(enum.Enum):
    ID = enum.auto()
    DATE = enum.auto()


class Column(enum.Enum):
    AMOUNT = enum.auto()
    PRICE = enum.auto()
    VALUE = enum.auto()
    INVESTMENT = enum.auto()
    PERIOD = enum.auto()
    NET_INVESTMENT = enum.auto()
    NET_INVESTMENT_MAX = enum.auto()
    RETURN_PER_UNIT = enum.auto()
    RETURN = enum.auto()
    NET_RETURN = enum.auto()
    RETURN_TAX = enum.auto()
    NET_SALE_PROFIT = enum.auto()
    NET_PROFIT = enum.auto()
    RELATIVE_NET_PROFIT = enum.auto()
    COMMENT = enum.auto()


class Attribute(enum.Enum):
    NAME = enum.auto()
    SYMBOL = enum.auto()
    GROUP = enum.auto()
    ACCOUNT = enum.auto()
    COLOR = enum.auto()
    INFO = enum.auto()
    FILENAME = enum.auto()
    VALUE = enum.auto()
    NET_INVESTMENT = enum.auto()
    NET_INVESTMENT_MAX = enum.auto()
    NET_RETURN = enum.auto()
    NET_PROFIT = enum.auto()
    RELATIVE_NET_PROFIT = enum.auto()
    ACTIVE = enum.auto()
    YFINANCE_FETCH_SUCCESSFUL = enum.auto()
    DISPLAY_PRICE = enum.auto()


def get_corresponding_attribute(column: Column) -> Attribute:
    map = {
        Column.VALUE: Attribute.VALUE,
        Column.NET_INVESTMENT: Attribute.NET_INVESTMENT,
        Column.NET_INVESTMENT_MAX: Attribute.NET_INVESTMENT_MAX,
        Column.NET_RETURN: Attribute.NET_RETURN,
        Column.NET_PROFIT: Attribute.NET_PROFIT,
        Column.RELATIVE_NET_PROFIT: Attribute.RELATIVE_NET_PROFIT,
    }

    return map[column]
