import dominate
import dominate.tags as dt
from dominate.util import raw
from dataclasses import dataclass
import typing
from pathlib import Path

_positive_color = "green"
_negative_color = "red"

_button_css = """{
    right: 2%;
    top: 2em;
    position: fixed;
    z-index: 100;
    width: 3em;
    height: 3em;
    padding: 0.6em;
    background-color: var(--hover_tab_indicator_color);
    border-radius: 50%;
    user-select: none;
    opacity: 0.2;
}"""


class _HtmlObject:
    def __init__(self):
        self._raw = ""

    def __str__(self):
        return self._raw

    def __add__(self, other):
        return str(self) + other

    def __radd__(self, other):
        return other + str(self)

    def __mul__(self, multiplier):
        return str(self) * multiplier

    def __rmul__(self, multiplier):
        return multiplier * str(self)


class Document:
    def __init__(self, title: str, css_variables: dict):
        self._document = dominate.document(title=title)
        self._document.head += raw('<meta charset="utf-8"/>')
        self._document.head += raw("<style type=text/css>")
        self._append_css_variables_to_head(css_variables)
        self._append_file_to_head("style.css")
        self._document.head += raw("</style>")

    def append(self, obj: typing.Union[_HtmlObject, str]):
        self._document += raw(str(obj))

    def _append_file_to_head(self, file: str):
        self._document.head += raw(Path(file).read_text())

    def _append_css_variables_to_head(self, variables: str):
        self._document.head += raw(":root {")
        for v in variables:
            self._document.head += raw(f"--{v}: {variables[v]};")
        self._document.head += raw("}")

    def __repr__(self):
        return str(self._document)


@dataclass
class Tab:
    label: str = None
    content: str = None
    checked: bool = False


tab_container_count = 0


class TabContainer(_HtmlObject):
    def __init__(self, tab: typing.List[Tab]):
        self._raw = '<div class="tab_container">'
        global tab_container_count
        self.container_index = tab_container_count
        self.tab_count = len(tab)
        container_name = f"container{self.container_index}"
        width = 100 / self.tab_count
        for i in range(self.tab_count):
            tab_name = f"{container_name}_tab{i}"
            self._raw += f'<input id="{tab_name}" type="radio" name="{container_name}"'
            self._raw += f" checked>" if tab[i].checked else f">"
            self._raw += (
                f'<label class="tab_label" style="width:{width:.2f}%" for="{tab_name}">'
            )
            if tab[i].label != None:
                self._raw += str(tab[i].label)
            self._raw += "</label>"

        for i in range(self.tab_count):
            id = f"{container_name}_content{i}"
            self._raw += f'<section id="{id}" class="tab-content">'
            if tab[i].content != None:
                self._raw += str(tab[i].content)
            self._raw += "</section>"

        self._raw += "</div>"
        tab_container_count += 1


@dataclass
class Column:
    content: str = None
    width: float = None


class Columns(_HtmlObject):
    def __init__(self, column: typing.List[Column]):
        self._raw = "<div>"
        column = self._fill_width_fields(column)
        for c in column:
            self._raw += f'<div class="column" style="width:{c.width:.1f}%">'
            if c.content != None:
                self._raw += str(c.content)
            self._raw += "</div>"
        self._raw += "</div>"

    def _fill_width_fields(self, column: list) -> list:
        output = column
        for i in range(len(output)):
            output[i] = (
                Column(output[i]) if type(output[i]) is not Column else output[i]
            )
        remaining_width = 100 - sum([c.width for c in output if c.width != None])
        floating_column_count = sum([1 for c in output if c.width == None])

        if floating_column_count > 0:
            width = remaining_width / floating_column_count
            for c in output:
                if c.width == None:
                    c.width = width
        return output


def _is_negative(value: str) -> bool:
    return "-" in value


def _is_not_zero(value: str) -> bool:
    # check if there are digits other than 0 in the string
    return any([str(d) in value for d in range(1, 10)])


class ValueChange(_HtmlObject):
    def __init__(self, daily: str = None, monthly: str = None):
        self._raw = '<span class="value_changes">'
        if (daily is not None) and (_is_not_zero(daily)):
            self._append_span(daily, "daily_change")
        if (monthly is not None) and (_is_not_zero(monthly)):
            self._append_span(monthly, "monthly_change")
        self._raw += "</span>"

    def _append_span(self, value_change: str, css_class: str):
        symbol = "▾" if _is_negative(value_change) else "▴"
        color = _negative_color if _is_negative(value_change) else _positive_color
        self._raw += f'<span class="{css_class}" style="color:{color};">'
        self._raw += f' {symbol}{value_change.replace("-", "")}</span>'


class Value(_HtmlObject):
    def __init__(
        self, value: str, textcolor: str = None, valuechange: ValueChange = None
    ):
        self.value = value
        self.text_color = textcolor
        self.value_change = valuechange

    def __str__(self):
        self._raw = f"<span"
        self._raw += (
            f' style="color:{self.text_color};">' if (self.text_color != None) else ">"
        )
        self._raw += f"{self.value}</span>"
        if self.value_change != None:
            self._raw += str(self.value_change)
        return self._raw

    def color(self):
        if _is_negative(self.value):
            self.text_color = _negative_color
        elif _is_not_zero(self.value):
            self.text_color = _positive_color
        return self


class Label(_HtmlObject):
    def __init__(self, name: str, value: _HtmlObject = None):
        self._raw = f'<span class="label_name">{name}</span>'
        if value != None:
            self._raw += f"<br>{value}"


class Button(_HtmlObject):
    def __init__(self, image_initial: str, image_alternate: str, identifier: str):
        # adding css styling of button
        self._raw = f"<style type=text/css>"
        self._raw += f".{identifier}_unchecked, .{identifier}_checked {_button_css}"
        self._raw += f".{identifier}_unchecked:hover, .{identifier}_checked:hover {{opacity: 1;}}"
        self._raw += f"input#{identifier} {{ display: none; }}"
        self._raw += f".{identifier}_unchecked {{ display: none; }}"
        self._raw += f"input#{identifier}:checked ~ .{identifier}_unchecked {{display: initial;}}"
        self._raw += (
            f"input#{identifier}:checked ~ .{identifier}_checked {{ display: none; }}"
        )
        self._raw += f"</style>"

        # adding html
        self._raw += (
            f'<input type="checkbox" id="{identifier}" name="{identifier}" checked>'
        )
        self._raw += f'<label for="{identifier}" class="{identifier}_unchecked">'
        self._raw += Path(image_initial).read_text()
        self._raw += f"</label>"
        self._raw += f'<label for="{identifier}" class="{identifier}_checked">'
        self._raw += Path(image_alternate).read_text()
        self._raw += f"</label>"


class Divider(_HtmlObject):
    def __init__(self):
        self._raw = "<hr>"


class Heading2(_HtmlObject):
    def __init__(self, text: str):
        self._raw = f"<h2>{text}</h2>"


class Paragraph(_HtmlObject):
    def __init__(self, text: str):
        self._raw = f'<p class="asset_info">{text}</p>'
