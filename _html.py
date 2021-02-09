import dominate
import dominate.tags as dt
from dominate.util import raw
from dataclasses import dataclass
import typing
from pathlib import Path

positive_color = 'green'
negative_color = 'red'

class _HtmlObject:
    def __init__(self):
        self._raw = ''

    def __str__(self):
        return self._raw

class Document:
    def __init__(self, title: str, css_variables: dict):
        self._document = dominate.document(title=title)
        self._document.head += raw('<meta charset="utf-8"/>')
        self._document.head += raw('<style type=text/css>')
        self._append_css_variables_to_head(css_variables)
        self._append_file_to_head('style.css')
        self._document.head += raw('</style>')
        self._document += raw('<div class="fixed_button_area">')
        self._document += raw('<div class="button">')
        self._document += raw('<input type="checkbox" id="pompa" name="pompa">')
        self._document += raw('<label for="pompa" class="button_image0">')
        self._document += raw('<div class="button_image_padding>"')
        self._document += raw(Path('calendar_month_nu.svg').read_text())
        self._document += raw('</div>')
        self._document += raw('</label>')
        self._document += raw('<label for="pompa" class="button_image1">')
        self._document += raw('<div class="button_image_padding>"')
        self._document += raw(Path('calendar_day_nu.svg').read_text())
        self._document += raw('</div>')
        self._document += raw('</label>')
        self._document += raw('</div>')
        self._document += raw('</div>')

    def append(self, obj: typing.Union[_HtmlObject, str]):
        self._document += raw(str(obj))

    def _append_file_to_head(self, file: str):
        self._document.head += raw(Path(file).read_text())

    def _append_css_variables_to_head(self, variables: str):
        self._document.head += raw(':root {')
        for v in variables:
            self._document.head += raw(f'--{v}: {variables[v]};')
        self._document.head += raw('}')

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
        container_name = f'container{self.container_index}'
        width = 100 / self.tab_count
        for i in range(self.tab_count):
            tab_name = f'{container_name}_tab{i}'
            self._raw += f'<input id="{tab_name}" type="radio" name="{container_name}"'
            self._raw += f' checked>' if tab[i].checked else f'>'
            self._raw += f'<label class="tab_label" style="width:{width:.2f}%" for="{tab_name}">'
            if tab[i].label != None:
                self._raw += str(tab[i].label)
            self._raw += '</label>'

        for i in range(self.tab_count):
            id = f'{container_name}_content{i}'
            self._raw += f'<section id="{id}" class="tab-content">'
            if tab[i].content != None:
                self._raw += str(tab[i].content)
            self._raw += '</section>'

        self._raw += '</div>'
        tab_container_count += 1

@dataclass
class Column:
    content: str = None
    width: float = None

class Columns(_HtmlObject):
    def __init__(self, column: typing.List[Column]):
        self._raw = '<div>'
        column = self._fill_width_fields(column)
        for c in column:
            self._raw += f'<div class="column" style="width:{c.width:.1f}%">'
            if c.content != None:
                self._raw += str(c.content)
            self._raw += '</div>'
        self._raw += '</div>'

    def _fill_width_fields(self, column: list) -> list:
        output = column
        for i in range(len(output)):
            output[i] = Column(output[i]) if type(output[i]) is not Column else output[i]
        remaining_width = 100 - sum([c.width for c in output if c.width != None])
        floating_column_count = sum([1 for c in output if c.width == None])

        if floating_column_count > 0:
            width = remaining_width / floating_column_count
            for c in output:
                if c.width == None:
                    c.width = width
        return output

class ValueChange(_HtmlObject):
    def __init__(self, values: typing.List[str], button_identifier: str):
        self._raw = ''

class Value(_HtmlObject):
    def __init__(self, value: str, text_color: str = None, value_change: str = None):
        self.value = value
        self.text_color = text_color
        self.value_change = value_change

    def __str__(self):
        self._raw = f'<span'
        self._raw += f' style="color:{self.text_color};">' if (self.text_color != None) else '>'
        self._raw += f'{self.value}</span>'
        if self.value_change != None:
            symbol = '▾' if self._is_negative(self.value_change) else '▴'
            color = negative_color if self._is_negative(self.value_change) else positive_color
            self._raw += f'<span class="value_change" style="color:{color};">'
            self._raw += f' {symbol}{self.value_change.replace("-", "")}</span>'
        return self._raw

    def color(self):
        if self._is_negative(self.value):
            self.text_color = negative_color
        elif self._is_not_zero(self.value):
            self.text_color = positive_color
        return self

    def _is_negative(self, value: str) -> bool:
        return ('-' in value)

    def _is_not_zero(self, value: str) -> bool:
        # check if there are digits other than 0 in the string
        return any([str(d) in value for d in range(1, 10)])

class Label(_HtmlObject):
    def __init__(self, name: str, value: _HtmlObject = None):
        self._raw = f'<span class="label_name">{name}</span>'
        if value != None:
            self._raw += f'<br>{value}'

button_count = 0

class Button(_HtmlObject):
    def __init__(self, images: typing.List[str], identifier: str):
        self.identifier = identifier
        self._raw = '<div class="button">'
        self._raw += f'<input type="checkbox" id="{identifier}" name="{identifier}">'
        for i in range(len(images)):
            self._raw += f'label for="{identifier}" class="{identifier}_img{i}"'
            self._raw += '<div class="button_image_padding>"'
            self._raw += Path(images[i]).read_text()
            self._raw += '</div>'
            self._raw += '</label>'
        self._raw += '</div>'

class FixedButtons(_HtmlObject):
    def __init__(self, buttons: typing.List[Button]):
        self._raw = '<div class="fixed_button_area">'
        self._raw += '<style type=text/css>'
        for b in buttons:
            id = b.identifier
            self._raw += f'input#{id} {{ display: none; }}'
            self._raw += f'.{id}_img0 {{ display: none; }}'
            self._raw += f'input#{id}:checked ~ .{id}_img0 {{ display: initial; }}'
            self._raw += f'input#{id}:checked ~ .{id}_img1 {{ display: none; }}'
        self._raw += '</style>'

        self._raw += '<div class="fixed_button_area">'
        for b in buttons:
            self._raw += b
        self._raw += '</div>'


