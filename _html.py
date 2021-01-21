import dominate
import dominate.tags as dt
from dominate.util import raw
from dataclasses import dataclass
import typing
from pathlib import Path

class _HtmlObject:
    def __init__(self):
        self._raw = ''

    def __str__(self):
        return self._raw

class Document:
    def __init__(self, title: str, css_variables: dict):
        self._document = dominate.document(title=title)
        self._document.head += raw('<meta charset="utf-8"/>')
        #self._document.head += raw('<link rel="stylesheet" href="style.css">')
        self._document.head += raw('<style type=text/css>')
        self._append_css_variables_to_head(css_variables)
        self._append_file_to_head('style.css')
        self._document.head += raw('</style>')

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
            self._raw += f'<label style="width:{width:.2f}%" for="{tab_name}">'
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

@dataclass
class Value:
    value: str = ''
    text_color: str = None

    def color(self):
        if '-' in self.value:
            self.text_color = 'red'
        # check if there are digits other than 0 in string
        elif any([str(d) in self.value for d in range(1, 10)]):
            self.text_color = 'green'
        return self

class Label(_HtmlObject):
    def __init__(self, name: str, value: Value = None):
        self._raw = f'<span class="label_name">{name}</span>'
        if value != None:
            self._raw += f'<br><span'
            self._raw += f' style=color:{value.text_color}>' if (value.text_color != None) else '>'
            self._raw += f'{value.value}</span>'
