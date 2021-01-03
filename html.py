import dominate
import dominate.tags as dt
from dominate.util import raw
from dataclasses import dataclass
import typing

class _HtmlObject:
    def __init__(self):
        self._raw = ''

    def __str__(self):
        return self._raw

class Document:
    def __init__(self, title: str):
        self._document = dominate.document(title=title)
        self._document.head += raw('<meta charset="utf-8"/>')
        self._document.head += raw('<link rel="stylesheet" href="style.css">')

    def append(self, obj: typing.Union[_HtmlObject, str]):
        self._document += raw(str(obj))
        # if type(obj) is Html.TabContainer:
        #     self._update_tab_stylesheet(obj.container_index, obj.tab_count)

    # def append(self, raw_html: str):
    #     self._document += raw(raw_html)

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
        container_name = 'container{:d}'.format(self.container_index)
        width = 100 / self.tab_count
        for i in range(self.tab_count):
            tab_name = container_name + '_tab{:d}'.format(i)
            self._raw += ('<input id="{}" '.format(tab_name)
                + 'type="radio" name="{}"'.format(container_name)
                + (' checked>' if tab[i].checked else '>')
                + '<label style="width:{:.2f}%" for="{}">'.format(width, tab_name))
            if tab[i].label != None:
                self._raw += str(tab[i].label)
            self._raw += '</label>'

        for i in range(self.tab_count):
            id = container_name + '_content{:d}'.format(i)
            self._raw += '<section id="{}" class="tab-content">'.format(id)
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
            self._raw += '<div class="column" style="width:{:.1f}%">'.format(c.width)
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
    value: float
    suffix: str = ''
    text_color: str = None

    def color(self):
        if self.value > 0:
            self.text_color = 'green'
        elif self.value < 0:
            self.text_color = 'red'
        return self

class Label(_HtmlObject):
    def __init__(self, name: str, value: Value = None):
        self._raw = '<span class="label_name">{}</span>'.format(name)
        if value != None:
            self._raw += ('<br><span'
                + (' style=color:{}>'.format(value.text_color) if value.text_color != None else '>')
                + '{:.2f}{}</span>'.format(value.value, value.suffix))
