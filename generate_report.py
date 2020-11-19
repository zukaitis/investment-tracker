#!/usr/bin/python3

import os
import yaml
import argparse
import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dominate
import dominate.tags as dt
from dominate.util import raw
from dataclasses import dataclass
import warnings

cyan = px.colors.qualitative.Plotly[5]

class Html:
    _tab_container_count = 0

    @dataclass
    class Tab:
        label: str = None
        content: str = None
        checked: bool = False

    def tab_container(tab: list) -> str:
        output = '<div class="tab_container">'
        container_name = 'container{:d}'.format(Html._tab_container_count)
        width = 100 / len(tab)
        for i in range(len(tab)):
            tab_name = container_name + '_tab{:d}'.format(i)
            output += ('<input id="{}" '.format(tab_name)
                + 'type="radio" name="{}"'.format(container_name)
                + (' checked>' if tab[i].checked else '>')
                + '<label style="width:{:.2f}%" for="{}">'.format(width, tab_name))
            if tab[i].label != None:
                output += tab[i].label
            output += '</label>'

        for i in range(len(tab)):
            id = container_name + '_content{:d}'.format(i)
            output += '<section id="{}" class="tab-content">'.format(id)
            if tab[i].content != None:
                output += tab[i].content
            output += '</section>'

        output += '</div>'
        Html._tab_container_count += 1
        return output

    @dataclass
    class Column:
        content: str = None
        width: float = None

    def columns(column: list) -> str:
        output = '<div>'
        column = Html._fill_width_fields(column)
        for c in column:
            output += '<div class="column" style="width:{:.1f}%">'.format(c.width)
            if c.content != None:
                output += c.content
            output += '</div>'
        output += '</div>'
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

    def label(name: str, value: Value = None) -> str:
        output = '<span class="label_name">{}</span>'.format(name)
        if value != None:
            output += ('<br><span'
                + (' style=color:{}>'.format(value.text_color) if value.text_color != None else '>')
                + '{:.2f}{}</span>'.format(value.value, value.suffix))
        return output

    def _fill_width_fields(column: list) -> list:
        output = column
        for i in range(len(output)):
            output[i] = Html.Column(output[i]) if type(output[i]) is not Html.Column else output[i]
        remaining_width = 100 - sum([c.width for c in output if c.width != None])
        floating_column_count = sum([1 for c in output if c.width == None])

        if floating_column_count > 0:
            width = remaining_width / floating_column_count
            for c in output:
                if c.width == None:
                    c.width = width
        return output

def process_data(input_data : list) -> pd.DataFrame:
    data = pd.DataFrame(input_data)

    data['date'] = pd.to_datetime(data['date'])
    if data.duplicated(subset='date').any():
        warnings.warn('There are duplicate dates in "{}" dataset:\n{}'.format(
            data.loc[data.index[0], 'name'],
            data.loc[data.duplicated(subset='date'),'date']))
        data.drop_duplicates(subset=['date'], inplace=True)
    data.sort_values(by=['date'], inplace=True)

    data['cash'] = pd.to_numeric(data['cash'])
    data['cash'] = data['cash'].fillna(0.0)

    if 'return' in data.columns:
        data['return'] = pd.to_numeric(data['return'])
        data['return'] = data['return'].fillna(0.0)
    else:
        data['return'] = 0.0

    if 'amount' in data.columns:
        data['amount'] = pd.to_numeric(data['amount'])
        data['amount'] = data['amount'].interpolate(method='pad')

    if 'price' in data.columns:
        data['price'] = pd.to_numeric(data['price'])
    
    if 'value' in data.columns:
        data['value'] = pd.to_numeric(data['value'])

    if ('amount' in data.columns) and ('price' in data.columns):
        if 'value' in data.columns:
            data['value'] = np.where(pd.notna(data['value']), data['value'], data['amount'] * data['price'])
            data['price'] = data['value'] / data['amount']
        else:
            data['value'] = data['amount'] * data['price']

    data['value'] = data['value'].interpolate(method='linear') if 'value' in data else 0.0

    data['cash_invested'] = data['cash'].cumsum()
    data['return_received'] = data['return'].cumsum()
    data['profit'] = data['value'] - data['cash_invested'] + data['return_received']
    data['cash_invested'] = np.where(data['cash_invested'] < 0, 0, data['cash_invested'])
    data['relative_profit'] = (data['profit'] / data['cash_invested']) * 100
    data['relative_profit'] = np.where(np.isinf(data['relative_profit']),
        (data['profit'] / max(data['cash_invested']) * 100), data['relative_profit'])

    return data

def plot_sunburst(input: pd.DataFrame,
        values: str, label_text) -> go.Sunburst:
    dataframe = input.copy()
    dataframe = dataframe.loc[dataframe[values] != 0]  # filter 0 values
    dataframe['rootnode'] = ''
    if type(label_text) is str:
        label_text = [label_text, '']
    dataframe['sign'] = np.where(dataframe[values] > 0,
        label_text[0], label_text[1])

    dataframe['account'] = np.where(dataframe['account'] == ' ',
        dataframe['group'], dataframe['account'])

    # Manipulating names in order to separate positive and negative values
    dataframe['account'] = np.where(dataframe[values] < 0,
        dataframe['account'] + ' ', dataframe['account'])
    dataframe['group'] = np.where(dataframe[values] < 0,
        dataframe['group'] + ' ', dataframe['group'])

    data = dataframe[['name', 'account', values, 'sign', 'color']].copy()
    data.columns = ['label', 'parent', 'value', 'sign', 'color']
    data['value'] = abs(data['value'])

    if dataframe[values].min() < 0:
        path = ['rootnode', 'sign', 'group', 'account', 'name']
    else:
        path = ['rootnode', 'group', 'account', 'name']

    for i in range(1, len(path) - 1):
        d = dataframe.groupby(path[i]).first().reset_index()
        d[values] = dataframe.groupby(path[i]).sum().reset_index()[values]
        d['sign'] = np.where(d[values] > 0, label_text[0], label_text[1])
        d[values] = abs(d[values])
        d = d[[path[i], path[i-1], values, 'sign', 'color']]
        d.columns = ['label', 'parent', 'value', 'sign', 'color']
        # remove rows, where label matches parent
        d = d.drop(d[d.label == d.parent].index)
        data = data.append(d)

    # Set colors for positive and negative sign labels if they exist
    data['color'] = np.where(data['label'] == label_text[0],
        'rgb(0, 255, 0)', data['color'])
    data['color'] = np.where(data['label'] == label_text[1],
        'rgb(255, 0, 0)', data['color'])

    return go.Sunburst(labels=data['label'], parents=data['parent'],
        values=data['value'], customdata=data['sign'],
        marker=dict(colors=data['color']), branchvalues='total', hovertemplate=
            '<b>%{label}</b><br>%{customdata}: %{value:.2f}€<extra></extra>')

def plot_relative_profit_sunburst(input: pd.DataFrame) -> go.Sunburst:
    dataframe = input.copy()
    dataframe['rootnode'] = ''

    dataframe = dataframe.loc[dataframe['relative_profit'] != 0]  # filter 0 values
    dataframe = dataframe.loc[dataframe['relative_profit'] != -100]  # filter -100 values (taxes)
    dataframe['sign'] = np.where(dataframe['relative_profit'] > 0, 'Profit', 'Loss')

    # Manipulating names in order to separate positive and negative values
    dataframe['account'] = np.where(dataframe['account'] == ' ', dataframe['group'], dataframe['account'])
    dataframe['account'] = np.where(dataframe['relative_profit'] < 0, dataframe['account'] + ' ', dataframe['account'])
    dataframe['group'] = np.where(dataframe['relative_profit'] < 0, dataframe['group'] + ' ', dataframe['group'])

    dataframe['relative_profit_sum'] = dataframe['relative_profit']

    data = dataframe[['name', 'account', 'relative_profit', 'relative_profit_sum', 'sign', 'color']].copy()
    data.columns = ['label', 'parent', 'relative_profit', 'relative_profit_sum', 'sign', 'color']
    data['relative_profit_sum'] = abs(data['relative_profit_sum'])

    if dataframe['relative_profit'].min() < 0:
        path = ['rootnode', 'sign', 'group', 'account', 'name']
    else:
        path = ['rootnode', 'group', 'account', 'name']

    for i in range(1, len(path) - 1):
        d = dataframe.groupby(path[i]).first().reset_index()
        d['relative_profit_sum'] = dataframe.groupby(path[i]).sum().reset_index()['relative_profit_sum']
        d['sign'] = np.where(d['relative_profit_sum'] > 0, 'Profit', 'Loss')
        d['relative_profit_sum'] = abs(d['relative_profit_sum'])
        d['relative_profit'] = (dataframe.groupby(path[i]).sum().reset_index()['profit'] / dataframe.groupby(path[i]).sum().reset_index()['cash_invested']) * 100
        d = d[[path[i], path[i-1], 'relative_profit', 'relative_profit_sum', 'sign', 'color']]
        d.columns = ['label', 'parent', 'relative_profit', 'relative_profit_sum', 'sign', 'color']
        d = d.drop(d[d.label == d.parent].index)  # remove rows, where label matches parent
        data = data.append(d)

    # Set colors for positive and negative sign labels if they exist
    data['color'] = np.where(data['label'] == 'Profit', 'rgb(0, 255, 0)', data['color'])
    data['color'] = np.where(data['label'] == 'Loss', 'rgb(255, 0, 0)', data['color'])

    return go.Sunburst(labels=data['label'], parents=data['parent'], values=data['relative_profit_sum'], customdata=data['relative_profit'], marker=dict(colors=data['color']),
        branchvalues='total', hovertemplate =
            "<b>%{label}</b><br>" +
            "Relative net profit: %{customdata:.2f}%<extra></extra>")

def plot_historical_data(dataframe: pd.DataFrame, values: str, label_text: str) -> go.Figure:
    value_by_date = dataframe.pivot(index='date', columns='name', values=values)

    fig = go.Figure()

    value_by_date_sum = value_by_date.sum(axis = 1, skipna = True)
    fig.add_trace(go.Scatter(x=value_by_date_sum.index, y=value_by_date_sum.values, mode='lines+markers', name='Total',
        marker=dict(color=cyan), hovertemplate =
            "%{x|%B %Y}<br>" +
            "<b>Total " + label_text.lower() + ":</b> %{y:.2f}€<extra></extra>"))

    for a in asset_properties.index:
        if not ((value_by_date[a].nunique() == 1) and (value_by_date[a].sum() == 0)):
            # only plot traces with at least one non-zero value
            fig.add_trace(go.Bar(x=value_by_date.index, y=value_by_date[a], name=a, customdata=np.dstack(a), marker=dict(color=asset_properties.loc[a, 'color']),
                hovertemplate =
                    "<b>%{name}</b><br>" +
                    "%{x|%B %Y}<br>" + label_text +
                    ": %{y:.2f}€<extra></extra>"))

    fig.update_layout(barmode='relative')
    six_months = [value_by_date.index[-1] - datetime.timedelta(days=(365/2 - 15)), value_by_date.index[-1] + datetime.timedelta(days=15)]
    fig.update_xaxes(range=six_months)
    fig.update_yaxes(ticksuffix="€")
    configure_historical_dataview(fig)

    return fig

def plot_historical_relative_profit(dataframe: pd.DataFrame) -> go.Figure:
    value_by_date = dataframe.pivot(index='date', columns='name', values='relative_profit')
    value_by_date = value_by_date.drop(value_by_date.columns[value_by_date.max() == -100], axis=1)

    fig = go.Figure()

    overall = dataframe.groupby('date').sum()[['profit', 'cash_invested']]
    overall['values'] = (overall['profit'] / overall['cash_invested']) * 100
    fig.add_trace(go.Scatter(x=overall.index, y=overall['values'], mode='lines+markers', name='Total',
        marker=dict(color=cyan), hovertemplate =
            "%{x|%B %Y}<br>"+
            "<b>Total profit:</b> %{y:.2f}%<extra></extra>"))

    for a in asset_properties.index:
        if a in value_by_date.columns:
            fig.add_trace(go.Bar(x=value_by_date.index, y=value_by_date[a], name=a, customdata=[a], marker=dict(color=asset_properties.loc[a,'color']),
                hovertemplate =
                    "<b>%{customdata[0]:s}</b><br>"+
                    "%{x|%B %Y}<br>"+
                    "Relative net profit: %{y:.2f}%<extra></extra>"))

    fig.update_layout(barmode='group')
    six_months = [value_by_date.index[-1] - datetime.timedelta(days=(365/2 - 15)), value_by_date.index[-1] + datetime.timedelta(days=15)]
    fig.update_xaxes(range=six_months)
    fig.update_yaxes(ticksuffix="%")
    configure_historical_dataview(fig)

    return fig

def plot_historical_asset_data(input: pd.DataFrame) -> go.Figure:
    data = input.copy()
    fig = go.Figure()

    data['value_and_return'] = data['cash_invested'] + data['profit']
    data['red_fill'] = np.where(
        data['cash_invested'] > data['return_received'],
        data['cash_invested'], data['return_received'])

    fig.add_trace(go.Scatter(x=data['date'], y=data['red_fill'],
        fill='tozeroy', mode='none', fillcolor='rgba(255,0,0,0.7)',
        hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=data['date'], y=data['cash_invested'],
        mode='none',
        hovertemplate = "Cash invested: %{y:.2f}€<extra></extra>"))
    data['f_profit'] = data['profit'].map('{:,.2f}€ / '.format)
    data['f_relative_profit'] = data['relative_profit'].map('{:,.2f}%'.format)
    data['profit_string'] = data['f_profit'] + data['f_relative_profit']
    
    fig.add_trace(go.Scatter(x=data['date'], y=data['value_and_return'], fill='tozeroy', mode='none', fillcolor='rgba(0,255,0,0.7)',
        customdata=data['profit_string'],
        hovertemplate =
            "Net profit: %{customdata}<extra></extra>"))

    blue_fill_mode = 'tozeroy'
    if max(data['return_received']) > 0:
        fig.add_trace(go.Scatter(x=data['date'], y=data['return_received'], fill='tozeroy', mode='none', fillcolor='rgba(0,0,0,0)',
            hovertemplate =
                "Return received: %{y:.2f}€<extra></extra>"))
        blue_fill_mode = 'tonexty'

    blue_fill = data[['date', 'red_fill', 'value_and_return']].copy()
    blue_fill.index *= 2
    blue_fill['y'] = np.where(blue_fill['red_fill'] < blue_fill['value_and_return'], blue_fill['red_fill'], data['value_and_return'])
    blue_fill['profitable'] = (blue_fill['y'] == blue_fill['red_fill'])
    mask = blue_fill.iloc[:-1]['profitable'] ^ blue_fill['profitable'].shift(-1)
    intermediate_values = blue_fill[mask].copy()
    intermediate_values.index += 1
    intermediate_values['y'] = np.nan
    blue_fill = blue_fill.append(intermediate_values).sort_index().reset_index(drop=True)
    blue_fill['slope'] = abs(blue_fill['value_and_return'] - blue_fill['red_fill']) / (abs(blue_fill['value_and_return'] - blue_fill['red_fill']) + abs(blue_fill.shift(-1)['value_and_return'] - blue_fill.shift(-1)['red_fill']))
    blue_fill['date'] = np.where(pd.isna(blue_fill['y']), (blue_fill.shift(-1)['date'] - blue_fill['date']) * blue_fill['slope'] + blue_fill['date'], blue_fill['date'])
    blue_fill['y'] = np.where(pd.isna(blue_fill['y']), (blue_fill.shift(-1)['red_fill'] - blue_fill['red_fill']) * blue_fill['slope'] + blue_fill['red_fill'], blue_fill['y'])
    fig.add_trace(go.Scatter(x=blue_fill['date'], y=blue_fill['y'], fill=blue_fill_mode, mode='none', fillcolor='rgba(0,0,255,0.5)',
        hoverinfo='skip'))

    if max(data['value']) > 0:
        fig.add_trace(go.Scatter(x=data['date'], y=data['value'], mode='lines', line=dict(color='yellow'),
            hovertemplate =
                "Value: %{y:.2f}€<extra></extra>"))
    fig.update_layout(hovermode='x', showlegend=False)
    fig.update_layout(hoverlabel=dict(bgcolor='white'))
    configure_historical_dataview(fig)
    one_year = [latest_date - datetime.timedelta(days=365), latest_date]
    fig.update_xaxes(range=one_year, rangeslider=dict(visible=True))
    fig.update_yaxes(ticksuffix='€')
    return fig

def plot_yearly_asset_data(data: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    yearly_data = data.groupby(data['date'].dt.year).head(1)
    yearly_data = yearly_data.append(data.iloc[-1])
    yearly_data['date'] = yearly_data['date'].dt.year - 1
    
    yearly_data['value_change'] = yearly_data['profit'] - yearly_data['return_received']
    yearly_data.loc[yearly_data.index[0], 'value_change'] = 0
    yearly_data['value_change'] = yearly_data['value_change'].diff()
    yearly_data.loc[yearly_data.index[0], 'return_received'] = 0
    yearly_data['return_received'] = yearly_data['return_received'].diff()
    if yearly_data.index[-1] == yearly_data.index[-2]:
        # if two last rows are the same, drop one of them
        yearly_data.drop(yearly_data.tail(1).index, inplace=True)
    else:
        # otherwise, set back year of last row
        yearly_data.loc[yearly_data.index[-1], 'date'] += 1
    yearly_data.drop(yearly_data.head(1).index, inplace=True)

    fig.add_trace(go.Bar(x=yearly_data['date'], y=yearly_data['return_received'], marker=dict(color='rgb(73,200,22)'),
        width=[0.5] if (len(yearly_data) == 1) else None,
        hovertemplate =
            "<b>%{x}</b><br>" +
            "Return received: %{y:.2f}€<extra></extra>"))
    fig.add_trace(go.Bar(x=yearly_data['date'], y=yearly_data['value_change'], marker=dict(color='rgb(36,99,139)'),
        width=[0.5] if (len(yearly_data) == 1) else None,
        hovertemplate =
            "<b>%{x}</b><br>" +
            "Value change: %{y:+.2f}€<extra></extra>"))

    fig.update_layout(barmode='relative', showlegend=False)
    fig.update_xaxes(type='category', fixedrange=True)
    fig.update_yaxes(ticksuffix="€", fixedrange=True)
    fig.update_layout(margin=dict(l=10, r=10, t=35, b=10))

    return fig

def configure_historical_dataview(figure: go.Figure):
    figure.update_xaxes(type='date', rangeslider_visible=False,
        rangeselector=dict( x=0, buttons=list([
            dict(count=6, label="6M", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1Y", step="year", stepmode="backward"),
            dict(count=2, label="3Y", step="year", stepmode="backward"),
            dict(label="ALL", step="all")])))
    figure.update_layout(margin=dict(l=10, r=10, t=20, b=10))
    figure.update_layout(xaxis=dict(title=dict(text='')), yaxis=dict(title=dict(text='')))

def append_figures(statistic: str, label_text) -> str:

    if statistic == 'relative_profit':
        sunburst = go.Figure(plot_relative_profit_sunburst(current_stats))
    else:
        sunburst = go.Figure(plot_sunburst(current_stats, statistic, label_text))

    sunburst = sunburst.update_traces(insidetextorientation='radial')
    sunburst.update_layout(margin=dict(l=10, r=10, t=10, b=10))

    if type(label_text) is list:
        label_text = label_text[0]

    if statistic == 'relative_profit':
        historical = plot_historical_relative_profit(assets)
    else:
        historical = plot_historical_data(assets, statistic, label_text)

    return Html.columns([Html.Column(width=30, content=sunburst.to_html(full_html=False, include_plotlyjs=True)),
        Html.Column(content=historical.to_html(full_html=False, include_plotlyjs=True))])

def append_asset_data_view(data: pd.DataFrame):

    name = data.loc[data.index[-1], 'name']
    if 'symbol' in data.columns:
        if pd.notnull(data.loc[data.index[-1], 'symbol']):
            name += ' ({})'.format(data.loc[data.index[-1], 'symbol'])
    if 'account' in data.columns:
        name += '<br>{}'.format(data.loc[data.index[-1], 'account'])
    output = '<h2>' + name + '</h2>'

    statistics = [Html.label('Value', Html.Value(data.loc[data.index[-1], 'value'], '€'))]
    statistics.append(Html.label('Funds invested', Html.Value(data.loc[data.index[-1], 'cash_invested'], '€')))
    if data.loc[data.index[-1], 'return_received'] != 0:
        statistics.append(Html.label('Return received', Html.Value(data.loc[data.index[-1], 'return_received'], '€')))
    statistics.append(Html.label('Net profit', Html.Value(data.loc[data.index[-1], 'profit'], '€').color()))
    statistics.append(Html.label('Relative net profit', Html.Value(data.loc[data.index[-1], 'relative_profit'], '%').color()))
    output += Html.columns(statistics)
    if (len(data) > 1):
        output += Html.columns([Html.Column(width=30, content=plot_yearly_asset_data(data).to_html(full_html=False, include_plotlyjs=True)),
            Html.Column(content=plot_historical_asset_data(data).to_html(full_html=False, include_plotlyjs=True))])
    return output

def append_overall_data_tabs(document: dominate.document):
    tabs = [Html.Tab(Html.label('Value', Html.Value(current_stats['value'].sum(), '€')),
        append_figures('value', 'Value'), checked=True)]
    tabs.append(Html.Tab(Html.label('Funds invested', Html.Value(current_stats['cash_invested'].sum(), '€')),
        append_figures('cash_invested', 'Funds invested')))
    if current_stats['return_received'].sum() > 0:
        tabs.append(Html.Tab(Html.label('Return received', Html.Value(current_stats['return_received'].sum(), '€')),
            append_figures('return_received', 'Return received')))
    tabs.append(Html.Tab(Html.label('Net profit', Html.Value(current_stats['profit'].sum(), '€').color()),
        append_figures('profit', ['Profit', 'Loss'])))
    tabs.append(Html.Tab(Html.label('Relative net profit', Html.Value((current_stats['profit'].sum()/current_stats['cash_invested'].sum())*100, '%').color()),
        append_figures('relative_profit', ['Profit', 'Loss'])))

    document += raw(Html.tab_container(tabs))

def append_asset_data_tabs(document: dominate.document):

    groups = assets['group'].unique()

    tabs = []
    for g in groups:
        group_assets = assets.loc[assets['group'] == g, 'name'].unique()
        content = ''
        if len(group_assets) > 1:
            group_total = assets[assets['group'] == g].groupby('date').sum().reset_index()
            group_total['name'] = '{} Total'.format(g)
            group_total['relative_profit'] = (group_total['profit'] / group_total['cash_invested']) * 100
            content += append_asset_data_view(group_total)
        for a in group_assets:
            content += append_asset_data_view(assets[assets['name'] == a])
        tabs.append(Html.Tab(Html.label(g), content))
    document += raw(Html.tab_container(tabs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( 'input_dir', type = str, nargs = '?', default = None )
    arguments = parser.parse_args()

    input_directory = input_directory = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + 'input_data'
    if arguments.input_dir != None:
        input_directory = arguments.input_dir

    assets = pd.DataFrame()
    for entry in os.scandir(input_directory):
        if entry.is_file() and (entry.path.endswith(".yaml") or entry.path.endswith(".yml")):
            with open( entry, 'r' ) as read_file:
                input = yaml.load( read_file, Loader=yaml.BaseLoader )
            ids = pd.DataFrame(input).drop(columns=['data'])
            data = pd.DataFrame(input['data']).join(ids)
            data = process_data(data)
            assets = assets.append(data)

    assets['account'] = assets['account'].fillna(' ')  # empty string doesn't work
    group_by_name = assets.groupby('name')[['name', 'group', 'account', 'value']].tail(1)
    group_by_name.sort_values(by=['value'], inplace=True, ascending=False)
    group_by_name = group_by_name.drop(columns=['value'])

    asset_properties = pd.DataFrame()
    index = 0
    for _, group in group_by_name.groupby('group'):
        boundary_colors = [px.colors.qualitative.Set1[index], px.colors.qualitative.Pastel1[index]]
        c = px.colors.n_colors(boundary_colors[0], boundary_colors[1], max(len(group.index), 4), colortype='rgb')
        c = c[:len(group.index)]
        group['color'] = c
        asset_properties = asset_properties.append(group)
        index += 1

    assets = assets.merge(asset_properties)
    asset_properties = asset_properties.sort_values(by=['group', 'name'])
    asset_properties = asset_properties.set_index('name')
    print(asset_properties)

    color_map = asset_properties.to_dict()['color']

    earliest_date = assets.groupby('date')['date'].head().iloc[0]
    latest_date = assets.groupby('date')['date'].tail().iloc[-1]

    current_stats = assets.groupby('name')[['name', 'symbol', 'group', 'account', 'cash_invested', 'return_received', 'value', 'profit', 'relative_profit', 'color', 'date']].tail(1)
    current_stats = current_stats[current_stats['date'] > latest_date - pd.DateOffset(months=6)]  # TODO: take months value from settings

    print(current_stats)
    print(assets[assets.duplicated()])

    d = dominate.document()
    d.head += raw('<meta charset="utf-8"/>')
    d.head += raw('<link rel="stylesheet" href="style.css">')
    d.head += raw('<title>Your investment portfolio</title>')
    d += raw('<h1>Your investment portfolio</h1>')
    d += raw('<h3>Data from ' + earliest_date.strftime('%Y-%m-%d') + ' to ' + latest_date.strftime('%Y-%m-%d') + '</h3>')
    append_overall_data_tabs(d)
    append_asset_data_tabs(d)

    d += raw('<p class="link">Report generated on ' + datetime.date.today().strftime('%Y-%m-%d') + ', using open source script: <a href="https://github.com/zukaitis/investment-tracker/">Investment Tracker</a></p>')

    with open('report.html', 'w') as f:
        print(d, file=f)
