#!/usr/bin/python3

import _html as html

import os
import yaml
import argparse
import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io
import yfinance as yf
from dataclasses import dataclass
import dataclasses
import warnings
import babel.numbers
import babel.dates

cyan = px.colors.qualitative.Plotly[5]

colors_light = dict(
    background_color='#e5ecf6',
    text_color='#555555',
    tab_background_color='#ffffff',
    tab_shadow_color='#f0f0f0',
    checked_tab_indicator_color='#00ccee',
    hover_tab_indicator_color='#c6fbff')

colors_dark = dict(
    background_color='#283442',
    text_color='#999999',
    tab_background_color='#111111',
    tab_shadow_color='#202020',
    checked_tab_indicator_color='#00ccee',
    hover_tab_indicator_color='#11363c')

def print_warning(message: str):
    warnings.warn(f'WARNING: {message}')

class Settings:
    owner = 'Your'
    currency = 'EUR'
    locale = 'en_US_POSIX'
    autofill_interval = '1d'
    autofill_price_mark = 'Close'
    theme = 'auto'

    def __setattr__(self, name, value):
        if name == 'owner':
            self.__dict__[name] = f"{value}'s"
        elif name == 'currency':
            try:
                yf.Ticker(f'{value}=X').info
            except ValueError:
                print_warning(f'Unknown currency - "{value}"')
            else:
                self.__dict__[name] = value
        elif name == 'locale':
            if babel.localedata.exists(value):
                self.__dict__[name] = value
            else:
                print_warning(f'Unknown locale - "{value}"')
        elif name == 'autofill_interval':
            allowed = ['1d', '5d', '1wk', '1mo', '3mo']
            if value in allowed:
                self.__dict__[name] = value
            else:
                print_warning(f'Unknown interval - "{value}". Allowed intervals: {allowed}')
        elif name == 'autofill_price_mark':
            allowed = ['Open', 'Close', 'High', 'Low']
            if value in allowed:
                self.__dict__[name] = value
            else:
                print_warning(f'Unknown price mark - "{value}". Allowed marks: {allowed}')
        elif name == 'theme':
            allowed = ['light', 'dark', 'auto']
            if value in allowed:
                self.__dict__[name] = value
            else:
                print_warning(f'Unknown theme - "{value}". Allowed themes: {allowed}')
        elif name not in self:
            print_warning(f'No such setting - "{name}"')
        else:
            self.__dict__[name] = value

    def __iter__(self):
        variables = [d for d in dir(self) if not d.startswith('_')]
        return iter(variables)

def currency_str(value: float) -> str:
    if np.isnan(value) or np.isinf(value):
        return '?'
    precision = 4 
    if abs(value) >= 0.50:
        precision = 2
    elif abs(value) >= 0.050:
        precision = 3
    return babel.numbers.format_currency(round(value, precision), settings.currency,
        locale=settings.locale, decimal_quantization=False)

def percentage_str(value: float) -> str:
    if np.isnan(value) or np.isinf(value):
        return '?'
    precision = 4
    if abs(value) >= 0.5:  # 50%
        precision = 2
    elif abs(value) >= 0.05:  # 5.0%
        precision = 3
    return babel.numbers.format_percent(round(value, precision), locale=settings.locale,
        decimal_quantization=False)

def decimal_str(value: float, precision: int = None) -> str:
    if np.isnan(value) or np.isinf(value):
        return '?'
    if precision != None:
        value = round(value, precision)
    return babel.numbers.format_decimal(value, locale=settings.locale)

def date_str(date: datetime.date) -> str:
    return babel.dates.format_date(date, locale=settings.locale)

def currency_symbol() -> str:
    return babel.numbers.get_currency_symbol(settings.currency, locale=settings.locale)

def currency_tick_prefix() -> str:
    if babel.numbers.Locale.parse(
            settings.locale).currency_formats['standard'].pattern.startswith('¤'):
        if babel.numbers.Locale.parse(
            settings.locale).currency_formats['standard'].pattern.startswith('¤\xa0'):
            return f'{currency_symbol()} '  # add space after the symbol
        return f'{currency_symbol()}'
    else:
        return None

def currency_tick_suffix() -> str:
    if babel.numbers.Locale.parse(
            settings.locale).currency_formats['standard'].pattern.endswith('¤'):
        if babel.numbers.Locale.parse(
            settings.locale).currency_formats['standard'].pattern.endswith('\xa0¤'):
            return f' {currency_symbol()}'  # add space before the symbol
        return f'{currency_symbol()}'
    else:
        return None

def percentage_tick_suffix() -> str:
    string = percentage_str(777)  # checking generated string of random percentage
    if string[-2] == ' ':
        return ' %'
    return '%'

def autofill(input_data: pd.DataFrame) -> pd.DataFrame:
    data = input_data.copy()

    symbol = input_data.loc[input_data.index[0], 'symbol']
    ticker = yf.Ticker(symbol)
    print(f'Fetching yfinance data for {symbol}')
    # download extra data, just to be sure, that the requred date will appear on yf dataframe
    start_date = input_data.loc[input_data.index[0], 'date'] - datetime.timedelta(days=7)  
    yfdata = ticker.history(start=start_date, interval=settings.autofill_interval)
    if ticker.info['currency'] != settings.currency:
        # convert currency, if it differs from the one selected in settings
        currency_ticker = yf.Ticker(f"{ticker.info['currency']}{settings.currency}=X")
        currency_rate = currency_ticker.history(start=start_date,
            interval=settings.autofill_interval)
        yfdata[settings.autofill_price_mark] *= currency_rate[settings.autofill_price_mark]
        yfdata['Dividends'] *= currency_rate[settings.autofill_price_mark]

    yfdata = yfdata.reset_index().rename(columns={'Date':'date',
        settings.autofill_price_mark:'price'})

    data = pd.merge(data, yfdata, on='date', how='outer').sort_values(by=['date'])

    if 'return_tax' not in data.columns:
        data['return_tax'] = 0.0
    data['return_tax'] = pd.to_numeric(data['return_tax']).interpolate(method='pad').fillna(0.0)

    for p in ['name', 'symbol', 'account', 'group']:
        if p in data.columns:
            data[p] = input_data.loc[input_data.index[0], p]

    data['amount'] = pd.to_numeric(data['amount']).interpolate(method='pad').fillna(0.0)
    data['price'] = data['price'].interpolate(method='pad')
    data['investment'] = pd.to_numeric(data['investment']).fillna(0.0)
    data['value'] = data['amount'] * data['price']
    if 'return' not in data.columns:
        data['return'] = data['amount'] * data['Dividends'].fillna(0.0) * (1 - data['return_tax'])

    return data

def process_data(input_data, discard_zero_values: bool = True) -> pd.DataFrame:
    if type(input_data) is pd.DataFrame:
        data = input_data.copy()
    else:
        data = pd.DataFrame(input_data)

    data['date'] = pd.to_datetime(data['date'])
    if data.duplicated(subset='date').any():
        dataset_name = data.loc[data.index[0], 'name']
        duplicates = data.loc[data.duplicated(subset='date'), 'date']
        print_warning(f'There are duplicate dates in "{dataset_name}" dataset:\n{duplicates}')
        data.drop_duplicates(subset=['date'], inplace=True)
    data.sort_values(by=['date'], inplace=True)

    if (('investment' in data.columns) and ('amount' in data.columns) and 
        ('price' not in data.columns) and ('value' not in data.columns)):
        data = autofill(data)
    else:  # process data the old way. TODO: move this to separate method
        if 'investment' in data.columns:
            data['investment'] = pd.to_numeric(data['investment'])
            data['investment'] = data['investment'].fillna(0.0)
        else:
            data['investment'] = 0.0

        if 'amount' in data.columns:
            data['amount'] = pd.to_numeric(data['amount'])
            data['amount'] = data['amount'].interpolate(method='pad')

        if 'price' in data.columns:
            data['price'] = pd.to_numeric(data['price'])
            data['price'] = data['price'].interpolate(method='linear')
        
        if 'value' in data.columns:
            data['value'] = pd.to_numeric(data['value'])

        if ('amount' in data.columns) and ('price' in data.columns):
            if 'value' in data.columns:
                data['value'] = np.where(pd.notna(data['value']),
                    data['value'], data['amount'] * data['price'])
            else:
                data['value'] = data['amount'] * data['price']

        if 'value' in data.columns:
            data['value'] = data['value'].interpolate(method='linear')
        else:
            data['value'] = 0.0
        data['value'] = data['value'].fillna(0.0)

    if 'return' in data.columns:
        data['return'] = pd.to_numeric(data['return'])
        data['return'] = data['return'].fillna(0.0)
    else:
        data['return'] = 0.0

    nonzero_after_zero_mask = ((data['value'] != 0) & (data['value'].shift(1) == 0))
    zero_mask = (data['value'] == 0) & (data['investment'] == 0)
    data['period'] = np.where(nonzero_after_zero_mask, 1, np.nan)
    data['period'] = data['period'].cumsum().interpolate(method='pad')
    data['period'] = data['period'].fillna(0)

    if discard_zero_values:
        data.drop(data[zero_mask].index, inplace=True)

    data = data.assign(return_received=0, net_investment=0, net_investment_max=0)
    for p in data['period'].unique():
        mask = (data['period'] == p)
        data.loc[mask, 'return_received'] = data.loc[mask, 'return'].cumsum()
        data.loc[mask, 'net_investment'] = data.loc[mask, 'investment'].cumsum()
        data.loc[mask, 'net_investment_max'] = data.loc[mask, 'net_investment'].cummax()

    data['total_return_received'] = data['return'].cumsum()
    data['profit'] = data['value'] - data['net_investment'] + data['return_received']
    data['net_investment'] = np.where(data['net_investment'] < 0, 0, data['net_investment'])
    data['relative_profit'] = (data['profit'] / data['net_investment_max'])

    return data

def calculate_monthly_values(input: pd.DataFrame) -> pd.DataFrame:
    daily = input.copy()
    daily['year'] = daily['date'].dt.year
    daily['month'] = daily['date'].dt.month
    group_by_month = daily.groupby(['id', 'year', 'month'])

    monthly = group_by_month.agg({'date': 'last', 'value': 'last', 'net_investment': 'last',
        'net_investment_max': 'last', 'return': 'sum', 'profit':'last',
        'relative_profit': 'last'}).reset_index()
    
    monthly['date'] = monthly['date'].map(lambda x: x.replace(day=1))
    monthly.drop(columns=['year', 'month'], inplace=True)

    return monthly

def fraction_string(input: pd.DataFrame) -> pd.Series:
    data = input.copy()
    data = data.assign(percentage_in_profitability='', of_profitability='', separator='',
        of_group='')

    data['percentage_in_profitability'] = np.where(data['fraction_in_profitability'] < 1,
        '<br>' + data['fraction_in_profitability'].apply(percentage_str), '')
    if data['profitability'].nunique() > 1:  # if values are separated by profitability
        data['of_profitability'] = np.where(data['fraction_in_profitability'] < 1,
            ' of ' + data['profitability'], '')
        data['separator'] = '<br>'
    else:
        data['separator'] = ' / '
    data['separator'] = np.where(data['fraction_in_group'] < 1, data['separator'], '')
    data['of_group'] = np.where(data['fraction_in_group'] < 1,
        data['fraction_in_group'].apply(percentage_str) + ' of ' + data['group'], '')
    # don't display fraction in group, when it matches fraction in profitability
    data.loc[data['fraction_in_profitability'] == data['fraction_in_group'],
        ['separator', 'of_group']] = [['', '']]
        
    return (data['percentage_in_profitability'] + data['of_profitability'] + data['separator']
        + data['of_group'])

def plot_sunburst(input: pd.DataFrame, values: str, label_text: str):
    data = input[input[values] != 0].copy()  # filter 0 values
    relevant_columns = ['group', 'account', 'name', values, 'color']
    if values == 'relative_profit':
        relevant_columns += ['profit', 'net_investment_max']
    data = data[relevant_columns]  # leave only relevant columns
    data['profitability'] = np.where(data[values] > 0, 'Profitable', 'Unprofitable')
    # move name in place of account to avoid empty rings in graph
    data.loc[data['account'] == ' ', ['account', 'name']] = np.array([
        data.loc[data['account'] == ' ', 'name'], None], dtype='object')

    data[['profitability_sum', 'group_sum']] = 0
    for p in data['profitability'].unique():
        p_subset = (data['profitability'] == p)
        data.loc[p_subset, 'profitability_sum'] = data.loc[p_subset, values].sum()
        for g in data.loc[p_subset, 'group'].unique():
            g_subset = p_subset & (data['group'] == g)
            data.loc[g_subset, 'group_sum'] = data.loc[g_subset, values].sum()
    
    graph_data = pd.DataFrame(columns=['label', 'id', 'parent', 'value', 'color'])
    graph_data[['label', 'id', 'parent']] = ''
    tree = ['profitability', 'group', 'account', 'name']
    if values in ['value', 'net_investment', 'return_received']:
        tree.remove('profitability')  # don't separate these graphs by profitability

    for t in range(len(tree)):
        d = data.groupby(tree[:t+1]).first().reset_index()
        d[values] = data.groupby(tree[:t+1]).sum().reset_index()[values]
        if values == 'relative_profit':
            d[['profit', 'net_investment_max']] = data.groupby(
                tree[:t+1]).sum().reset_index()[['profit', 'net_investment_max']]
            d['displayed_fraction'] = d['profit'] / d['net_investment_max']
            d['display_string'] = d['displayed_fraction'].apply(percentage_str)
        else:
            d['fraction_in_group'] = d[values] / d['group_sum']
            d['fraction_in_profitability'] = d[values] / d['profitability_sum']
            d['display_string'] = d[values].apply(currency_str) + fraction_string(d)
        d[values] = abs(d[values])
        d['parent'] = ''
        for i in range(t):
            d['parent'] += d[tree[i]]
        d['id'] = d['parent'] + d[tree[t]]
        d = d[[tree[t], 'id', 'parent', values, 'display_string', 'color']]
        d.columns = ['label', 'id', 'parent', 'value', 'display_string', 'color']
        graph_data = graph_data.append(d)

    # Set colors of Profitable and Unprofitable labels if they exist
    graph_data.loc[graph_data['label'] == 'Profitable', 'color'] = 'green'
    graph_data.loc[graph_data['label'] == 'Unprofitable', 'color'] = 'red'

    return go.Sunburst(labels=graph_data['label'], ids=graph_data['id'],
        parents=graph_data['parent'], values=graph_data['value'],
        customdata=graph_data['display_string'], marker=dict(colors=graph_data['color']),
        branchvalues='total', hovertemplate=
            f'<b>%{{label}}</b><br>{label_text}: %{{customdata}}<extra></extra>')

def plot_historical_data(dataframe: pd.DataFrame, values: str, label_text: str) -> go.Figure:
    value_by_date = dataframe.pivot(index='date', columns='id', values=values)
    if values != 'profit':
        value_by_date.interpolate(method='pad', inplace=True)
    str_value_by_date = value_by_date.applymap(currency_str)

    fig = go.Figure()

    value_by_date_sum = value_by_date.sum(axis = 1, skipna = True)
    str_value_by_date_sum = value_by_date_sum.apply(currency_str)
    fig.add_trace(go.Scatter(x=value_by_date_sum.index, y=value_by_date_sum.values,
        mode='lines+markers', name='Total', marker=dict(color=cyan),
        customdata=np.transpose(str_value_by_date_sum.values), hovertemplate=(
            f'%{{x|%B %Y}}<br>'
            f'<b>Total {label_text.lower()}:</b> %{{customdata}}<extra></extra>')))

    for a in value_by_date.columns:
        if any(value_by_date[a] != 0):  # only plot traces with at least one non-zero value
            name = asset_properties.loc[a, 'name']
            fig.add_trace(go.Bar(x=value_by_date.index, y=value_by_date[a], name=name,
                marker=dict(color=asset_properties.loc[a, 'color']),
                customdata=np.transpose(str_value_by_date[a]), hovertemplate=(
                    f'%{{x|%B %Y}}<br>'
                    f'<b>{name}</b><br>'
                    f'{label_text}: %{{customdata}}<extra></extra>')))

    fig.update_layout(barmode='relative')
    six_months = [value_by_date.index[-1] - datetime.timedelta(days=(365/2 - 15)), value_by_date.index[-1] + datetime.timedelta(days=15)]
    fig.update_xaxes(range=six_months)
    fig.update_yaxes(ticksuffix=currency_tick_suffix(), tickprefix=currency_tick_prefix())
    configure_historical_dataview(fig, latest_date - value_by_date.index[0])

    return fig

def plot_historical_return(dataframe: pd.DataFrame, label_text: str) -> go.Figure:
    value_by_date = dataframe.pivot(index='date', columns='id', values='return').fillna(0.0)
    value_by_date_cumulative = value_by_date.cumsum()
    str_value_by_date = value_by_date.applymap(currency_str)
    str_value_by_date_cumulative = value_by_date_cumulative.applymap(currency_str)

    fig = go.Figure()

    value_by_date_sum = value_by_date.sum(axis=1, skipna=True).cumsum()
    str_value_by_date_sum = value_by_date_sum.apply(currency_str)
    fig.add_trace(go.Scatter(x=value_by_date_sum.index, y=value_by_date_sum.values,
        mode='lines+markers', name='Total', marker=dict(color=cyan), yaxis='y2',
        customdata=np.transpose(str_value_by_date_sum.values), hovertemplate=(
            f'%{{x|%B %Y}}<br>'
            f'<b>Total return received:</b> %{{customdata}}<extra></extra>')))

    for a in value_by_date.columns:
        if any(value_by_date[a] != 0):  # only plot traces with at least one non-zero value
            name = asset_properties.loc[a, 'name']
            fig.add_trace(go.Bar(x=value_by_date.index, y=value_by_date[a], name=name,
                customdata=np.transpose([str_value_by_date[a], str_value_by_date_cumulative[a]]),
                marker=dict(color=asset_properties.loc[a, 'color']), hovertemplate=(
                    f'%{{x|%B %Y}}<br>' 
                    f'<b>{name}</b><br>'
                    f'{label_text}: %{{customdata[0]}}<br>'
                    f'Total return received: %{{customdata[1]}}<extra></extra>')))

    fig.update_yaxes(ticksuffix=currency_tick_suffix(), tickprefix=currency_tick_prefix())
    fig.update_layout(yaxis2=dict(title='',
        titlefont=dict(color='cyan'), tickfont=dict(color='cyan'),
        ticksuffix=currency_tick_suffix(), tickprefix=currency_tick_prefix(), side='right', overlaying='y',
        range=[0, max(value_by_date_sum.values) * 1.05], fixedrange=True),legend=dict(x=1.1))
    fig.update_layout(barmode='group')
    six_months = [value_by_date.index[-1] - datetime.timedelta(days=(365/2 - 15)), value_by_date.index[-1] + datetime.timedelta(days=15)]
    fig.update_xaxes(range=six_months)
    configure_historical_dataview(fig, latest_date - value_by_date.index[0])

    return fig

def plot_historical_relative_profit(dataframe: pd.DataFrame) -> go.Figure:
    value_by_date = dataframe.pivot(index='date', columns='id', values='relative_profit')
    value_by_date = value_by_date.drop(value_by_date.columns[value_by_date.max() == -1], axis=1)
    str_value_by_date = value_by_date.applymap(percentage_str)

    fig = go.Figure()

    overall = dataframe.groupby('date').sum()[['profit', 'net_investment_max']]
    overall['values'] = (overall['profit'] / overall['net_investment_max'])
    overall['strings'] = overall['values'].apply(percentage_str)
    fig.add_trace(go.Scatter(x=overall.index, y=overall['values']*100, mode='lines+markers',
        name='Total', marker=dict(color=cyan), customdata=overall['strings'], hovertemplate=(
            f'%{{x|%B %Y}}<br>'
            f'<b>Total profit:</b> %{{customdata}}<extra></extra>')))

    for a in value_by_date.columns:
        if any(value_by_date[a] != 0):  # only plot traces with at least one non-zero value
            name = asset_properties.loc[a, 'name']
            fig.add_trace(go.Bar(x=value_by_date.index, y=value_by_date[a]*100, name=name,
                marker=dict(color=asset_properties.loc[a,'color']),
                customdata=np.transpose(str_value_by_date[a]), hovertemplate=(
                    f'%{{x|%B %Y}}<br>'
                    f'<b>{name}</b><br>'
                    f'Relative net profit: %{{customdata}}<extra></extra>')))

    fig.update_layout(barmode='group')
    six_months = [value_by_date.index[-1] - datetime.timedelta(days=(365/2 - 15)), value_by_date.index[-1] + datetime.timedelta(days=15)]
    fig.update_xaxes(range=six_months)
    fig.update_yaxes(ticksuffix=percentage_tick_suffix())
    configure_historical_dataview(fig, latest_date - value_by_date.index[0])

    return fig

def plot_historical_asset_data(input: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    for p in input['period'].unique():
        data = input[input['period'] == p].copy()
        data['value_and_return'] = data['net_investment'] + data['profit']
        data['red_fill'] = np.where(
            data['net_investment'] > data['return_received'],
            data['net_investment'], data['return_received'])

        data['str_net_investment'] = data['net_investment'].apply(currency_str)
        data['str_return_received'] = data['return_received'].apply(currency_str)
        data['str_value'] = data['value'].apply(currency_str)
        if ((data['price'] != 0) & (pd.notna(data['price']))).any():
            data['str_value'] += ('<br>Price: ' + data['price'].apply(currency_str)
                + ' / Amt.: ' + data['amount'].apply(decimal_str))

        fig.add_trace(go.Scatter(x=data['date'], y=data['red_fill'], fill='tozeroy',
            mode='none', fillcolor='rgba(255,0,0,0.7)', hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=data['date'], y=data['net_investment'], mode='none',
            customdata=data['str_net_investment'],
            hovertemplate=f'Net investment: %{{customdata}}<extra></extra>'))
        data['f_profit'] = data['profit'].apply(currency_str) + ' / '
        data['f_relative_profit'] = data['relative_profit'].apply(percentage_str)
        data['profit_string'] = data['f_profit'] + data['f_relative_profit']
        
        fig.add_trace(go.Scatter(x=data['date'], y=data['value_and_return'], fill='tozeroy',
            mode='none', fillcolor='rgba(0,255,0,0.7)', customdata=data['profit_string'],
            hovertemplate=f'Net profit: %{{customdata}}<extra></extra>'))

        blue_fill_mode = 'tozeroy'
        if max(data['return_received']) > 0:
            fig.add_trace(go.Scatter(x=data['date'], y=data['return_received'],
                fill='tozeroy', mode='none', fillcolor='rgba(0,0,0,0)',
                customdata=data['str_return_received'],
                hovertemplate=f'Return received: %{{customdata}}<extra></extra>'))
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
            fig.add_trace(go.Scatter(x=data['date'], y=data['value'],
                mode='lines', line=dict(color='yellow'), customdata=data['str_value'],
                hovertemplate=f'Value: %{{customdata}}<extra></extra>'))

    fig.update_layout(hovermode='x', showlegend=False)
    fig.update_layout(hoverlabel=dict(bgcolor=theme_colors['tab_background_color'],
        font=dict(color=theme_colors['text_color'])))
    configure_historical_dataview(fig, latest_date - input.loc[input.index[0],'date'])
    one_year = [latest_date - datetime.timedelta(days=365), latest_date]
    fig.update_xaxes(range=one_year, rangeslider=dict(visible=True))
    fig.update_yaxes(ticksuffix=currency_tick_suffix(), tickprefix=currency_tick_prefix())
    return fig

def plot_yearly_asset_data(data: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    # take earliest value of each year, and append latest value
    yearly_data = data.groupby(data['date'].dt.year).head(1)
    yearly_data = yearly_data.append(data.iloc[-1])
    
    yearly_data['value_change'] = yearly_data['profit'] - yearly_data['return_received']
    yearly_data.loc[yearly_data.index[0], 'value_change'] = 0
    yearly_data['value_change'] = yearly_data['value_change'].diff()
    yearly_data.loc[yearly_data.index[0], 'total_return_received'] = 0
    yearly_data['total_return_received'] = yearly_data['total_return_received'].diff()

    # subtract one year from each date, since these lines are going to represent value change,
    # which occured during previous year
    yearly_data['date'] = yearly_data['date'].dt.year - 1
    if yearly_data.index[-1] != yearly_data.index[-2]:
        yearly_data.loc[yearly_data.index[-1], 'date'] += 1  # set back year of last row
    yearly_data.drop(yearly_data.head(1).index, inplace=True)  # remove first row
    yearly_data.drop_duplicates(subset=['date'], inplace=True)

    yearly_data['value_change_positive'] = np.where(yearly_data['value_change'] > 0,
        yearly_data['value_change'], 0)
    yearly_data['value_change_negative'] = np.where(yearly_data['value_change'] < 0,
        yearly_data['value_change'], 0)
    yearly_data['str_total_return_received'] = yearly_data['total_return_received'].apply(
        currency_str)
    yearly_data['str_value_change_positive'] = '+' + yearly_data['value_change_positive'].apply(
        currency_str)
    yearly_data['str_value_change_negative'] = yearly_data['value_change_negative'].apply(
        currency_str)

    bar_width = [0.5] if (len(yearly_data) == 1) else None # single column looks ugly otherwise

    fig.add_trace(go.Bar(x=yearly_data['date'],
        y=yearly_data['total_return_received'], marker=dict(color='rgb(73,200,22)'),
        width=bar_width, customdata=np.transpose(yearly_data['str_total_return_received']),
        hovertemplate=(
            f'<b>%{{x}}</b><br>'
            f'Return received: %{{customdata}}<extra></extra>')))

    hovertemplate = (
        f'<b>%{{x}}</b><br>'
        f'Value change: %{{customdata}}<extra></extra>')
    fig.add_trace(go.Bar(x=yearly_data['date'], y=yearly_data['value_change_positive'],
        marker=dict(color='rgba(0, 255, 0, 0.7)'), width=bar_width, hovertemplate=hovertemplate,
        customdata=np.transpose(yearly_data['str_value_change_positive'])))
    fig.add_trace(go.Bar(x=yearly_data['date'], y=yearly_data['value_change_negative'],
        marker=dict(color='rgba(255, 0, 0, 0.7)'), width=bar_width, hovertemplate=hovertemplate,
        customdata=np.transpose(yearly_data['str_value_change_negative'])))

    fig.update_layout(barmode='relative', showlegend=False)
    fig.update_xaxes(type='category', fixedrange=True)
    fig.update_yaxes(ticksuffix=currency_tick_suffix(), tickprefix=currency_tick_prefix(),
        fixedrange=True)
    fig.update_layout(margin=dict(l=10, r=10, t=35, b=10))

    return fig

def configure_historical_dataview(figure: go.Figure, timerange: datetime.timedelta):
    # buttons, that are always present
    buttons = [dict(count=6, label='6M', step='month', stepmode='backward'),
        dict(count=1, label='YTD', step='year', stepmode='todate'),
        dict(count=1, label='1Y', step='year', stepmode='backward')]

    # adding buttons depending on input time range
    years = [1, 2, 5, 10, 20, 50, 100]
    for i in range(1, len(years)):
        if timerange.days > years[i-1] * 365:
            buttons.append(dict(count=years[i], label=f'{years[i]}Y',
                step='year', stepmode='backward'))

    buttons.append(dict(label='ALL', step='all'))  # ALL is also always available

    figure.update_xaxes(type='date', rangeslider_visible=False,
        rangeselector=dict(x=0, buttons=buttons, font=dict(color=theme_colors['text_color']),
            bgcolor=theme_colors['background_color'],
            activecolor=theme_colors['hover_tab_indicator_color']))
    figure.update_layout(margin=dict(l=10, r=10, t=20, b=10))
    figure.update_layout(xaxis=dict(title=dict(text='')), yaxis=dict(title=dict(text='')))

def append_figures(statistic: str, label_text: str) -> html.Columns:
    sunburst = go.Figure(plot_sunburst(current_stats, statistic, label_text))
    sunburst = sunburst.update_traces(insidetextorientation='radial')
    sunburst.update_layout(margin=dict(l=10, r=10, t=10, b=10))

    if statistic == 'relative_profit':
        historical = plot_historical_relative_profit(monthly_data)
    elif statistic == 'return_received':
        historical = plot_historical_return(monthly_data, label_text)
    else:
        historical = plot_historical_data(monthly_data, statistic, label_text)

    return html.Columns([html.Column(width=30, content=sunburst.to_html(
            full_html=False, include_plotlyjs=True)),
        html.Column(content=historical.to_html(
            full_html=False, include_plotlyjs=True))])

def append_asset_data_view(data: pd.DataFrame):

    name = data.loc[data.index[-1], 'name']
    if 'symbol' in data.columns:
        if pd.notnull(data.loc[data.index[-1], 'symbol']):
            name += f" ({data.loc[data.index[-1], 'symbol']})"
    if 'account' in data.columns:
        name += f"<br>{data.loc[data.index[-1], 'account']}"
    output = f'<h2>{name}</h2>'

    statistics = [html.Label('Value', html.Value(currency_str(data.loc[data.index[-1], 'value'])))]
    statistics.append(html.Label('Funds invested', html.Value(currency_str(data.loc[data.index[-1], 'net_investment']))))
    if data.loc[data.index[-1], 'return_received'] != 0:
        statistics.append(html.Label('Return received', html.Value(currency_str(data.loc[data.index[-1], 'return_received']))))
    statistics.append(html.Label('Net profit', html.Value(currency_str(data.loc[data.index[-1], 'profit'])).color()))
    statistics.append(html.Label('Relative net profit', html.Value(percentage_str(data.loc[data.index[-1], 'relative_profit'])).color()))
    output += f'{html.Columns(statistics)}'
    if (len(data) > 1):
        output += f'{html.Columns([html.Column(width=30, content=plot_yearly_asset_data(data).to_html(full_html=False, include_plotlyjs=True)), html.Column(content=plot_historical_asset_data(data).to_html(full_html=False, include_plotlyjs=True))])}'
    return output

def append_overall_data_tabs(document: html.Document):
    tabs = [html.Tab(html.Label('Value', html.Value(currency_str(current_stats['value'].sum()))),
        append_figures('value', 'Value'), checked=True)]
    tabs.append(html.Tab(html.Label('Funds invested', html.Value(currency_str(current_stats['net_investment'].sum()))),
        append_figures('net_investment', 'Funds invested')))
    if current_stats['return_received'].max() > 0:
        tabs.append(html.Tab(html.Label('Return received', html.Value(currency_str(current_stats['return_received'].sum()), value_change=currency_str(-52))),
            append_figures('return_received', 'Return received')))
    tabs.append(html.Tab(html.Label('Net profit', html.Value(currency_str(current_stats['profit'].sum()), value_change=currency_str(3.50)).color()),
        append_figures('profit', 'Net profit')))
    tabs.append(html.Tab(html.Label('Relative net profit', html.Value(percentage_str(current_stats['profit'].sum()/current_stats['net_investment_max'].sum())).color()),
        append_figures('relative_profit', 'Relative net profit')))

    document.append(html.TabContainer(tabs))

def calculate_total_historical_data(input: pd.DataFrame, name: str) -> pd.DataFrame:
    data = input.copy()
    group_assets = data['id'].unique()
    all_dates = data.groupby('date').tail(1)['date']
    group_total = pd.DataFrame(columns=assets.columns).set_index(
        'date').reindex(all_dates).fillna(0)
    group_total[['id', 'name', 'symbol', 'group', 'account', 'color']] = ''
    for a in group_assets:
        asset_data = process_data(data[data['id'] == a].set_index(
            'date').reindex(all_dates).reset_index(), discard_zero_values=False)
        asset_data = asset_data.set_index('date').fillna(0)
        asset_data[['id', 'name', 'symbol', 'group', 'account', 'color']] = ''
        group_total = group_total.add(asset_data)
    group_total = process_data(group_total.reset_index())
    group_total['name'] = name
    group_total['symbol'] = np.NaN
    group_total[['price', 'amount']] = 0
    return group_total

def append_asset_data_tabs(document: html.Document):
    groups = sorted(assets['group'].unique())
    tabs = []
    for g in groups:
        content = ''

        group_data = assets[assets['group'] == g]
        group_accounts = sorted(group_data['account'].unique())
        
        # display group total if there is more than one account, or only "mixed" account
        if ((len(group_accounts) > 1) or
                ((len(group_accounts) == 1) and (group_accounts[0] == ' '))):
            group_total = calculate_total_historical_data(group_data, f'{g} Total')
            content += append_asset_data_view(group_total)
        
        for acc in group_accounts:
            account_data = group_data[group_data['account'] == acc]
            account_assets = sorted(account_data['name'].unique())

            if (len(account_assets) > 1) and (acc != ' '):
                account_total = calculate_total_historical_data(account_data, f'{acc} Total')
                content += append_asset_data_view(account_total)

            for a in account_assets:
                content += append_asset_data_view(account_data[account_data['name'] == a])

        tabs.append(html.Tab(html.Label(g), content))
    document.append(html.TabContainer(tabs))

if __name__ == '__main__':
    # making warnings not show source, since it's irrelevant in this case
    warnings.formatwarning = lambda msg, *args, **kwargs: f'{msg}\n'

    settings = Settings()

    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str, nargs='?',
        default=f'{os.path.dirname(os.path.realpath(__file__))}{os.path.sep}input_data')
    for s in settings:  # all settings are possible arguments
        parser.add_argument(f'--{s}', type=type(getattr(settings, s)))
    arguments = parser.parse_args()
    for s in settings:
        if (getattr(arguments, s) != None):
            setattr(settings, s, getattr(arguments, s))

    pd.set_option('display.max_rows', None)  # makes pandas print all dataframe rows

    settings_found = 'no'
    for entry in os.scandir(arguments.input_dir):
        if entry.is_file() and (entry.path.endswith('.yaml') or entry.path.endswith('.yml')):
            with open( entry, 'r' ) as read_file:
                input = yaml.load( read_file, Loader=yaml.BaseLoader )
            # check if there are general settings in file
            if any([s in input for s in settings]):
                for s in input:
                    setattr(settings, s, input[s])
                if settings_found == 'yes':
                    print_warning(f'Multiple settings files detected, expect trouble')
                    settings_found = 'warned'
                elif settings_found == 'no':
                    settings_found = 'yes'

    assets = pd.DataFrame()
    for entry in os.scandir(arguments.input_dir):
        if entry.is_file() and (entry.path.endswith('.yaml') or entry.path.endswith('.yml')):
            with open( entry, 'r' ) as read_file:
                input = yaml.load( read_file, Loader=yaml.BaseLoader )
            # check that there are no general settings in file
            if not any([s in input for s in settings]):
                ids = pd.DataFrame(input).drop(columns=['data'])
                data = pd.DataFrame(input['data']).join(ids)
                data = process_data(data)
                assets = assets.append(data)

    assets['account'] = assets['account'].fillna(' ')  # empty string doesn't work
    assets['id'] = assets['group'] + assets['account'] + assets['name']
    group_by_name = assets.groupby('id')[['id', 'name', 'group', 'account', 'value']].tail(1)
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
    asset_properties = asset_properties.sort_values(by='id')
    asset_properties = asset_properties.set_index('id')

    earliest_date = min(assets['date'])
    latest_date = max(assets['date'])

    current_stats = assets.groupby('id')[['id', 'name', 'symbol', 'group', 'account', 'net_investment', 'net_investment_max', 'return_received', 'value', 'profit', 'relative_profit', 'color', 'date']].tail(1)
    current_stats = current_stats[current_stats['date'] > latest_date - pd.DateOffset(months=6)]  # TODO: take months value from settings

    monthly_data = calculate_monthly_values(assets)

    if settings.theme == 'auto':
        settings.theme = 'light' if current_stats['profit'].sum() > 0 else 'dark'
    
    if settings.theme == 'light':
        plotly.io.templates.default = 'plotly_white'
        theme_colors = colors_light
    else:
        plotly.io.templates.default = 'plotly_dark'
        theme_colors =  colors_dark

    title = f'{settings.owner} investment portfolio'
    report = html.Document(title, css_variables=theme_colors)

    print('Generating report')
    report.append(f'<h1>{title}</h1>')
    report.append(f'<h3>Data from {date_str(earliest_date)} to {date_str(latest_date)}</h3>')
    append_overall_data_tabs(report)
    append_asset_data_tabs(report)

    report.append(f'<p class="footer">Report generated on {date_str(datetime.date.today())}, '
        f'using open source script: '
        f'<a href="https://github.com/zukaitis/investment-tracker/">Investment Tracker</a>'
        f'<br>All charts are displayed using '
        f'<a href="https://plotly.com/python/">Plotly</p>')

    print('Writing to file')
    with open('report.html', 'w') as f:
        print(report, file=f)

    print('Completed successfully')
