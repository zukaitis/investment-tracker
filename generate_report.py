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
import yfinance as yf
from dataclasses import dataclass
import dataclasses
import warnings
import babel.numbers

cyan = px.colors.qualitative.Plotly[5]

@dataclass
class Settings:
    owner: str = 'Your'
    currency: str = 'EUR'
    locale: str = 'en_US_POSIX'
    autofill_interval: str = '1d'
    autofill_price_mark: str = 'Open'

def currency_str(value: float) -> str:
    if np.isnan(value) or np.isinf(value):
        return '?'
    precision = 2 
    if abs(value) < 0.0500:
        precision = 4
    elif abs(value) < 0.50:
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

def decimal_str(value: float) -> str:
    if np.isnan(value) or np.isinf(value):
        return '?'
    return babel.numbers.format_decimal(value, locale=settings.locale)

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

    ticker = yf.Ticker(input_data.loc[input_data.index[0], 'symbol'])
    # download extra data, just to be sure, that the requred date will appear on yf dataframe
    start_date = input_data.loc[input_data.index[0], 'date'] - datetime.timedelta(days=7)  
    yfdata = ticker.history(start=start_date, interval=settings.autofill_interval)
    if ticker.info['currency'] != settings.currency:
        # convert currency, if it differs from the one selected in settings
        currency_ticker = yf.Ticker(f"{settings.currency}{ticker.info['currency']}=X")
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
    data['investment'] = pd.to_numeric(data['investment']).fillna(0.0)
    data['value'] = data['amount'] * data['price'].interpolate(method='pad')
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
        warnings.warn(
            f'WARNING: There are duplicate dates in "{dataset_name}" dataset:\n{duplicates}')
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
    group_by_month = daily.groupby(['name', 'year', 'month'])

    monthly = group_by_month.agg({'date': 'last', 'value': 'last', 'net_investment': 'last',
        'net_investment_max': 'last', 'return': 'sum', 'profit':'last',
        'relative_profit': 'last'}).reset_index()
    
    monthly['date'] = monthly['date'].map(lambda x: x.replace(day=1))
    monthly.drop(columns=['year', 'month'], inplace=True)

    return monthly

def plot_sunburst(input: pd.DataFrame, values: str, label_text: str) -> go.Sunburst:
    dataframe = input.copy()
    dataframe = dataframe.loc[dataframe[values] != 0]  # filter 0 values
    dataframe['rootnode'] = ''
    dataframe['profitability'] = np.where(dataframe[values] > 0, 'Profitable', 'Unprofitable')

    dataframe['account'] = np.where(dataframe['account'] == ' ',
        dataframe['group'], dataframe['account'])

    # Manipulating names in order to separate positive and negative values
    dataframe['account'] = np.where(dataframe[values] < 0,
        dataframe['account'] + ' ', dataframe['account'])
    dataframe['group'] = np.where(dataframe[values] < 0,
        dataframe['group'] + ' ', dataframe['group'])

    dataframe['group_sum'] = 0
    for g in dataframe['group'].unique():
        dataframe.loc[dataframe['group'] == g, 'group_sum'] = \
            dataframe.loc[dataframe['group'] == g, values].sum()
    dataframe['profitability_sum'] = 0
    for g in dataframe['profitability'].unique():
        dataframe.loc[dataframe['profitability'] == g, 'profitability_sum'] = \
            dataframe.loc[dataframe['profitability'] == g, values].sum()

    data = pd.DataFrame(columns=['label', 'parent', 'value', 'profitability', 'color'])

    if dataframe[values].min() < 0:
        path = ['rootnode', 'profitability', 'group', 'account', 'name']
    else:
        path = ['rootnode', 'group', 'account', 'name']

    for i in range(1, len(path)):
        d = dataframe.groupby(path[i]).first().reset_index()
        d[values] = dataframe.groupby(path[i]).sum().reset_index()[values]
        d['fraction_in_group'] = d[values] / d['group_sum']
        d['fraction_in_profitability'] = d[values] / d['profitability_sum']

        # creating displayed string from 5 parts, which are formated depending on the values
        d = d.assign(s1='', s2='', s3='', s4='')
        if d['fraction_in_profitability'].iloc[0] < 1:
            if d['group'].nunique() > 1:
                d['s1'] = '<br>' + d['fraction_in_profitability'].apply(percentage_str)
                if dataframe[values].min() < 0:
                    d['s2'] = ' of ' + d['profitability']
                    d['s3'] = '<br>'
                else:
                    d['s3'] = ' / '
            else:
                d['s3'] = np.where(d['fraction_in_group'] < 1, '<br>', '')
            d['s4'] = np.where(d['fraction_in_group'] < 1,
                d['fraction_in_group'].apply(percentage_str) + ' of ' + d['group'], '')
            d['s3'] = np.where(d['fraction_in_group'] < 1, d['s3'], '')
        d['display_string'] = d[values].apply(currency_str) + d['s1'] + d['s2'] + d['s3'] + d['s4']

        d[values] = abs(d[values])
        d = d[[path[i], path[i-1], values, 'display_string', 'color']]
        d.columns = ['label', 'parent', 'value', 'display_string', 'color']
        # remove rows, where label matches parent
        d = d.drop(d[d.label == d.parent].index)
        data = data.append(d)

    # Set colors of Profitable and Unprofitable labels if they exist
    data.loc[data['label'] == 'Profitable', 'color'] = 'green'
    data.loc[data['label'] == 'Unprofitable', 'color'] = 'red'

    return go.Sunburst(labels=data['label'], parents=data['parent'],
        values=data['value'], customdata=data['display_string'],
        marker=dict(colors=data['color']), branchvalues='total', hovertemplate=
            f'<b>%{{label}}</b><br>{label_text}: %{{customdata}}<extra></extra>')

def plot_relative_profit_sunburst(input: pd.DataFrame) -> go.Sunburst:
    dataframe = input.copy()
    dataframe['rootnode'] = ''

    dataframe = dataframe.loc[dataframe['relative_profit'] != 0]  # filter 0 values
    dataframe = dataframe.loc[dataframe['relative_profit'] != -100]  # filter -100 values (taxes)
    dataframe['sign'] = np.where(dataframe['relative_profit'] > 0, 'Profitable', 'Unprofitable')

    # Manipulating names in order to separate positive and negative values
    dataframe['account'] = np.where(dataframe['account'] == ' ', dataframe['group'], dataframe['account'])
    dataframe['account'] = np.where(dataframe['relative_profit'] < 0, dataframe['account'] + ' ', dataframe['account'])
    dataframe['group'] = np.where(dataframe['relative_profit'] < 0, dataframe['group'] + ' ', dataframe['group'])

    dataframe['relative_profit_sum'] = dataframe['relative_profit']

    data = dataframe[['name', 'account', 'relative_profit', 'relative_profit_sum', 'sign', 'color']].copy()
    data.columns = ['label', 'parent', 'relative_profit', 'relative_profit_sum', 'sign', 'color']
    data['relative_profit_sum'] = abs(data['relative_profit_sum'])
    data['str_relative_profit'] = data['relative_profit'].apply(percentage_str)

    if dataframe['relative_profit'].min() < 0:
        path = ['rootnode', 'sign', 'group', 'account', 'name']
    else:
        path = ['rootnode', 'group', 'account', 'name']

    for i in range(1, len(path) - 1):
        d = dataframe.groupby(path[i]).first().reset_index()
        d['relative_profit_sum'] = dataframe.groupby(path[i]).sum().reset_index()['relative_profit_sum']
        d['sign'] = np.where(d['relative_profit_sum'] > 0, 'Profitable', 'Unprofitable')
        d['relative_profit_sum'] = abs(d['relative_profit_sum'])
        d['relative_profit'] = (dataframe.groupby(path[i]).sum().reset_index()['profit'] / dataframe.groupby(path[i]).sum().reset_index()['net_investment'])
        d['str_relative_profit'] = d['relative_profit'].apply(percentage_str)
        d = d[[path[i], path[i-1], 'str_relative_profit', 'relative_profit_sum', 'sign', 'color']]
        d.columns = ['label', 'parent', 'str_relative_profit', 'relative_profit_sum', 'sign', 'color']
        d = d.drop(d[d.label == d.parent].index)  # remove rows, where label matches parent
        data = data.append(d)

    # Set colors for positive and negative sign labels if they exist
    data.loc[data['label'] == 'Profitable', 'color'] = 'green'
    data.loc[data['label'] == 'Unprofitable', 'color'] = 'red'

    return go.Sunburst(labels=data['label'], parents=data['parent'],
        values=data['relative_profit_sum'], customdata=data['str_relative_profit'],
        marker=dict(colors=data['color']), branchvalues='total', hovertemplate=(
            f'<b>%{{label}}</b><br>'
            f'Relative net profit: %{{customdata}}<extra></extra>'))

def plot_historical_data(dataframe: pd.DataFrame, values: str, label_text: str) -> go.Figure:
    value_by_date = dataframe.pivot(index='date', columns='name', values=values)
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

    for a in asset_properties.index:
        if any(value_by_date[a] != 0):
            # only plot traces with at least one non-zero value
            fig.add_trace(go.Bar(x=value_by_date.index, y=value_by_date[a], name=a,
                marker=dict(color=asset_properties.loc[a, 'color']),
                customdata=np.transpose(str_value_by_date[a]), hovertemplate=(
                    f'%{{x|%B %Y}}<br>'
                    f'<b>{a}</b><br>'
                    f'{label_text}: %{{customdata}}<extra></extra>')))

    fig.update_layout(barmode='relative')
    six_months = [value_by_date.index[-1] - datetime.timedelta(days=(365/2 - 15)), value_by_date.index[-1] + datetime.timedelta(days=15)]
    fig.update_xaxes(range=six_months)
    fig.update_yaxes(ticksuffix=currency_tick_suffix(), tickprefix=currency_tick_prefix())
    configure_historical_dataview(fig, latest_date - value_by_date.index[0])

    return fig

def plot_historical_return(dataframe: pd.DataFrame, label_text: str) -> go.Figure:
    value_by_date = dataframe.pivot(index='date', columns='name', values='return').fillna(0.0)
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

    for a in asset_properties.index:
        if any(value_by_date[a] != 0):
            # only plot traces with at least one non-zero value
            fig.add_trace(go.Bar(x=value_by_date.index, y=value_by_date[a], name=a,
                customdata=np.transpose([str_value_by_date[a], str_value_by_date_cumulative[a]]),
                marker=dict(color=asset_properties.loc[a, 'color']), hovertemplate=(
                    f'%{{x|%B %Y}}<br>' 
                    f'<b>{a}</b><br>'
                    f'{label_text}: %{{customdata[0]}}<br>'
                    f'Total return received: %{{customdata[1]}}<extra></extra>')))

    fig.update_yaxes(ticksuffix=currency_tick_suffix(), tickprefix=currency_tick_prefix())
    fig.update_layout(yaxis2=dict(title='',
        titlefont=dict(color='cyan'), tickfont=dict(color='cyan'),
        ticksuffix=currency_tick_suffix(), tickprefix=currency_tick_prefix(), side='right', overlaying='y',
        range=[0, max(value_by_date_sum.values) * 1.05], fixedrange=True),legend=dict(x=1.07))
    fig.update_layout(barmode='group')
    six_months = [value_by_date.index[-1] - datetime.timedelta(days=(365/2 - 15)), value_by_date.index[-1] + datetime.timedelta(days=15)]
    fig.update_xaxes(range=six_months)
    configure_historical_dataview(fig, latest_date - value_by_date.index[0])

    return fig

def plot_historical_relative_profit(dataframe: pd.DataFrame) -> go.Figure:
    value_by_date = dataframe.pivot(index='date', columns='name', values='relative_profit')
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

    for a in asset_properties.index:
        if a in value_by_date.columns:
            fig.add_trace(go.Bar(x=value_by_date.index, y=value_by_date[a]*100, name=a,
                marker=dict(color=asset_properties.loc[a,'color']),
                customdata=np.transpose(str_value_by_date[a]), hovertemplate=(
                    f'%{{x|%B %Y}}<br>'
                    f'<b>{a}</b><br>'
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
        if (data['price'] != 0).any():
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
    fig.update_layout(hoverlabel=dict(bgcolor='white'))
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
    yearly_data.drop(yearly_data.head(1).index, inplace=True)

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

def configure_historical_dataview(figure: go.Figure, time_range: datetime.timedelta):
    # buttons, that are always present
    buttons = [dict(count=6, label='6M', step='month', stepmode='backward'),
        dict(count=1, label='YTD', step='year', stepmode='todate'),
        dict(count=1, label='1Y', step='year', stepmode='backward')]

    # adding buttons depending on input time range
    years = [1, 2, 5, 10, 20, 50, 100]
    for i in range(1, len(years)):
        if time_range.days > years[i-1] * 365:
            buttons.append(dict(count=years[i], label=f'{years[i]}Y',
                step='year', stepmode='backward'))

    buttons.append(dict(label='ALL', step='all'))  # ALL is also always available

    figure.update_xaxes(type='date', rangeslider_visible=False,
        rangeselector=dict(x=0, buttons=buttons))
    figure.update_layout(margin=dict(l=10, r=10, t=20, b=10))
    figure.update_layout(xaxis=dict(title=dict(text='')), yaxis=dict(title=dict(text='')))

def append_figures(statistic: str, label_text: str) -> html.Columns:
    if statistic == 'relative_profit':
        sunburst = go.Figure(plot_relative_profit_sunburst(current_stats))
    else:
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
        tabs.append(html.Tab(html.Label('Return received', html.Value(currency_str(current_stats['return_received'].sum()))),
            append_figures('return_received', 'Return received')))
    tabs.append(html.Tab(html.Label('Net profit', html.Value(currency_str(current_stats['profit'].sum())).color()),
        append_figures('profit', 'Net profit')))
    tabs.append(html.Tab(html.Label('Relative net profit', html.Value(percentage_str(current_stats['profit'].sum()/current_stats['net_investment'].sum())).color()),
        append_figures('relative_profit', 'Relative net profit')))

    document.append(html.TabContainer(tabs))

def append_asset_data_tabs(document: html.Document):

    groups = assets['group'].unique()

    tabs = []
    for g in groups:
        group_assets = assets.loc[assets['group'] == g, 'name'].unique()
        content = ''
        if len(group_assets) > 1:
            all_dates = assets[assets['group'] == g].groupby('date').tail(1)['date']
            group_total = pd.DataFrame(columns=assets.columns).set_index('date').reindex(all_dates).fillna(0.0)
            group_total[['name', 'symbol', 'group', 'account', 'color']] = ''
            for a in group_assets:
                asset_data = process_data(assets[assets['name'] == a].set_index('date').reindex(all_dates).reset_index(), discard_zero_values=False)
                asset_data = asset_data.set_index('date').fillna(0.0)
                asset_data[['name', 'symbol', 'group', 'account', 'color']] = ''
                group_total = group_total.add(asset_data)
            group_total = process_data(group_total.reset_index())
            group_total['name'] = f'{g} Total'
            group_total['symbol'] = np.NaN
            group_total[['price', 'amount']] = 0

            content += append_asset_data_view(group_total)
        for a in group_assets:
            content += append_asset_data_view(assets[assets['name'] == a])
        tabs.append(html.Tab(html.Label(g), content))
    document.append(html.TabContainer(tabs))

if __name__ == '__main__':
    # making warnings not show source, since it's irrelevant in this case
    warnings.formatwarning = lambda msg, *args, **kwargs: f'{msg}\n'

    parser = argparse.ArgumentParser()
    parser.add_argument( 'input_dir', type = str, nargs = '?', default = None )
    arguments = parser.parse_args()

    input_directory = input_directory = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + 'input_data'
    if arguments.input_dir != None:
        input_directory = arguments.input_dir

    settings = Settings
    for entry in os.scandir(input_directory):
        if entry.is_file() and (entry.path.endswith('.yaml') or entry.path.endswith('.yml')):
            with open( entry, 'r' ) as read_file:
                input = yaml.load( read_file, Loader=yaml.BaseLoader )
            if any([f.name in input for f in dataclasses.fields(Settings)]):
                for s in input:
                    setattr(settings, s, input[s])
                if 'owner' in input:
                    settings.owner = f"{input['owner']}'s"

    assets = pd.DataFrame()
    for entry in os.scandir(input_directory):
        if entry.is_file() and (entry.path.endswith('.yaml') or entry.path.endswith('.yml')):
            with open( entry, 'r' ) as read_file:
                input = yaml.load( read_file, Loader=yaml.BaseLoader )
            if not any([f.name in input for f in dataclasses.fields(Settings)]):
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

    pd.set_option('display.max_rows', None)
    print(asset_properties)

    color_map = asset_properties.to_dict()['color']

    earliest_date = min(assets['date'])
    latest_date = max(assets['date'])

    current_stats = assets.groupby('name')[['name', 'symbol', 'group', 'account', 'net_investment', 'return_received', 'value', 'profit', 'relative_profit', 'color', 'date']].tail(1)
    current_stats = current_stats[current_stats['date'] > latest_date - pd.DateOffset(months=6)]  # TODO: take months value from settings

    monthly_data = calculate_monthly_values(assets)

    title = f'{settings.owner} investment portfolio'
    report = html.Document(title)

    report.append(f'<h1>{title}</h1>')
    report.append(f'<h3>Data from {earliest_date:%Y-%m-%d} to {latest_date:%Y-%m-%d}</h3>')
    append_overall_data_tabs(report)
    append_asset_data_tabs(report)

    report.append(f'<p class="link">Report generated on {datetime.date.today():%Y-%m-%d}, '
        f'using open source script: '
        f'<a href="https://github.com/zukaitis/investment-tracker/">Investment Tracker</a>'
        f'<br>All charts are displayed using '
        f'<a href="https://plotly.com/python/">Plotly</p>')

    with open('report.html', 'w') as f:
        print(report, file=f)
