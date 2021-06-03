#!/usr/bin/python3

import _html as html
from _settings import Settings
from _common import print_warning

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
import babel.numbers
import babel.dates
import warnings

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

def contains_non_zero_values(column: pd.Series) -> bool:
    if any((column.values != 0) & (pd.notna(column.values))):
        return True
    return False

def calculate_value_change(values: pd.Series,
        iscurrency: bool = False, ispercentage: bool = False) -> html.ValueChange:
    if len(values) < 2:  # can't have value changes with 1 or 0 values
        return html.ValueChange()

    daily_change = None
    monthly_change = None
    data = values.copy().sort_index()

    delta = pd.tseries.frequencies.to_offset(settings.value_change_span)
    day_ago_index = data.index.get_loc(
        data.index[-1] - datetime.timedelta(days=1), method='nearest')
    # check how recent the data is
    if (((data.index[-1] - data.index[day_ago_index]) <= delta) and
            ((latest_date - data.index[-1]) <= delta)):
        change = data.iloc[-1] - data.iloc[day_ago_index]
        if iscurrency:
            if abs(change) >= 0.01:
                daily_change = currency_str(round(change, 2))
        elif ispercentage:
            if abs(change) >= 0.0001:  # 0.01%
                daily_change = percentage_str(change)
        else:
            if change != 0:
                daily_change = decimal_str(change)

    data = data.to_frame()
    data['year'] = data.index.year
    data['month'] = data.index.month
    monthly = data.groupby(['year', 'month']).last().reset_index()
    monthly.columns = ['year', 'month', 'value']
    monthly['m'] = monthly['year'] * 12 + monthly['month']  # month number for easier operations
    latest_m = latest_date.year * 12 + latest_date.month

    if (((monthly.loc[monthly.index[-1], 'm'] - monthly.loc[monthly.index[-2], 'm']) == 1) and
            (latest_m - monthly.loc[monthly.index[-1], 'm'] <= 1)):
        change = monthly.loc[monthly.index[-1], 'value'] - monthly.loc[monthly.index[-2], 'value']
        if iscurrency:
            if abs(change) >= 0.01:
                monthly_change = currency_str(change)
        elif ispercentage:
            if abs(change) >= 0.0001:  # 0.01%
                monthly_change = percentage_str(change)
        else:
            if change != 0:
                monthly_change = decimal_str(change)

    return html.ValueChange(daily=daily_change, monthly=monthly_change)

def calculate_frequency(dates: pd.Series) -> str:
    frequency = None
    if len(dates) >= 4:  # calculate frequency from three dates before the last
        frequency = pd.infer_freq(dates.iloc[-4:-1])
    if (frequency == None) and (len(dates) >= 2):
        period = dates.iloc[-1] - dates.iloc[-2]
        if period < datetime.timedelta(hours=2):
            frequency = 'H'  # hourly
        elif period < datetime.timedelta(days=2):
            frequency = 'D'  # daily
    return frequency

def autofill(input_data: pd.DataFrame) -> pd.DataFrame:
    data = input_data.copy()

    symbol = input_data.loc[input_data.index[0], 'symbol']
    ticker = yf.Ticker(symbol)
    print(f'Fetching yfinance data for {symbol}')
    # download extra data, just to be sure, that the requred date will appear on yf dataframe
    start_date = input_data.loc[input_data.index[0], 'date'] - datetime.timedelta(days=7)
    fine = ticker.history(period='5d', interval='60m').astype(float)
    coarse_end_date = None
    if len(fine) > 0:
        fine.index = fine.index.tz_convert(settings.timezone).tz_localize(None)
        coarse_end_date = fine.index[0].date()
    coarse = ticker.history(start=start_date, end=coarse_end_date, interval='1d')
    yfdata = pd.concat([coarse, fine])
    if 'currency' in ticker.info:
        if ticker.info['currency'] != settings.currency:
            # convert currency, if it differs from the one selected in settings
            currency_ticker = yf.Ticker(f"{ticker.info['currency']}{settings.currency}=X")
            currency_rate = currency_ticker.history(start=start_date, interval='1d')
            yfdata[settings.autofill_price_mark] *= currency_rate[settings.autofill_price_mark]
            yfdata['Dividends'] *= currency_rate[settings.autofill_price_mark]
    else:
        print_warning('Ticker currency info is missing. '
            f'Assuming, that ticker currency matches input currency ({settings.currency})')

    yfdata = yfdata.reset_index().rename(columns={'index':'date', 'Date':'date',
        settings.autofill_price_mark:'price'})

    data = pd.merge(data, yfdata, on='date', how='outer').sort_values(by=['date'])

    if 'return_tax' not in data.columns:
        data['return_tax'] = 0.0
    data['return_tax'] = pd.to_numeric(data['return_tax']).interpolate(method='pad').fillna(0.0)

    for p in ['name', 'symbol', 'account', 'group', 'info']: # TODO: move this part out of the method
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
    data = input_data.copy() if (type(input_data) is pd.DataFrame) else pd.DataFrame(input_data)

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
            data['value'] = data['value'].interpolate(method='linear').fillna(0.0)
        else:
            data['value'] = 0.0

    if 'return' in data.columns:
        data['return'] = pd.to_numeric(data['return'])
        data['return'] = data['return'].fillna(0.0)
    else:
        data['return'] = 0.0

    nonzero_after_zero_mask = ((data['value'] != 0) & (data['value'].shift(1) == 0))
    zero_mask = (data['value'] == 0) & (data['investment'] == 0) & (data['return'] == 0)
    data['period'] = np.where(nonzero_after_zero_mask, 1, np.nan)
    data['period'] = data['period'].cumsum().interpolate(method='pad')
    data['period'] = data['period'].fillna(0) + 1
    if discard_zero_values:
        # drop zero values of last period
        last_period_mask = (data['period'] == max(data['period']))
        data.drop(data[zero_mask & last_period_mask].index, inplace=True)
    data.loc[zero_mask, 'period'] = 0  # period 0 marks price-only periods
    data['sold'] = ((data['value'] == 0) & (data['value'].shift(1) != 0))

    if (data['value'] == 0).all():
        # if time till next date is too long, add 'sold' mark for montly calculations
        data['relevance_period'] = data['date'] + pd.tseries.frequencies.to_offset(
            settings.relevance_period)
        data.loc[data['relevance_period'] < data['date'].shift(-1), 'sold'] = True
        data.drop(columns=['relevance_period'], inplace=True)
        # add sold mark to the last row, if the row is older than specified period
        if (data.loc[data.index[-1], 'date'] + pd.tseries.frequencies.to_offset(
            settings.relevance_period)) < datetime.date.today():
            data.loc[data.index[-1], 'sold'] = True

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
    daily = input[input['period'] != 0].copy()  # only take values, containing full data
    daily['year'] = daily['date'].dt.year
    daily['month'] = daily['date'].dt.month
    group_by_month = daily.groupby(['id', 'year', 'month'])

    monthly = group_by_month.agg({'date': 'last', 'value': 'last', 'net_investment': 'last',
        'net_investment_max': 'last', 'return': 'sum', 'profit': 'last',
        'relative_profit': 'last', 'period': 'last', 'sold': 'last'}).reset_index()
    
    monthly['date'] = monthly['date'].map(lambda x: x.replace(day=1, hour=0, minute=0, second=0))
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
    # reshape array into a desired form, and fill missing values by using previous values
    value_by_date = dataframe.pivot(index='date', columns='id', values=values)
    value_by_date.interpolate(method='pad', inplace=True)

    # create a period array in the same form as value_by_date array, and set period to 0,
    #   after asset is sold, then fill missing values in the same fashion as value_by_date array
    period = dataframe.pivot(index='date', columns='id', values='period')
    sold = dataframe.pivot(index='date', columns='id', values='sold').shift(1)
    for c in period.columns:
        period[c] = np.where((sold[c] == True) & np.isnan(period[c]), 0, period[c])
    period.interpolate(method='pad', inplace=True)

    # set value to 0, whereever period is 0
    value_by_date *= period.applymap(lambda p: 1 if p > 0 else 0)

    str_value_by_date = value_by_date.applymap(currency_str)
    value_by_date_sum = value_by_date.sum(axis=1, skipna=True)
    str_value_by_date_sum = value_by_date_sum.apply(currency_str)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=value_by_date_sum.index, y=value_by_date_sum.values,
        mode='lines+markers', name='Total', marker=dict(color=cyan),
        customdata=np.transpose(str_value_by_date_sum.values), hovertemplate=(
            f'%{{x|%B %Y}}<br>'
            f'<b>Total {label_text.lower()}:</b> %{{customdata}}<extra></extra>')))

    for a in value_by_date.columns:
        if contains_non_zero_values(value_by_date[a]):
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
        if contains_non_zero_values(value_by_date[a]):
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
    overall['values'] = overall['profit'] / overall['net_investment_max']
    overall['strings'] = overall['values'].apply(percentage_str)
    fig.add_trace(go.Scatter(x=overall.index, y=overall['values']*100, mode='lines+markers',
        name='Total', marker=dict(color=cyan), customdata=overall['strings'], hovertemplate=(
            f'%{{x|%B %Y}}<br>'
            f'<b>Total profit:</b> %{{customdata}}<extra></extra>')))

    for a in value_by_date.columns:
        if contains_non_zero_values(value_by_date[a]):
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

    data = input.copy()
    data['value_and_return'] = data['net_investment'] + data['profit']

    data['str_net_investment'] = data['net_investment'].apply(currency_str)
    data['str_return_received'] = data['return_received'].apply(currency_str)
    data['str_value'] = data['value'].apply(currency_str)
    if contains_non_zero_values(data['amount']):
        data['str_net_investment'] += ('<br><b>Amount:</b> ' + data['amount'].apply(decimal_str))

    data['f_profit'] = data['profit'].apply(currency_str) + ' / '
    data['f_relative_profit'] = data['relative_profit'].apply(percentage_str)
    data['profit_string'] = data['f_profit'] + data['f_relative_profit']

    for p in input['period'].dropna().unique():
        if p == 0:
            continue
        pdata = data[data['period'] == p].copy()

        fig.add_trace(go.Scatter(x=pdata['date'], y=pdata['net_investment'], mode='none',
            customdata=pdata['str_net_investment'], showlegend=False,
            hovertemplate=f'<b>Net investment:</b> %{{customdata}}<extra></extra>'))

        pdata['red_fill'] = np.where(pdata['net_investment'] > pdata['return_received'],
            pdata['net_investment'], pdata['return_received'])
        fig.add_trace(go.Scatter(x=pdata['date'], y=pdata['red_fill'], fill='tozeroy',
            mode='none', fillcolor='rgba(255,0,0,0.7)', hoverinfo='skip', showlegend=False))

        fig.add_trace(go.Scatter(x=pdata['date'], y=pdata['value_and_return'], fill='tozeroy',
            mode='none', fillcolor='rgba(0,255,0,0.7)', customdata=pdata['profit_string'],
            hovertemplate=f'<b>Net profit:</b> %{{customdata}}<extra></extra>', showlegend=False))

        blue_fill_mode = 'tozeroy'
        if max(pdata['return_received']) > 0:
            fig.add_trace(go.Scatter(x=pdata['date'], y=pdata['return_received'],
                fill='tozeroy', mode='none', fillcolor='rgba(0,0,0,0)',
                customdata=pdata['str_return_received'], showlegend=False,
                hovertemplate=f'<b>Return received:</b> %{{customdata}}<extra></extra>'))
            blue_fill_mode = 'tonexty'

        blue_fill = pdata[['date', 'red_fill', 'value_and_return']].copy()
        blue_fill.index *= 2
        blue_fill['y'] = np.where(blue_fill['red_fill'] < blue_fill['value_and_return'],
            blue_fill['red_fill'], pdata['value_and_return'])
        blue_fill['profitable'] = (blue_fill['y'] == blue_fill['red_fill'])
        mask = blue_fill.iloc[:-1]['profitable'] ^ blue_fill['profitable'].shift(-1)
        intermediate_values = blue_fill[mask].copy()
        intermediate_values.index += 1
        intermediate_values['y'] = np.nan
        blue_fill = blue_fill.append(intermediate_values).sort_index().reset_index(drop=True)
        blue_fill['slope'] = (abs(blue_fill['value_and_return'] - blue_fill['red_fill']) /
            (abs(blue_fill['value_and_return'] - blue_fill['red_fill'])
            + abs(blue_fill.shift(-1)['value_and_return'] - blue_fill.shift(-1)['red_fill'])))
        blue_fill['date'] = np.where(pd.isna(blue_fill['y']),
            ((blue_fill.shift(-1)['date'] - blue_fill['date']) * blue_fill['slope']
                + blue_fill['date']),
            blue_fill['date'])
        blue_fill['y'] = np.where(pd.isna(blue_fill['y']),
            ((blue_fill.shift(-1)['red_fill'] - blue_fill['red_fill']) * blue_fill['slope']
                + blue_fill['red_fill']),
            blue_fill['y'])
        fig.add_trace(go.Scatter(x=blue_fill['date'], y=blue_fill['y'], fill=blue_fill_mode,
            mode='none', fillcolor='rgba(0,0,255,0.5)', hoverinfo='skip', showlegend=False))

        if max(pdata['value']) > 0:
            fig.add_trace(go.Scatter(x=pdata['date'], y=pdata['value'],
                mode='lines', line=dict(color='yellow'), customdata=pdata['str_value'],
                hovertemplate=f'<b>Value:</b> %{{customdata}}<extra></extra>', showlegend=False))

    fig.update_layout(hovermode='x', showlegend=True, legend=dict(yanchor='bottom', y=1.02, 
        xanchor='right', x=1.04))
    fig.update_layout(hoverlabel=dict(bgcolor=theme_colors['tab_background_color'],
        font=dict(color=theme_colors['text_color'])))
    frequency = calculate_frequency(data['date'])
    earliest_entry_date = input.loc[input.index[0],'date']
    latest_entry_date = input.loc[input.index[-1],'date']
    configure_historical_dataview(fig, latest_date - earliest_entry_date, frequency)
    span = pd.DateOffset(years=1)
    if ((frequency in ['H', 'BH'])  # hourly data
            and (latest_date - latest_entry_date < datetime.timedelta(days=3))):
        span = pd.DateOffset(days=5)
    elif ((frequency in ['H', 'BH', 'B', 'C', 'D'])  # daily data
            and (latest_date - latest_entry_date < datetime.timedelta(days=14))):
        span = pd.DateOffset(months=1)
    range = [latest_date - span, latest_date]
    fig.update_xaxes(range=range, rangeslider=dict(visible=True))
    max_value = max(max(data['net_investment']), max(data['value_and_return']))
    fig.update_yaxes(ticksuffix=currency_tick_suffix(), tickprefix=currency_tick_prefix(),
        range=[0, max_value * 1.05])

    if contains_non_zero_values(data['price']):  # check if price data is present
        data['price_string'] = data['price'].apply(currency_str)
        fig.add_trace(go.Scatter(x=data['date'], y=data['price'],
            mode='lines', name='Price', marker=dict(color=cyan), yaxis='y2',
            customdata=data['price_string'], visible='legendonly',
            hovertemplate=f'<b>Price:</b> %{{customdata}}<extra></extra>'))
        margin = (max(data['price']) - min(data['price'])) * 0.05
        fig.update_layout(yaxis2=dict(title='', title_standoff=0,
            titlefont=dict(color='cyan'), tickfont=dict(color='cyan'), overlaying='y',
            ticksuffix=currency_tick_suffix(), tickprefix=currency_tick_prefix(), side='right',
            range=[min(data['price']) - margin, max(data['price']) + margin]))
        fig.update_layout(margin=dict(r=2))

    comments = data[pd.notnull(data['comment'])]
    
    fig.add_trace(go.Scatter(x=comments['date'],
        y=[max_value * 0.05]*len(comments),  # display comment mark at 5% of max y value
        mode='markers', marker=dict(line=dict(width=2, color='purple'), size=12,
        symbol='asterisk'), customdata=comments['comment'],
        hovertemplate=f'<b>*</b> %{{customdata}}<extra></extra>', showlegend=False))

    return fig

def plot_yearly_asset_data(data: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    # filter price-only data, where period is 0
    # take earliest value of each year, and append overall latest value
    yearly_data = data[data['period'] != 0].groupby(data['date'].dt.year).head(1)
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

    yearly_data['relative_value_change'] = abs(yearly_data['value_change'] /
        yearly_data['net_investment_max'])
    yearly_data['relative_return_received'] = (yearly_data['total_return_received'] /
        yearly_data['net_investment_max'])

    yearly_data['value_change_positive'] = np.where(yearly_data['value_change'] > 0,
        yearly_data['value_change'], 0)
    yearly_data['value_change_negative'] = np.where(yearly_data['value_change'] < 0,
        yearly_data['value_change'], 0)
    yearly_data['str_total_return_received'] = yearly_data['total_return_received'].apply(
        currency_str) + ' / ' + yearly_data['relative_return_received'].apply(percentage_str)
    yearly_data['str_value_change_positive'] = '+' + yearly_data['value_change_positive'].apply(
        currency_str) + ' / ' + yearly_data['relative_value_change'].apply(percentage_str)

    if contains_non_zero_values(data['value']):
        yearly_data['str_value_change_negative'] = yearly_data['value_change_negative'].apply(
            currency_str) + ' / ' + yearly_data['relative_value_change'].apply(percentage_str)
    else:
        yearly_data['str_value_change_negative'] = abs(
            yearly_data['value_change_negative']).apply(currency_str)

    bar_width = [0.5] if (len(yearly_data) == 1) else None # single column looks ugly otherwise

    fig.add_trace(go.Bar(x=yearly_data['date'],
        y=yearly_data['total_return_received'], marker=dict(color='rgb(73,200,22)'),
        width=bar_width, customdata=np.transpose(yearly_data['str_total_return_received']),
        hovertemplate=(
            f'<b>%{{x}}</b><br>'
            f'Return received:<br>%{{customdata}}<extra></extra>')))

    if contains_non_zero_values(data['value']):
        hovertemplate = (
            f'<b>%{{x}}</b><br>'
            f'Value change:<br>%{{customdata}}<extra></extra>')
    else:
        hovertemplate = (
            f'<b>%{{x}}</b><br>'
            f'Funds invested:<br>%{{customdata}}<extra></extra>')

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

def configure_historical_dataview(figure: go.Figure, timerange: datetime.timedelta,
        frequency: str = None):
    buttons = []

    if frequency in ['H', 'BH']:  # hourly data
        buttons += [dict(count=1, label='1D', step='day', stepmode='backward'),
            dict(count=5, label='5D', step='day', stepmode='backward')]

    if frequency in ['H', 'BH', 'B', 'C', 'D']:  # daily data
        buttons += [dict(count=1, label='1M', step='month', stepmode='backward'),
            dict(count=3, label='3M', step='month', stepmode='backward')]

    # buttons, that are always present
    buttons += [dict(count=6, label='6M', step='month', stepmode='backward'),
        dict(count=1, label='YTD', step='year', stepmode='todate'),
        dict(count=1, label='1Y', step='year', stepmode='backward')]

    # adding buttons depending on input time range
    years = [1, 2, 5]
    for i in range(1, len(years)):
        if timerange.days > years[i-1] * 365:
            buttons += [
                dict(count=years[i], label=f'{years[i]}Y', step='year', stepmode='backward')]

    buttons += [dict(label='ALL', step='all')]  # ALL is also always available

    figure.update_xaxes(type='date', rangeslider_visible=False,
        rangeselector=dict(x=0, buttons=buttons, font=dict(color=theme_colors['text_color']),
            bgcolor=theme_colors['background_color'],
            activecolor=theme_colors['hover_tab_indicator_color']))
    figure.update_layout(margin=dict(l=10, r=10, t=20, b=10))
    figure.update_layout(xaxis=dict(title=dict(text='')), yaxis=dict(title=dict(text='')))

def get_overall_figures(statistic: str, label_text: str) -> html.Columns:

    if statistic == 'relative_profit':
        historical = plot_historical_relative_profit(monthly_data)
    elif statistic == 'return_received':
        historical = plot_historical_return(monthly_data, label_text)
    else:
        historical = plot_historical_data(monthly_data, statistic, label_text)

    if all(current_stats[statistic] == 0):
        return html.Columns([html.Column(content=historical.to_html(
            full_html=False, include_plotlyjs=True))])

    sunburst = go.Figure(plot_sunburst(current_stats, statistic, label_text))
    sunburst.update_traces(insidetextorientation='radial')
    sunburst.update_layout(margin=dict(l=10, r=10, t=10, b=10))

    return html.Columns([html.Column(width=30, content=sunburst.to_html(
            full_html=False, include_plotlyjs=True)),
        html.Column(content=historical.to_html(
            full_html=False, include_plotlyjs=True))])

def create_asset_data_view(input: pd.DataFrame) -> str:
    data = input.copy()
    last_row = data.iloc[-1].to_dict()

    title = last_row['name']
    if ('symbol' in last_row) and (pd.notnull(last_row['symbol'])):
        title += f" ({last_row['symbol']})"
    if ('account' in last_row) and (pd.notnull(last_row['account'])):
        title += f"<br>{last_row['account']}"

    if ('info' not in last_row) or (type(last_row['info']) is not str):
        last_row['info'] = ''
    output = html.Columns([html.Column(width=50, content=html.Heading2(title)),
        html.Column(content=html.Paragraph(last_row['info']))])
    statistics = []

    if contains_non_zero_values(data['value']):
        data['value_change'] = data['profit'] - data['return_received']
        c = calculate_value_change(data.set_index('date')['value_change'], iscurrency=True)
        statistics.append(html.Label('Value',
            html.Value(currency_str(last_row['value']), valuechange=c)))

    # don't display Funds invested, if asset was sold
    if not ((contains_non_zero_values(data['value'])) and (last_row['value'] == 0)):
        c = calculate_value_change(data.set_index('date')['net_investment'], iscurrency=True)
        statistics.append(html.Label('Funds invested',
            html.Value(currency_str(last_row['net_investment']), valuechange=c)))

    last_row['return_received'] = round(last_row['return_received'], 2)
    if last_row['return_received'] != 0:
        c = calculate_value_change(data.set_index('date')['return_received'], iscurrency=True)
        statistics.append(html.Label('Return received', html.Value(
            currency_str(last_row['return_received']), valuechange=c)))

    c = calculate_value_change(data.set_index('date')['profit'], iscurrency=True)
    statistics.append(html.Label('Net profit',
        html.Value(currency_str(last_row['profit']), valuechange=c).color()))

    c = calculate_value_change(data.set_index('date')['relative_profit'], ispercentage=True)
    statistics.append(html.Label('Relative net profit',
        html.Value(percentage_str(last_row['relative_profit']), valuechange=c).color()))

    output += f'{html.Columns(statistics)}'

    if len(data) > 1:
        yearly_figure = plot_yearly_asset_data(data).to_html(
            full_html=False, include_plotlyjs=True)
        historical_figure = plot_historical_asset_data(data).to_html(
            full_html=False, include_plotlyjs=True)
        figures = html.Columns([html.Column(width=30, content=yearly_figure),
            html.Column(content=historical_figure)])
        output += f'{figures}'

    return output

def append_overall_data_tabs(document: html.Document):
    # calculate total using only currently active values
    total = calculate_total_historical_data(assets[assets['active']])
    last_row = total.iloc[-1].to_dict()

    tabs = []
    if assets['value'].any() > 0:
        total['value_change'] = total['profit'] - total['return_received']
        change = calculate_value_change(total.set_index('date')['value_change'], iscurrency=True)
        label = html.Label('Value', html.Value(currency_str(last_row['value']),
            valuechange=change))
        content = get_overall_figures('value', 'Value')
        tabs.append(html.Tab(label, content, checked=True))

    change = calculate_value_change(total.set_index('date')['net_investment'], iscurrency=True)
    label = html.Label('Funds invested', html.Value(currency_str(last_row['net_investment']),
        valuechange=change))
    content = get_overall_figures('net_investment', 'Funds invested')
    tabs.append(html.Tab(label, content))

    if assets['return_received'].any() > 0:
        change = calculate_value_change(
            total.set_index('date')['return_received'], iscurrency=True)
        value = None
        if last_row['return_received'] != 0:
            value = html.Value(currency_str(last_row['return_received']), valuechange=change)
        label = html.Label('Return received', value)
        content = get_overall_figures('return_received', 'Return received')
        tabs.append(html.Tab(label, content))

    change = calculate_value_change(total.set_index('date')['profit'], iscurrency=True)
    label = html.Label('Net profit', html.Value(currency_str(last_row['profit']),
        valuechange=change).color())
    content = get_overall_figures('profit', 'Net profit')
    tabs.append(html.Tab(label, content))

    change = calculate_value_change(total.set_index('date')['relative_profit'], ispercentage=True)
    label = html.Label('Relative net profit', html.Value(
        percentage_str(last_row['relative_profit']), valuechange=change).color())
    content = get_overall_figures('relative_profit', 'Relative net profit')
    tabs.append(html.Tab(label, content))

    document.append(html.TabContainer(tabs))

def calculate_total_historical_data(input: pd.DataFrame, name: str = 'Total') -> pd.DataFrame:
    data = input.copy()
    group_assets = data['id'].unique()
    all_dates = data.groupby('date').tail(1)['date']
    group_total = pd.DataFrame(columns=assets.columns).set_index(
        'date').reindex(all_dates).fillna(0)
    # TODO: fix this stuff in BS data handling, as it's growing out of control
    group_total[['id', 'name', 'symbol', 'group', 'account', 'color', 'comment', 'info']] = np.nan
    for a in group_assets:
        asset_data = process_data(data[data['id'] == a].set_index(
            'date').reindex(all_dates).reset_index(), discard_zero_values=False)
        asset_data = asset_data.set_index('date').fillna(0)
        asset_data[['id', 'name', 'symbol', 'group', 'account', 'color', 'comment', 'info']] = np.nan
        group_total = group_total.add(asset_data)
    group_total = process_data(group_total.reset_index())
    group_total['name'] = f'<i>{name}</i>'
    group_total[['price', 'amount']] = 0
    return group_total

def append_asset_data_tabs(document: html.Document):
    groups = assets.groupby('id').tail(1).groupby('group')[['group', 'value', 'active']]
    groups = groups.agg({'value': 'sum', 'active': 'any'}).reset_index()
    groups.sort_values(by=['value', 'active'], inplace=True, ascending=False)
    groups = groups['group'].unique()

    tabs = []
    for g in groups:
        content = ''

        group_data = assets[assets['group'] == g]
        group_assets = group_data.groupby('id').tail(1)
        group_accounts = group_assets.groupby('account')[['account', 'value', 'active']]
        group_accounts = group_accounts.agg({'value': 'sum', 'active': 'any'}).reset_index()
        group_accounts.sort_values(by=['value', 'active'], inplace=True, ascending=False)
        group_accounts = group_accounts['account'].unique()

        # display group total if there is more than one account, or only "mixed" account
        if ((len(group_accounts) > 1) or
                ((len(group_accounts) == 1) and
                (group_accounts[0] == ' ') and
                (group_data['name'].nunique() > 1))):
            group_total = calculate_total_historical_data(group_data, f'{g} Total')
            content += create_asset_data_view(group_total)
            content += html.Divider() + html.Divider()  # double divider after Total
        
        for acc in group_accounts:
            account_data = group_data[group_data['account'] == acc]
            account_assets = account_data.groupby('name').tail(1).copy()
            account_assets.sort_values(by=['value', 'active'], inplace=True, ascending=False)
            account_assets = account_assets['name'].unique()

            if (len(account_assets) > 1) and (acc != ' '):
                account_total = calculate_total_historical_data(account_data, f'{acc} Total')
                content += create_asset_data_view(account_total)
                content += html.Divider()

            for a in account_assets:
                content += create_asset_data_view(account_data[account_data['name'] == a])
                if a != account_assets[-1]:  # add a divider, unless it's the last asset
                    content += html.Divider()

            if acc != group_accounts[-1]:  # add a double divider, unless it's the last account
                content += html.Divider() + html.Divider()

        tabs.append(html.Tab(html.Label(g), content))

    if len(groups) > 1:
        total = calculate_total_historical_data(assets)
        content = create_asset_data_view(total)
        tabs.append(html.Tab(html.Label('<i>Total</i>'), content, checked=True))
    
    document.append(html.TabContainer(tabs))

if __name__ == '__main__':
    # making warnings not show source, since it's irrelevant in this case
    warnings.formatwarning = lambda msg, *args, **kwargs: f'{msg}\n'

    settings = Settings()
    default_input_dir = f'{os.path.dirname(os.path.realpath(__file__))}{os.path.sep}input_data'

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-i', type=str, default=default_input_dir,
        help='Directory, containing input files')
    for s in settings:  # all settings are possible arguments
        parser.add_argument(f'--{s}', type=type(getattr(settings, s)),
            help=settings.get_description(s), choices=settings.get_allowed(s))
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
                    print_warning('Multiple settings files detected, expect trouble')
                    settings_found = 'warned'  # three states, so warning would only pop once
                elif settings_found == 'no':
                    settings_found = 'yes'

    assets = pd.DataFrame()
    for entry in os.scandir(arguments.input_dir):
        if entry.is_file() and (entry.path.endswith('.yaml') or entry.path.endswith('.yml')):
            with open( entry, 'r' ) as read_file:
                input = yaml.load( read_file, Loader=yaml.BaseLoader )
            # check that there are no general settings in file
            if not any([s in input for s in settings]):
                try:
                    info = pd.DataFrame(input)
                except ValueError:
                    info = pd.DataFrame()  # create empty dataframe in case of an error in input

                if 'data' in info.columns:  # protection against files with no data
                    info.drop(columns=['data'], inplace=True)
                    data = pd.DataFrame(input['data']).join(info)
                    data = process_data(data)
                    assets = assets.append(data)

    assets['account'] = assets['account'].fillna(' ')  # empty string doesn't work
    assets['id'] = assets['group'] + assets['account'] + assets['name']  # + assets['symbol']
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

    current_stats = assets.groupby('id').tail(1)
    for i in current_stats['id']:
        if contains_non_zero_values(assets.loc[i == assets['id'], 'value']):
            # drop assets, which had non-zero value, but were sold
            # all() is needed, because it is assumed there might be more than one matching value
            if all(current_stats.loc[i == current_stats['id'], 'value'] == 0):
                current_stats = current_stats[i != current_stats['id']]
        else:
            # don't display taxes among current stats,
            #   if they weren't updated for a certain amount of time
            if current_stats.loc[i == current_stats['id'], 'date'].iloc[0] < (latest_date -
                    pd.tseries.frequencies.to_offset(settings.relevance_period)):
                current_stats = current_stats[i != current_stats['id']]

    assets['active'] = assets['id'].isin(current_stats['id'])
    if 'comment' not in assets:
        assets['comment'] = np.nan
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
    report.append(html.Button(image_initial='calendar_day.svg', 
        image_alternate='calendar_month.svg', identifier='value_change_button'))
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
