import pandas as pd
import numpy as np

def add_basic_features(df):
    df['Leverage'] = (
        df['Size USD'] /
        df['Start Position'].replace(0, np.nan)
    )

    df['is_win'] = (df['Closed PnL'] > 0).astype(int)

    return df

def create_daily_trader_features(df):
    daily_df = (
        df.groupby(['Account', 'date'])
        .agg({
            'Closed PnL': 'sum',
            'Size USD': 'mean',
            'Leverage': 'mean',
            'is_win': 'mean',
            'Trade ID': 'count',
            'classification': 'first'
        })
        .reset_index()
    )

    daily_df.rename(columns={
        'Trade ID': 'trade_count',
        'is_win': 'win_rate'
    }, inplace=True)

    return daily_df

def create_pnl_bucket(df):
    def bucket(x):
        if x > 0:
            return "Profit"
        elif x < 0:
            return "Loss"
        else:
            return "Neutral"

    df['pnl_bucket'] = df['Closed PnL'].apply(bucket)

    return df

def create_next_day_target(df):
    df = df.sort_values(['Account', 'date'])

    df['next_day_bucket'] = (
        df.groupby('Account')['pnl_bucket'].shift(-1)
    )

    df = df.dropna(subset=['next_day_bucket'])

    return df