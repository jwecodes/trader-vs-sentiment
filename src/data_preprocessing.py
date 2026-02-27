import pandas as pd

def load_and_merge():

    df1 = pd.read_csv("data/historical_data.csv")
    df2 = pd.read_csv("data/fear_greed_index.csv")

    # Fix timestamps
    df1['Timestamp IST'] = pd.to_datetime(df1['Timestamp IST'], dayfirst=True)
    df1['date'] = df1['Timestamp IST'].dt.normalize()

    df2['timestamp'] = pd.to_datetime(df2['timestamp'], unit='s')
    df2['date'] = df2['timestamp'].dt.normalize()

    merged_df = pd.merge(df1, df2, on='date', how='left')

    # Create leverage
    merged_df['Leverage'] = (
        merged_df['Size USD'] /
        merged_df['Start Position'].replace(0, 1)
    )

    return merged_df