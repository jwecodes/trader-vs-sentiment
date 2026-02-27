# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_preprocessing import load_and_merge
from src.feature_engineering import (
    add_basic_features,
    create_daily_trader_features,
    create_pnl_bucket,
    create_next_day_target
)
from src.clustering import cluster_traders
from src.modeling import train_and_save_model

st.set_page_config(page_title="Trader Sentiment Analysis", layout="wide")

st.title("📊 Trader Sentiment Analysis Dashboard")
st.markdown("Analyzing trader behavior under Fear & Greed sentiment")

@st.cache_data
def load_data():
    df = load_and_merge()
    df = add_basic_features(df)
    return df

df = load_data()

st.sidebar.header("Filters")

selected_sentiment = st.sidebar.multiselect(
    "Select Sentiment",
    options=df['classification'].dropna().unique(),
    default=df['classification'].dropna().unique()
)

filtered_df = df[df['classification'].isin(selected_sentiment)]

st.subheader("📈 PnL Distribution")

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots()
    filtered_df['Closed PnL'].hist(ax=ax1, bins=30)
    ax1.set_title("PnL Distribution")
    st.pyplot(fig1)

with col2:
    pnl_by_sentiment = (
        filtered_df.groupby('classification')['Closed PnL']
        .mean()
        .reset_index()
    )

    fig2, ax2 = plt.subplots()
    ax2.bar(pnl_by_sentiment['classification'],
            pnl_by_sentiment['Closed PnL'])
    ax2.set_title("Average PnL by Sentiment")
    st.pyplot(fig2)

st.subheader("📊 Trading Behavior")

col3, col4 = st.columns(2)

with col3:
    trade_freq = (
        filtered_df.groupby('classification')
        .size()
        .reset_index(name="trade_count")
    )

    fig3, ax3 = plt.subplots()
    ax3.bar(trade_freq['classification'],
            trade_freq['trade_count'])
    ax3.set_title("Trade Frequency by Sentiment")
    st.pyplot(fig3)

with col4:
    avg_leverage = (
        filtered_df.groupby('classification')['Leverage']
        .mean()
        .reset_index()
    )

    fig4, ax4 = plt.subplots()
    ax4.bar(avg_leverage['classification'],
            avg_leverage['Leverage'])
    ax4.set_title("Average Leverage by Sentiment")
    st.pyplot(fig4)

st.subheader("👥 Trader Archetypes (Clustering)")

clustered = cluster_traders(filtered_df)

st.write("Cluster Summary:")
st.dataframe(
    clustered.groupby('Cluster')
    .mean(numeric_only=True)
)

fig5, ax5 = plt.subplots()
sns.scatterplot(
    data=clustered,
    x='Leverage',
    y='Closed PnL',
    hue='Cluster',
    ax=ax5
)
ax5.set_title("Trader Clusters")
st.pyplot(fig5)

st.subheader("🤖 Next-Day Profitability Prediction")

daily_df = create_daily_trader_features(filtered_df)
daily_df = create_pnl_bucket(daily_df)
model_df = create_next_day_target(daily_df)

if not model_df.empty:

    model = train_and_save_model(model_df)

    st.success("Model trained successfully!")

    st.write("Feature Importance:")

    importances = pd.DataFrame({
        "Feature": ['Size USD', 'Leverage', 'trade_count'],
        "Importance": model.feature_importances_
    })

    st.dataframe(importances)

else:
    st.warning("Not enough data for model training.")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")