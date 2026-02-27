from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def cluster_traders(df):

    trader_features = (
        df.groupby('Account')
        .agg({
            'Closed PnL': 'mean',
            'Leverage': 'mean',
            'Size USD': 'mean'
        })
        .reset_index()
    )

    scaler = StandardScaler()
    scaled = scaler.fit_transform(
        trader_features[['Closed PnL', 'Leverage', 'Size USD']]
    )

    kmeans = KMeans(n_clusters=3, random_state=42)
    trader_features['Cluster'] = kmeans.fit_predict(scaled)

    return trader_features