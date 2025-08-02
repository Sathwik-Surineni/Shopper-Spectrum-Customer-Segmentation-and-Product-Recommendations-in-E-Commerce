# main.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Load Dataset
df = pd.read_csv("data.csv", encoding='ISO-8859-1')

# Clean Data
df.dropna(subset=['CustomerID'], inplace=True)
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])


# RFM Feature Engineering

snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'count',
    'TotalPrice': 'sum'
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# Normalize RFM
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Save models
joblib.dump(scaler, "scaler.pkl")
joblib.dump(kmeans, "kmeans.pkl")


# Product Recommendation

pivot = df.pivot_table(index='CustomerID', columns='StockCode', values='Quantity', fill_value=0)
similarity = cosine_similarity(pivot.T)
item_similarity_df = pd.DataFrame(similarity, index=pivot.columns, columns=pivot.columns)
item_similarity_df.to_pickle("product_similarity.pkl")

print(" Models trained and saved. You can now run the Streamlit app.")
