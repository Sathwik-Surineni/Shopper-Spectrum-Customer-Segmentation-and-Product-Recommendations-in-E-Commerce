# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models
scaler = joblib.load("scaler.pkl")
kmeans = joblib.load("kmeans.pkl")
item_similarity_df = pd.read_pickle("product_similarity.pkl")

# Recommendation Function
def recommend(product_id, n=5):
    try:
        recs = item_similarity_df[product_id].sort_values(ascending=False)[1:n+1]
        return recs
    except KeyError:
        return None

# UI
st.title("🛒 Shopper Spectrum Web App")

tab1, tab2 = st.tabs(["📦 Product Recommendation", "👤 Customer Segmentation"])

with tab1:
    st.subheader("Find Similar Products")
    product_id = st.text_input("Enter Product ID (example: 84029E)")
    if st.button("Get Recommendations"):
        result = recommend(product_id)
        if result is not None:
            st.write("Top 5 Similar Products:")
            st.dataframe(result)
        else:
            st.error("Product ID not found.")

with tab2:
    st.subheader("Predict Customer Segment")
    recency = st.number_input("Recency (days)", min_value=0)
    frequency = st.number_input("Frequency (transactions)", min_value=0)
    monetary = st.number_input("Monetary Value", min_value=0)

    if st.button("Predict Segment"):
        input_data = np.array([[recency, frequency, monetary]])
        input_scaled = scaler.transform(input_data)
        label = kmeans.predict(input_scaled)[0]
        segments = {0: "High-Value", 1: "Regular", 2: "Occasional", 3: "At-Risk"}
        st.success(f"Predicted Segment: {segments.get(label, 'Unknown')}")
