import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
import plotly.express as px

# Load Dataset
@st.cache
def load_data():
    data = pd.read_csv("covid_19_indonesia_time_series_all.csv")
    data["Case Fatality Rate"] = data["Case Fatality Rate"].str.replace('%', '').astype(float) / 100
    return data

# Supervised Learning: Linear Regression
def train_regression_model(data):
    X = data[["Total Deaths", "Total Recovered", "Population Density", "Case Fatality Rate"]]
    y = data["Total Cases"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, mse, r2

# Unsupervised Learning: KMeans Clustering
def apply_clustering(data, n_clusters=5):
    features = data[["Total Cases", "Total Deaths", "Total Recovered", "Population Density"]]
    normalized_data = (features - features.mean()) / features.std()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data["Cluster"] = kmeans.fit_predict(normalized_data)
    return data, kmeans

# Dashboard
def main():
    st.title("COVID-19 Analysis Dashboard")
    st.sidebar.title("Options")

    # Load the fixed dataset directly
    data = load_data()
    st.dataframe(data.head())

    # Supervised Learning Section
    st.header("Supervised Learning: Predict Total Cases")
    model, mse, r2 = train_regression_model(data)
    st.write(f"Mean Squared Error: {mse}")
    st.write(f"R² Score: {r2}")

    # Unsupervised Learning Section
    st.header("Unsupervised Learning: Cluster Analysis")
    n_clusters = st.slider("Number of Clusters", 2, 10, 5)
    clustered_data, kmeans = apply_clustering(data, n_clusters)
    st.dataframe(clustered_data[["Province", "Cluster"]])

    # Visualization
    st.header("Interactive Visualizations")
    map_fig = px.scatter_mapbox(
        clustered_data,
        lat="Latitude",
        lon="Longitude",
        color="Cluster",
        hover_name="Province",
        title="Clustering Results on Map",
        mapbox_style="carto-positron",
        zoom=4
    )
    st.plotly_chart(map_fig)

    line_fig = px.line(
        data, 
        x="Date", 
        y="New Cases", 
        color="Province", 
        title="Daily New Cases Trend"
    )
    st.plotly_chart(line_fig)

if __name__ == "__main__":
    main()
