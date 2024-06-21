import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def k_median_clustering(df, k, fitur1_index, fitur2_index):

    file = df.copy()

    for column in file.columns:
        if file[column].dtype == type(object) or len(set(file[column])) <= 10:
            le = preprocessing.LabelEncoder()
            file[column] = le.fit_transform(file[column])

    feature_columns = file.columns.tolist()

    X = file[feature_columns]
    X = X.values

    np.random.seed(42)
    initial_indices = np.random.choice(range(len(X)), size=k, replace=False)
    centroids = X[initial_indices]

    for _ in range(100):
        distances = cdist(X, centroids, metric='cityblock')
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([np.median(X[labels == i], axis=0) for i in range(k)])
        
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids

    kerekatan = silhouette_score(X, labels)
    st.subheader(f"Kerekatan\t: {int(kerekatan*100)}%")

    X_fitur = X
    y_label = labels

    X_train, X_test, y_train, y_test = train_test_split(X_fitur, y_label, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    akurasi = accuracy_score(y_test, y_pred)
    st.subheader(f"Akurasi\t:{int(akurasi*100)}%")
    st.write(file)
    sns.set_style("darkgrid")

    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(
        X[:, fitur1_index], X[:, fitur2_index], 
        c=labels, s=50, cmap='viridis', edgecolors='black', linewidth=1, alpha=0.75
    )

    ax.scatter(
        centroids[:, fitur1_index], centroids[:, fitur2_index], 
        c='red', s=200, alpha=0.75, edgecolors='black'
    )

    colorbar = plt.colorbar(scatter, ax=ax)
    colorbar.set_label('Cluster Label', color='white')
    colorbar.ax.yaxis.set_tick_params(color='white')

    colorbar.ax.yaxis.set_tick_params(labelcolor='white')

    ax.set_xlabel(feature_columns[fitur1_index], fontsize=12, color='white')
    ax.set_ylabel(feature_columns[fitur2_index], fontsize=12, color='white')
    ax.set_title(f'K-Median Clustering (n_clusters={k})', fontsize=14, color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    fig.patch.set_facecolor((0, 0, 0, 0.0))  
    ax.set_facecolor((0.2, 0.4, 0.6, 0.2)) 

    st.pyplot(fig)