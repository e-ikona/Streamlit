import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
from sklearn import preprocessing


def naivbay(df, targeting, uji = 0.5):
    file = df.copy()
    target = file[targeting]
    file.drop(columns=[targeting], inplace=True)

    for column in file.columns:
        if file[column].dtype == type(object) or len(set(file[column])) <= 10:
            le = preprocessing.LabelEncoder()
            file[column] = le.fit_transform(file[column])

    feature_columns = file.columns.tolist()

    X = file[feature_columns]

    X = X.values

    y = target.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=uji, random_state=42)

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    akurasi = accuracy_score(y_test, y_pred)
    st.subheader(f"Akurasi\t:{int(akurasi*100)}%")

