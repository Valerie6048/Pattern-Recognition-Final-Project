import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.express as px

# Suppress future warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

df_water_potability = pd.read_csv('water_potability_cleaned.csv')

df_water_potability = df_water_potability.drop(df_water_potability.columns[0], axis=1)

model = RandomForestClassifier()
X = df_water_potability.drop('Potability', axis=1)
y = df_water_potability['Potability']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.08, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )
    
# Allow user input for new data
new_data = {}  # Add input fields for new data

model.fit(X_train_scaled, y_train)

original_feature_ranges = {feature_name: (X[feature_name].min(), X[feature_name].max()) for feature_name in X.columns}

with st.sidebar:
    st.image('OIG.jpeg')
    st.title('Anggota Kelompok')
    st.write('1. Akhmad Nizar Zakaria\n2. Attar Syifa\n3. Muh Fijar Sukma Kartika\n4. Muh Zuman Ananta')
    st.caption('Pengpol Kelompok 8 2023')

icon("🤖")
"""
# Pattern Recognition Final Project
## Overview
In this final project, we aim to build a machine learning model to predict the potability of water based on various water quality parameters. The dataset used for this project includes information on water characteristics such as pH, hardness, solids, chloramines, sulfate, conductivity, organic carbon, trihalomethanes, turbidity, and the target variable, potability.
"""

st.subheader('Water Potability Dataset and Visualization')
st.write(df_water_potability)

correlation_matrix = df_water_potability.corr()

tabs1, tabs2= st.tabs(["Data Visualization", "Prediction Result"])

with tabs1:

    st.title('Water Potability Distribution Based on Potability Category')

    colors = ['skyblue', 'salmon']

    fig, ax = plt.subplots()
    sns.countplot(x='Potability', data=df_water_potability, palette=colors, ax=ax)
    ax.set_title('Potability Count')
    ax.set_xlabel('Potability')
    ax.set_ylabel('Count')

    # Display the plot in Streamlit
    st.pyplot(fig)

    st.title('Heatmap Visualization')
    
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='rocket', ax=ax)
    st.pyplot(fig)

with tabs2:
    st.header('User Input Features')

    for feature_name in X.columns:
        original_min, original_max = original_feature_ranges[feature_name]
        user_input = st.slider(
            f'Input {feature_name}', 
            float(original_min), 
            float(original_max), 
            float(X[feature_name].mean())
        )
        # Map the user input back to the original range
        normalized_input = (user_input - original_min) / (original_max - original_min)
        new_data[feature_name] = normalized_input
    
    new_data = pd.DataFrame([new_data])

    # Normalize the new data using the same scaler
    new_data_scaled = scaler.transform(new_data)

    # Predict the potability for new data
    prediction = model.predict(new_data_scaled)
    probability = model.predict_proba(new_data_scaled)[:, 1]

    # Display the prediction result
    st.subheader('Prediction Result')
    result_text = f"The water is {'potable' if prediction[0] == 1 else 'not potable'} with a probability of {probability[0]:.2%}."
    st.write(result_text)

st.caption('Pengpol Kelompok 8 2023')
