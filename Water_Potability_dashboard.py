import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

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
    st.image('OIG (1).jpeg')
    st.title('Anggota Kelompok')
    st.write('1. Akhmad Nizar Zakaria\n2. Attar Syifa Kamal\n3. Muh Fijar Sukma Kartika\n4. Muh Zuman Ananta')
    st.caption('Pengpol Kelompok 8 2023')

icon("🤖")
"""
# Pattern Recognition Final Project
## Overview
In this final project, we aim to build a machine-learning model to predict the potability of water based on various water quality parameters. The dataset used for this project includes information on water characteristics such as pH, hardness, solids, chloramines, sulfate, conductivity, organic carbon, trihalomethanes, turbidity, and the target variable, potability.
"""

st.subheader('Water Potability Dataset and Visualization')
st.write(df_water_potability)

correlation_matrix = df_water_potability.corr()

tabs1, tabs2= st.tabs(["Data Visualization", "Prediction Result"])

with tabs1:

    st.subheader('Water Potability Distribution Based on Potability Category')

    colors = ['crimson', 'deeppink']

    fig, ax = plt.subplots()
    sns.countplot(x='Potability', data=df_water_potability, palette=colors, ax=ax)
    ax.set_xticklabels(['Not Potable', 'Potable'])
    ax.set_title('Potability Count')
    ax.set_xlabel('Potability')
    ax.set_ylabel('Count')

    # Display the plot in Streamlit
    st.pyplot(fig)

    columns_to_visualize = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']

    st.subheader('Normal Distribution Visualization')

    feature = st.selectbox('Select Feature', columns_to_visualize)
    data = df_water_potability[feature]   
    
    fig, ax = plt.subplots()

    ax.hist(data, bins=25, density=True, alpha=0.6)  
    
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, data.mean(), data.std())
    ax.plot(x, p, 'k', linewidth=2)
    
    ax.set_title(feature) 
    ax.set_xlabel(feature)
    ax.set_ylabel("Probability Density")
    
    st.pyplot(fig)

    st.subheader('Heatmap Visualization')
    
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='rocket', ax=ax)
    st.pyplot(fig)

with tabs2:
    st.header('User Input Features')
    
    col1, col2 = st.columns(2)

    features_1 = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate'] 
    features_2 = ['Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
    
    new_data = {}
    with col1:
        for feature in features_1:
            new_data[feature] = st.number_input(feature, min_value=None, max_value=None, value=None)
            
    with col2:
        for feature in features_2:
            new_data[feature] = st.number_input(feature, min_value=None, max_value=None, value=None)
            
    new_data = pd.DataFrame([new_data])
    
    # Normalize and predict
    new_data_scaled = scaler.transform(new_data)  
    prediction = model.predict(new_data_scaled)
    
    # Display prediction
    probability = prediction[0]
    st.write(f"Prediction: {'Potable' if probability>=0.5 else 'Not Potable'}")

    
    new_data = {}

st.caption('Pengpol Kelompok 8 2023')
