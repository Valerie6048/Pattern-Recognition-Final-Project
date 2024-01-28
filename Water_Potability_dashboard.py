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
    st.image('potonihcuy.jpg')
    st.title('Biodata')
    """
    Name: Akhmad Nizar Zakaria
    Github: [Valerie6048](https://github.com/Valerie6048)
    LinkedIn: [Akhmad Nizar Zakaria](https://www.linkedin.com/in/akhmad-nizar-zakaria-8a692b229/)

    """
    st.caption('@Valerie6048')

icon("🤖")
"""
# Pattern Recognition Final Project
## Overview
In this final project, I aim to build a machine-learning model to predict the potability of water based on various water quality parameters. The dataset used for this project includes information on water characteristics such as pH, hardness, solids, chloramines, sulfate, conductivity, organic carbon, trihalomethanes, turbidity, and the target variable, potability.
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
    
    new_data = {}
    
    for feature_name in X.columns:
        user_input = st.number_input(label=feature_name, 
                                    min_value=None, 
                                    max_value=None,  
                                    value=X[feature_name].mean())
        
        new_data[feature_name] = user_input
    
    new_data = pd.DataFrame([new_data]) 
    
    # Normalize 
    new_data_scaled = scaler.transform(new_data)  
    
    # Predict  
    prediction = model.predict(new_data_scaled)
    probability = model.predict_proba(new_data_scaled)[:, 1]
    
    # Display
    st.subheader('Prediction Result')
    result_text = f"The water is {'potable' if prediction[0] == 1 else 'not potable'} with a probability of {probability[0]:.2%}." 
    st.write(result_text)

st.caption('@Valerie6048')
