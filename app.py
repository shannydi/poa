import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Load the RandomForest model
model = joblib.load('random_forest_regressor.pkl')

# Load the data
data = pd.read_csv('algae.csv')

# Setting up page configurations
st.set_page_config(page_title="Algae Population Predictor", layout="wide")
st.markdown("""
    <style>
    .main {background-color: #e8f5e9;}
    </style>
    """, unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home Page", "Analysis Page", "Prediction Page"])

# Page 1: Home Page
if page == "Home Page":
    st.title("Understanding Algal Bloom")
    st.markdown("""
    Algal blooms can significantly affect aquatic ecosystems, leading to issues like hypoxia or dead zones. They are influenced by various factors including light, temperature, and nutrient availability.
    """)
    image = Image.open("algae_image.jpg")
    st.image(image, caption="Algal Bloom", use_column_width=True)

# Page 2: Analysis Page
elif page == "Analysis Page":
    st.title("Variable Analysis")
    var1 = st.selectbox("Select the first variable", data.columns)
    var2 = st.selectbox("Select the second variable", data.columns, index=1 if len(data.columns) > 1 else 0)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"Histogram of {var1}")
        fig1, ax1 = plt.subplots()
        sns.histplot(data[var1], kde=True, color='green', ax=ax1)
        st.pyplot(fig1)
    
    with col2:
        st.markdown(f"Histogram of {var2}")
        fig2, ax2 = plt.subplots()
        sns.histplot(data[var2], kde=True, color='green', ax=ax2)
        st.pyplot(fig2)

# Page 3: Prediction Page
elif page == "Prediction Page":
    st.title("Predict Algae Population")
    input_data = {feature: st.number_input(f"Enter {feature}:", format="%.2f") for feature in data.columns}
    if st.button("Predict"):
        features = pd.DataFrame([input_data])
        prediction = model.predict(features)[0]
        st.success(f"The predicted algae population is {prediction:.2f}")