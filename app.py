import streamlit as st 
import pandas as pd
import numpy as np
import os
import pickle

# Set up page configuration
st.set_page_config(page_title="Smart Crop Recommender", page_icon="🌿", layout='centered', initial_sidebar_state="collapsed")

# Function to load the saved machine learning model
def load_prediction_model(filepath):
    with open(filepath, 'rb') as file:
        model = pickle.load(file)
    return model

def main():
    # Title and introductory section
    st.markdown("""
    <div>
    <h1 style="color:MEDIUMSEAGREEN;text-align:left;"> Smart Crop Recommendation System 🌱 </h1>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 2])
    
    # Information section
    with col1: 
        with st.expander(" ℹ️ Information", expanded=True):
            st.write("""
            Crop recommendation is a critical component of precision agriculture. It involves using a variety of factors to determine the most suitable crops for a particular site. By utilizing site-specific data, precision agriculture can enhance crop selection decisions. However, it is essential that these recommendations are accurate and precise to avoid potential material and financial losses.
            """)

        st.write("""
        ## How It Works ❓ 
        Simply input all the required parameters, and the machine learning model will predict the best crops to grow on your farm based on the data provided.
        """)

    # Input section
    with col2:
        st.subheader(" Find the Most Suitable Crop for Your Farm 👨‍🌾")
        N = st.number_input("Nitrogen (N)", min_value=1, max_value=10000, step=1)
        P = st.number_input("Phosphorus (P)", min_value=1, max_value=10000, step=1)
        K = st.number_input("Potassium (K)", min_value=1, max_value=10000, step=1)
        temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=100000.0, step=0.1)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100000.0, step=0.1)
        ph = st.number_input("Soil pH", min_value=0.0, max_value=100000.0, step=0.1)
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=100000.0, step=0.1)

        # Prepare the input data for prediction
        feature_list = [N, P, K, temperature, humidity, ph, rainfall]
        input_features = np.array(feature_list).reshape(1, -1)
        
        # Prediction button
        if st.button('Predict'):
            model = load_prediction_model('model.pkl')
            predicted_crop = model.predict(input_features)
            col1.write('''
		    ## Results 🔍 
		    ''')
            col1.success(f"{predicted_crop.item().title()} is recommended by the A.I. based on your input data.")

    # Hide Streamlit's default menu
    hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    </style>
    """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

if __name__ == '__main__':
	main()
