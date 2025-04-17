import pickle
import streamlit as st
import xgboost as xgb
import numpy as np
import pandas as pd
import time
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Lobster&display=swap');
    
    .stTitle {
        font-family: 'Lobster', cursive;
        font-size: 40px;
        color: #FF6347;  /* Tomato color */
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True
)
# load the pre-trained model
model = pickle.load(open('xgb_model1.pkl', 'rb'))

# Title and description
st.title('Metal-doped MnO2 Specific Capacity Predictor')
st.write(f'An application for predicting the specific capacity of metal-doped manganese dioxide as cathode material for aqueous zinc-ion battery.')

# Layout improvements using columns
col1, col2 = st.columns(2)

# Load the atom property
atom_data = pd.read_csv('Atom.csv')
element_symbols = atom_data.iloc[:, 0].tolist()  

with col1:
    selected_element = st.selectbox('Select Element:', element_symbols)
    element_properties = atom_data[atom_data.iloc[:, 0] == selected_element].iloc[0, 1:]
    State, MW, EN, CR = element_properties

# Display selected element's properties
with col2:
    st.write(f'Selected Element: {selected_element}')
    st.write(f'Molecular weight: {MW}, Electronegativity: {EN}, Covalent Radius: {CR}, State: {State}')

# Experimental method selection
Experiment_method = pd.read_csv('methods.csv')
methods_symbols = Experiment_method.iloc[:, 0].tolist() 
selected_method = st.selectbox('Experimental Method:', methods_symbols)
EM = Experiment_method[Experiment_method.iloc[:, 0] == selected_method].iloc[0, 1]
# Other features with sliders
st.subheader('Input Parameters')
Ratio = st.slider('Ratio', min_value=0.01, max_value=1.0, value=0.01, step=0.001)
Zn2_plus = st.slider('Zn2+ Concentration (mol/L)', min_value=0.0, max_value=4.0, value=2.0, step=0.1)
Mn2_plus = st.slider('Mn2+ Concentration (mol/L)', min_value=0.0, max_value=1.0, value=0.2, step=0.05)
high_voltage = st.slider('High Voltage (V)', min_value=0.0, max_value=2.0, value=1.8, step=0.01)
low_voltage = st.slider('Low Voltage (V)', min_value=0.0, max_value=2.0, value=0.8, step=0.01)
Current_Density = st.slider('Current Density (A/g)', min_value=0.0, max_value=5.0, value=0.5, step=0.01)

# Prediction button with progress bar
if st.button('Predict'):
    placeholder = st.empty()
    placeholder.text('Calculating...')
    time.sleep(0.2) 
    # Prepare the input data for prediction
    feature_columns = ['MW', 'EN', 'CR', 'State', 'Ratio', 'EM', 'Zn2+', 'Mn2+', 'Low', 'High', 'CD']
    input_df = pd.DataFrame([[
        MW, EN, CR, State, Ratio, EM, Zn2_plus, Mn2_plus, low_voltage, high_voltage, Current_Density
    ]], columns=feature_columns)
    
    # Make prediction
    prediction = model.predict(input_df)
    
    st.write(f'Predicted Specific Capacity: {prediction[0]:.4f} mAh/g')
    placeholder.empty()
