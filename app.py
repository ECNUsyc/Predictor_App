# app.py
import pickle
import streamlit as st
import xgboost as xgb
import numpy as np
import pandas as pd

# load Pre-Trained XGBoost Model
model = pickle.load(open('xgb_model.pkl', 'rb'))

st.title('Metal Doping MnO2 Specific Capacity Predictor')
# 读取元素性质数据
atom_data = pd.read_csv('Atom.csv')
# 用户选择元素
element_symbols = atom_data.iloc[:, 0].tolist()  # 获取元素符号列表
selected_element = st.selectbox('Select Element:', element_symbols)
element_properties = atom_data[atom_data.iloc[:, 0] == selected_element].iloc[0, 1:4]
MW, EN, RC = element_properties

# 其他输入特征
Ratio = st.number_input('Ratio', value=0.01, format="%.3f", step=0.001)  # 比例
Zn2_plus = st.number_input('Zn2+', value=2.0,  step=0.5)  # Zn2+浓度
st.write('Unit: mol/L')  # Zn2+的单位
Mn2_plus = st.number_input('Mn2+', value=0.2, step=0.05)  # Mn2+浓度
st.write('Unit: mol/L')  # Mn2+的单位
Potential_Windows = st.number_input('Potential Windows', value=0.8,  step=0.05)  # 电压窗口
st.write('Unit: V')  # 电压窗口的单位
Current_Density = st.number_input('Current Density', value=0.5, format="%.2f", step=0.05)  # 电流密度
st.write('Unit: A/g')  # 电流密度的单位

# 显示选定元素的属性
st.write(f'Selected Element: {selected_element}')
st.write(f'Molecular Weight: {MW}, Electronegativity: {EN}, Covalent Radius: {RC}')

# 当用户点击预测按钮时进行模型预测
if st.button('Predict'):
    input_features = [[MW, EN, RC, Ratio, Zn2_plus, Mn2_plus, Potential_Windows, Current_Density]]  # 替换为实际特征名
    prediction = model.predict(input_features)
    st.write(f'Predicted Specific Capacity: {prediction[0]:.4f} mAh/g')
