# app.py
import pickle
import streamlit as st
import xgboost as xgb
import numpy as np
import pandas as pd

# 加载训练好的 XGBoost 模型
model = pickle.load(open('xgb_model.pkl', 'rb'))

st.title('Metal-doped MnO2 Specific Capacity Predictor')
# 读取元素性质数据
atom_data = pd.read_csv('Atom.csv')
# 用户选择元素
element_symbols = atom_data.iloc[:, 0].tolist()  # 获取元素符号列表
selected_element = st.selectbox('Select Element:', element_symbols)
element_properties = atom_data[atom_data.iloc[:, 0] == selected_element].iloc[0, 1:]
CG, MW, EN, CR = element_properties

# 定义CG值与金属种类的映射关系
metal_types = {
    1: "Alkali Metal",
    2: "Transition Metal",
    3: "Rare Earth Metal",
    4: "Post-Transition Metal"
}

# 获取金属种类名称
metal_type = metal_types.get(CG, "Unknown")

# 其他输入特征
Ratio = st.number_input('Ratio', value=0.01, format="%.3f", step=0.001)  # 比例
Zn2_plus = st.number_input('Zn2+ concentration (Unit: mol/L)', value=2.0,  step=0.5)  # Zn2+浓度
Mn2_plus = st.number_input('Mn2+ concentration (Unit: mol/L)', value=0.2, step=0.05)  # Mn2+浓度
high_voltage = st.number_input('High Voltage (Unit: V)', value=1.8,  step=0.05)  # 电化学测试中的电压较高值
low_voltage = st.number_input('Low Voltage (Unit: V)', value=0.8,  step=0.05)  # 电化学测试中的电压较低值
Current_Density = st.number_input('Current Density (Unit: A/g)', value=0.5, format="%.2f", step=0.05)  # 电流密度

# 显示选定元素的属性
st.write(f'Selected Element: {selected_element}')
st.write(f'Category: {metal_type}, Molecular weight: {MW}, Electronegativity: {EN}, Covalent Radius: {CR}')

# 当用户点击预测按钮时进行模型预测
if st.button('Predict'):

    # 转换为DataFrame并指定列名（需与训练时完全一致）
    feature_columns = [
        'MW', 'EN', 'CR','CG',
        'Ratio', 'Zn2+', 'Mn2+',
        'Low','High',  'CD'
    ]
    
    input_df = pd.DataFrame([[
        MW, EN, CR, CG,
        Ratio,
        Zn2_plus,
        Mn2_plus,
        low_voltage,
        high_voltage,
        Current_Density
    ]], columns=feature_columns)
    
    prediction = model.predict(input_df)
    st.write(f'Predicted Specific Capacity: {prediction[0]:.4f} mAh/g')

