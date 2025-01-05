import streamlit as st
from joblib import load
import pandas as pd
import numpy as np


st.title('Crop Recommendation System')

with open('label_encoder.joblib','rb') as f:
    enc=load(f)

with open('model_rf.joblib','rb') as f:
    model=load(f)    

N=st.number_input("Ratio of Nitrogen content in soil")    
P=st.number_input("Ratio of Phosphorus content in soil")   
K=st.number_input("Ratio of Potassium content in soil") 
temp=st.number_input("Temperature(C)")
hum=st.number_input("Relative Humidity(%)") 
ph=st.number_input("Soil pH") 
rain=st.number_input("Rainfall(mm)")

if st.button('Recommend Crop'):
    query=np.array([N,P,K,temp,hum,ph,rain]).reshape(1,7)
    prediction=model.predict(query)
    st.title(enc.inverse_transform(prediction)[0])
    