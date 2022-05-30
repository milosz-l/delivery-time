import streamlit as st
import time
import numpy as np
import datetime

def getPrediction():
     print("get prediction")


status_text = st.sidebar.empty()

d_start = st.date_input("Date of purchase")

t_start = st.time_input('Time of purchase')

model = st.selectbox(
     'Choose model',
     ('Basic model (linear regression)', 'Advanced model (random forest)'))

submit = st.button("Get prediction", on_click=getPrediction)

time = 0

prediction = st.write("Your predicted time of delivery is " + str(t_start) + ", the delivery will take " + str(time))

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Re-run")