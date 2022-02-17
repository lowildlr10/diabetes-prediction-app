import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import seaborn as sns
import time


@st.cache(suppress_st_warning=True)
def load_csv_data(file_dir, head=0, tail=0):
    df = pd.read_csv(file_dir)
    latest_iteration = st.empty()
    bar = st.progress(0)
    for i in range(100):
        latest_iteration.text(f'Loading CSV Data... {i+1}%')
        bar.progress(i+1)
        time.sleep(0.01)
    bar.empty()
    latest_iteration.text('')
    if head > 0 and tail == 0:
        return df.head(head)
    elif head == 0 and tail > 0:
        return df.head(tail)
    return df


def load_chart(data, kind):
    if kind == 'line':
        st.write("Line Chart")
        st.line_chart(data)
    elif kind == 'area':
        st.write("Area Chart")
        st.area_chart(df[columns])
    elif kind == 'bar':
        st.write("Bar Chart")
        st.bar_chart(df[columns])
    else:
        st.write("Line Chart")
        st.line_chart(data)
        

        
#st.set_page_config(layout="wide")
st.title("Diabetes Predictor App")
st.write("From the diabetes data, we built a machine learning model for diabetes predictions.")


# Initialize CSV data
filename = "diabetes_classification.csv"
df = load_csv_data(filename, head=20)

# Initialize columns and target
columns = ['Glucose', 'BMI', 'Age', 'BloodPressure']
target = 'Outcome'

# Loading the model
filename = 'finalized_model_diabetes.sav'
loaded_model = joblib.load(filename)


# Sidebar
st.sidebar.title("Diabetes Predictor App Parameters")

# Dataframe visibility
st.sidebar.subheader("Data Frame Visibility")
option_sidebar = st.sidebar.checkbox("Hide")
if not option_sidebar:
    st.caption(f"Data Frame: '{filename}'")
    st.write(df)
    st.write("\n\n")
    
st.sidebar.subheader("Tweak to change predictions")

# Glucose
glucose = st.sidebar.slider("Glucose", 0, 200, 70)

# BMI
bmi = st.sidebar.slider("BMI", 0.0, 100.9, 50.0)

# Age
age = st.sidebar.slider("Age", 0, 150, 15)

# Blood Pressure
blood_pressure = st.sidebar.slider("Blood Pressure", 0, 300, 100)

# [Glucose, BMI, Age, BloodPressure]
prediction = round(loaded_model.predict([[glucose, bmi, age, blood_pressure]])[0])

if prediction == 0:
    risk_status = "No"
else:
    risk_status = "Yes"
    
st.sidebar.subheader("Predictions")
st.sidebar.write(f"Risk to Diabetes?: {risk_status}")

    
# Line chart
load_chart(df[columns], "line")

# Area chart
load_chart(df[columns], "area")

# Bar chart
load_chart(df[columns], "bar")

# Main Page
st.subheader("Predictions")

st.write(f"Risk to Diabetes?: {risk_status}")

# Load data
data = pd.read_csv("diabetes_classification.csv")

if st.checkbox("Show Graphs"):
    sns.pairplot(data[['Glucose', 'BMI', 'Age', 'BloodPressure']], height=8, kind='reg', diag_kind='kde')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
