import streamlit as st
import time
import pickle

# Pickle files (model,and scaler)
with open("model.pkl","rb") as obj1:
   tree_model=pickle.load(obj1)
with open("scaler.pkl", "rb") as obj2:
   minmax=pickle.load(obj2)

# Function

def predict(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
  lst1=[Pregnancies, Glucose,	BloodPressure,SkinThickness,	Insulin, BMI,	DiabetesPedigreeFunction,	Age]
  var1=minmax.transform([[i for i in lst1]])
  pred=tree_model.predict(var1)
  if pred==1:
    return"Diabetes Positive."
  else:
    return "Diabetes Negative"

# Streamlit
# Title and description

st.title("🩺 Diabetes Prediction")
st.markdown("""
Welcome to the Diabetes Prediction App! This app predicts whether a patient has diabetes based on their medical details.
""")
   

# Sidebar for individual input
with st.sidebar:
    st.header("Individual Prediction")
    Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    Glucose = st.number_input("Glucose", min_value=0.0, max_value=200.0, value=100.0)
    BloodPressure = st.number_input("Blood Pressure", min_value=0.0, max_value=150.0, value=70.0)
    SkinThickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=20.0)
    Insulin = st.number_input("Insulin", min_value=0.0, max_value=900.0, value=80.0)
    BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    Age = st.number_input("Age", min_value=0, max_value=120, value=30)

    if st.button("Predict Individual"):
        with st.spinner("Predicting..."):
            time.sleep(3)  
            result = predict(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
            if result==0:
               st.snow()
            st.success(f"Prediction: **{result}**")




