import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

st.write("""
# Simple Mobile Price Range Machine Learning App
This app predicts the **Mobile Price Range**!
""")

url = "https://raw.githubusercontent.com/nurulizzahzambri/finalassignmentmaybeidk/main/test.csv"
phone_data = pd.read_csv(url)

st.write(phone_data.head(5))

option = st.sidebar.selectbox(
     'Select a Classification method',
      ['K-NN','SVM','Logistic Regression','Gaussian Naive Bayes','Random Forest'])

if option == 'K-NN':
  st.write('This is the K-NN method')
    

elif option == 'SVM':
  st.write('This is the SVM method')
  

elif option == 'Logistic Regression':
  st.write('This is the Logistic Regression method')
    

elif option == 'Gaussian Naive Bayes':
  st.write('This is the Gaussian Naive Bayes method')
  
  
elif option == 'Random Forest':
  st.write('This is the Random Forest method')
