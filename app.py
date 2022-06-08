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
