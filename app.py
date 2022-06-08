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
## This app predicts the **Mobile Price Range** using multiple methods.
""")

url = "https://raw.githubusercontent.com/nurulizzahzambri/finalassignmentmaybeidk/main/train.csv"
phone_data = pd.read_csv(url)

st.write(phone_data.head(5))

st.write("""
         ## The summary of numeric X variables
         """)
         
st.write(phone_data[['battery_power','clock_speed','fc','int_memory','m_dep','mobile_wt','pc','px_height','px_width','ram','sc_h','sc_w','talk_time']].describe())

X = phone_data.drop(['price_range'], axis = 1)
y = phone_data['price_range']

# split data
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, 
                                                test_size = 0.2, 
                                                random_state = 999)


option = st.sidebar.selectbox(
     'Select a Classification method',
      ['K-NN','SVM','Logistic Regression','Gaussian Naive Bayes','Random Forest'])

if option == 'K-NN':
  st.write('This is the K-NN method')

  knn = KNeighborsClassifier()
  knn.fit(Xtrain, ytrain)
  ypred = knn.predict(Xtest)
  

elif option == 'SVM':
  st.write('This is the SVM method')

  svc = SVC()
  svc.fit(Xtrain, ytrain)
  ypred = svc.predict(Xtest)
  

elif option == 'Logistic Regression':
  st.write('This is the Logistic Regression method')

  logreg = LogisticRegression()
  logreg.fit(Xtrain, ytrain)
  ypred = logreg.predict(Xtest)
  

elif option == 'Gaussian Naive Bayes':
  st.write('This is the Gaussian Naive Bayes method')

  nb = GaussianNB()
  nb.fit(Xtrain, ytrain)
  ypred = nb.predict(Xtest)
  
  
elif option == 'Random Forest':
  st.write('This is the Random Forest method')

  rf = RandomForestClassifier()
  rf.fit(Xtrain, ytrain)
  ypred = rf.predict(Xtest)
  
  
report = classification_report(ytest, ypred, output_dict=True)
cf = pd.DataFrame(report).transpose() 
st.write(cf)

