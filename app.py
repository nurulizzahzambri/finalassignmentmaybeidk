import streamlit as st
import pandas as pd

# import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

st.write("""
# Simple Mobile Price Range Machine Learning App
This app predicts the **Mobile Price Range** using multiple methods.
""")

url = "https://raw.githubusercontent.com/nurulizzahzambri/finalassignmentmaybeidk/main/train.csv"
phone_data = pd.read_csv(url)

st.write("""
         ## The summary of numeric X variables
         """)
         
st.write(phone_data[['battery_power','clock_speed','fc','int_memory','m_dep','mobile_wt','pc','px_height','px_width','ram','sc_h','sc_w','talk_time']].describe())

#optionx = st.sidebar.selectbox(
 #    'Select an X',
  #    ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
   #    'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
    #   'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',
     #  'touch_screen', 'wifi', 'price_range'])

#optiony = st.sidebar.selectbox(
 #    'Select a y',
  #    ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
   #    'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
    #   'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',
     #  'touch_screen', 'wifi', 'price_range'])

#phone_data.plot.scatter(x = 'optionx', y = 'optiony')

st.write("""
         ## Filtered DataFrame
         """)

# filters

n_core = list(phone_data['n_cores'].drop_duplicates())
int_memories = list(phone_data['int_memory'].drop_duplicates())

ncore_choice = st.sidebar.multiselect(
    'Choose number of cores:', n_core, default=n_core)

intmem_choice = st.sidebar.slider(
    'Choose internal memory capacity:', min_value=2, max_value=64, step=1, value=16)

df = phone_data[phone_data['n_cores'].isin(ncore_choice)]
df = phone_data[phone_data['int_memory'] < intmem_choice]

st.write(df)

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

