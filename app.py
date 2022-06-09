import streamlit as st
import pandas as pd
import seaborn as sns
import joblib

import matplotlib.pyplot as plt
import numpy as np

# import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

st.write("""
# Machine Learning - Mobile Price Range
This app shows the results of some methods to classify the mobile price ranges based on multiple predictors.
""")
st.write("""
Get this dataset [here](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification?datasetId=11167&sortBy=voteCount)
""")

url = "https://raw.githubusercontent.com/nurulizzahzambri/finalassignmentmaybeidk/main/train.csv"
phone_data = pd.read_csv(url)

st.write("""
         ## These are the predictors for this dataset
         """)
st.write(phone_data.columns)

st.write("""
         ## The summary of only numeric X variables
         """)
         
st.write(phone_data[['battery_power','clock_speed','fc','int_memory','m_dep','mobile_wt','pc','px_height','px_width','ram','sc_h','sc_w','talk_time']].describe())


st.write("""
         ## Scatter plot of x vs y
         """)

#fig = plt.figure(figsize=(10, 4))
#sns.heatmap(phone_data.corr(), cmap = "PuOr", annot = True, vmin = -1, vmax = 1, center = 0)
#st.pyplot(fig)


col1, col2 = st.columns(2)

with col1:
    x = st.radio(
     'Select an X',
      ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
       'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
       'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',
       'touch_screen', 'wifi', 'price_range'])

with col2:
    y = st.radio(
     'Select a y',
      ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
       'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
       'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',
       'touch_screen', 'wifi', 'price_range'])


st.vega_lite_chart(phone_data, {
     'width': 800,
     'height': 800,
     'mark': {'type': 'circle', 'tooltip': True},
     'encoding': {
         'x': {'field': x, 'type': 'quantitative'},
         'y': {'field': y, 'type': 'quantitative'},
     },
 })

arr = phone_data['battery_power']
arry = phone_data['ram']
fig, ax = plt.subplots()
ax.scatter(arr,arry)
ax.set_xlabel("battery_power")
ax.set_ylabel("ram")
st.pyplot(fig)

arr = phone_data['battery_power']
fig, ax = plt.subplots()
ax.hist(arr)
ax.set_xlabel("battery_power")
ax.set_ylabel("count")
st.pyplot(fig)

arr = phone_data['battery_power']
arry = phone_data['price_range']
fig, ax = plt.subplots()
ax.bar(arr,arry,width=1,linewidth=0.7)
ax.set_xlabel("battery_power")
ax.set_ylabel("price_range")
st.pyplot(fig)

X = phone_data[['int_memory', 'px_height', 'px_width', 'battery_power', 'ram']]
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
  st.write('## K-NN method')
  st.write('### Classification Report')

  knn = KNeighborsClassifier()
  knn.fit(Xtrain, ytrain)
  ypred = knn.predict(Xtest)
  

elif option == 'SVM':
  st.write('## SVM method')
  st.write('### Classification Report')
         
  svc = SVC()
  svc.fit(Xtrain, ytrain)
  ypred = svc.predict(Xtest)
         
 # joblib.dump(svc, "svc.pkl")
  

elif option == 'Logistic Regression':
  st.write('## Logistic Regression method')
  st.write('### Classification Report')
         
  logreg = LogisticRegression()
  logreg.fit(Xtrain, ytrain)
  ypred = logreg.predict(Xtest)
  

elif option == 'Gaussian Naive Bayes':
  st.write('## Gaussian Naive Bayes method')
  st.write('### Classification Report')
         
  nb = GaussianNB()
  nb.fit(Xtrain, ytrain)
  ypred = nb.predict(Xtest)
  
  
elif option == 'Random Forest':
  st.write('## Random Forest method')
  st.write('### Classification Report')
  rf = RandomForestClassifier()
  rf.fit(Xtrain, ytrain)
  ypred = rf.predict(Xtest)
  
  
report = classification_report(ytest, ypred, output_dict=True)
cf = pd.DataFrame(report).transpose() 
st.write(cf)



# Header
st.write("## Price Range Predictor")
# X = 'int_memory','px_height','px_width','battery_power','ram'
# Input bar 1
int_memory = st.slider("Internal memory",min_value=2.0,max_value=64.0,value=32.0,step=0.1)

# Input bar 2:5
px_height = st.slider("Phone Height",min_value=20,max_value=1960,value=100,step=1)
px_width = st.slider("Phone Width",min_value=500,max_value=1998,value=600,step=1)
battery_power = st.slider("Battery Power",min_value=501,max_value=1998,value=600,step=1)
ram =  st.slider("Ram",min_value=256,max_value=3998,value=400,step=1)

         
# If button is pressed
if st.button("Confirm"):
  # Store inputs into dataframe
  X = pd.DataFrame([[int_memory,px_height,px_width,battery_power,ram]], 
                     columns = ["int_memory", "px_height", "px_width","battery_power","ram"])
         
  if option == 'K-NN':
    st.write('### predicted price_range')

    knn = KNeighborsClassifier()
    knn.fit(Xtrain, ytrain)
    ypred = knn.predict(Xtest)
  
    # Get prediction
    prediction = knn.predict(X)[0]

  elif option == 'SVM':
    st.write('### predicted price_range')

    svc = SVC()
    svc.fit(Xtrain, ytrain)
    ypred = svc.predict(Xtest)
       
    # Get prediction
    prediction = svc.predict(X)[0]
         

  elif option == 'Logistic Regression':
    st.write('### predicted price_range')

    logreg = LogisticRegression()
    logreg.fit(Xtrain, ytrain)
    ypred = logreg.predict(Xtest)
         
    # Get prediction
    prediction = logreg.predict(X)[0]
  

  elif option == 'Gaussian Naive Bayes':
    st.write('### predicted price_range')

    nb = GaussianNB()
    nb.fit(Xtrain, ytrain)
    ypred = nb.predict(Xtest)
         
    # Get prediction
    prediction = nb.predict(X)[0]
  
  
  elif option == 'Random Forest':
    st.write('### predicted price_range')

    rf = RandomForestClassifier()
    rf.fit(Xtrain, ytrain)
    ypred = rf.predict(Xtest)
 
    # Get prediction
    prediction = rf.predict(X)[0]
    

  # Output prediction
  st.write(f"This price range is {prediction}")
