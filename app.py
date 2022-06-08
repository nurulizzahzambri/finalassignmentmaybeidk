import streamlit as st
import pandas as pd
import seaborn as sns

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
  st.write('## Classification Report for K-NN method')

  knn = KNeighborsClassifier()
  knn.fit(Xtrain, ytrain)
  ypred = knn.predict(Xtest)
  

elif option == 'SVM':
  st.write('## Classification Report for SVM method')

  svc = SVC()
  svc.fit(Xtrain, ytrain)
  ypred = svc.predict(Xtest)
  

elif option == 'Logistic Regression':
  st.write('## Classification Report for Logistic Regression method')

  logreg = LogisticRegression()
  logreg.fit(Xtrain, ytrain)
  ypred = logreg.predict(Xtest)
  

elif option == 'Gaussian Naive Bayes':
  st.write('## Classification Report for Gaussian Naive Bayes method')

  nb = GaussianNB()
  nb.fit(Xtrain, ytrain)
  ypred = nb.predict(Xtest)
  
  
elif option == 'Random Forest':
  st.write('## Classification Report for Random Forest method')

  rf = RandomForestClassifier()
  rf.fit(Xtrain, ytrain)
  ypred = rf.predict(Xtest)
  
  
report = classification_report(ytest, ypred, output_dict=True)
cf = pd.DataFrame(report).transpose() 
st.write(cf)


st.write("""
         ## Scatter plot of x vs y
         """)

#fig = plt.figure(figsize=(10, 4))
#sns.heatmap(phone_data.corr(), cmap = "PuOr", annot = True, vmin = -1, vmax = 1, center = 0)
#st.pyplot(fig)

x = st.selectbox(
     'Select an X',
      ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
       'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
       'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',
       'touch_screen', 'wifi', 'price_range'])

y = st.selectbox(
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
