import streamlit as st
import pandas as pd
import seaborn as sns
import joblib

import matplotlib.pyplot as plt
import plotly.express as px


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

st.write("""
# Machine Learning - Mobile Price Range
This app lets user choose a few classification methods to predict mobile price ranges based on multiple predictors, as well as some other useful informations such as the scatter plot, histogram, and the classification report of those methods.
""")
st.write("""
Get this mobile phone price range dataset [here](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification?datasetId=11167&sortBy=voteCount)
""")

st.write("""
Created by Nurul Izzah ([Portfolio](https://sites.google.com/view/izzahwebportfolio),[Github](https://github.com/nurulizzahzambri)),
""")

st.write("""
with the assistance and guidance of [Dr. Yu Yong Poh](https://www.linkedin.com/in/yong-poh-yu/) and [Dr. Tan Yan Bin](https://www.linkedin.com/in/yyanbin-tan/)
""")

# read dataset from github repo
url = "https://raw.githubusercontent.com/nurulizzahzambri/finalassignmentmaybeidk/main/train.csv"
phone_data = pd.read_csv(url)

# list predictors
st.write("""
         ## Predictors for this dataset
         We will only use int_memory, px_height, px_width, battery_power, ram for the machine learning as these predictors have the highest correlations with the price_range.
         """)


col1, col2, col3 = st.columns(3)

with col1:
    st.empty()

with col2:
    st.write(pd.DataFrame(phone_data.columns, columns = ['Predictors']))

with col3:
    st.empty()

# show correlation heatmap
fig = px.imshow(phone_data.corr(),color_continuous_scale='RdBu',zmin=-1,zmax=1)
st.plotly_chart(fig)

# Summary of numeric X variables
st.write("""
         ## The summary of X variables
         This table summarizes only Xs with numeric entries.
         """)
         
st.write(phone_data[['battery_power','clock_speed','fc','int_memory','m_dep','mobile_wt','pc','px_height','px_width','ram','sc_h','sc_w','talk_time']].describe())

# y variable description
st.write("""
         ## The description of y variable
         This table describes the entries of y variable.
         """)
st.write(pd.DataFrame({'price_range': [0, 1, 2, 3], 'description': ['low cost', 'medium cost', 'high cost', 'very high cost']}))

# Scatter plot with user input x and y
st.write("""
         ## Scatter Plot of x vs y
         """)
st.warning('not all scatter plots are useful, pick x and y wisely')

col4, col5 = st.columns(2)

# x and y options
with col4:
    x = st.radio(
     'Select an X for the scatter plot and the histogram',
      ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
       'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
       'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',
       'touch_screen', 'wifi', 'price_range'])

with col5:
    y = st.radio(
     'Select a y for the scatter plot',
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

# Histogram of user input x
st.write("""
         ## Histogram of x(user selection)
         The x variable of this histogram can be changed in previous radio selector. 
         """)

arr = phone_data[x]
fig, ax = plt.subplots()
ax.hist(arr)
ax.set_xlabel(f"{x}")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# X and y for machine learning
X = phone_data[['int_memory', 'px_height', 'px_width', 'battery_power', 'ram']]
y = phone_data['price_range']

# split data into test and train set
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, 
                                                test_size = 0.2, 
                                                random_state = 999)

# user select classification method
option = st.sidebar.selectbox(
     'Select a Classification method',
      ['K-NN','SVM','Logistic Regression','Gaussian Naive Bayes','Random Forest'])

# KNN
if option == 'K-NN':
  st.write('## Chosen classification method: K-NN')
  
  knn = KNeighborsClassifier()
  knn.fit(Xtrain, ytrain)
  ypred = knn.predict(Xtest)
  
# SVM
elif option == 'SVM':
  st.write('## Chosen classification method: SVM ')
        
  svc = SVC()
  svc.fit(Xtrain, ytrain)
  ypred = svc.predict(Xtest)

# Logistic Regression
elif option == 'Logistic Regression':
  st.write('## Chosen classification method: Logistic Regression')
         
  logreg = LogisticRegression()
  logreg.fit(Xtrain, ytrain)
  ypred = logreg.predict(Xtest)
  
# Gaussian Naive Bayes
elif option == 'Gaussian Naive Bayes':
  st.write('## Chosen classification method: Gaussian Naive Bayes')
         
  nb = GaussianNB()
  nb.fit(Xtrain, ytrain)
  ypred = nb.predict(Xtest)
  
# Random Forest  
elif option == 'Random Forest':
  st.write('## Chosen classification method: Random Forest')
  
  rf = RandomForestClassifier()
  rf.fit(Xtrain, ytrain)
  ypred = rf.predict(Xtest)

# Classification report for selected ml method
st.write('### Classification Report')  
report = classification_report(ytest, ypred, output_dict=True)
cf = pd.DataFrame(report).transpose() 
st.write(cf)



# Price range predictor 
st.write("### Price Range Predictor")

st.write('Predictors are sorted from the highest correlation to the lowest with the price_range')
st.write('Please wait until streamlit stops running after choosing the value for each predictor')

# Input sliders: ram, battery power, px width, px height, internal memory
ram =  st.slider("Ram",min_value=256,max_value=3998,value=400,step=1)
battery_power = st.slider("Battery Power",min_value=501,max_value=1998,value=600,step=1)
px_width = st.slider("Phone Width",min_value=500,max_value=1998,value=600,step=1)   
px_height = st.slider("Phone Height",min_value=20,max_value=1960,value=100,step=1)
int_memory = st.slider("Internal memory",min_value=2.0,max_value=64.0,value=32.0,step=0.1)

# If button is pressed
if st.button("Confirm"):
  # Store inputs into new dataframe
  Xnew = pd.DataFrame([[int_memory,px_height,px_width,battery_power,ram]], 
                     columns = ["int_memory", "px_height", "px_width","battery_power","ram"])
         
  if option == 'K-NN':
    st.write('#### predicted price_range')

    # Get prediction
    prediction = knn.predict(Xnew)[0]

  elif option == 'SVM':
    st.write('#### predicted price_range')

    # Get prediction
    prediction = svc.predict(Xnew)[0]
         

  elif option == 'Logistic Regression':
    st.write('#### predicted price_range')
   
    # Get prediction
    prediction = logreg.predict(Xenw)[0]
  

  elif option == 'Gaussian Naive Bayes':
    st.write('#### predicted price_range')

    # Get prediction
    prediction = nb.predict(X)[0]
  
  
  elif option == 'Random Forest':
    st.write('#### predicted price_range')

    # Get prediction
    prediction = rf.predict(X)[0]
    

  # Output prediction and price range description table
  st.write(f"This mobile phone's predicted price range is {prediction}")
  st.write(pd.DataFrame({'price_range': [0, 1, 2, 3], 'description': ['low cost', 'medium cost', 'high cost', 'very high cost']}))
  col6, col7, col8 = st.columns(3)

  with col6:
      st.empty()

  with col7:
      st.write("-Thank you for using my app! Have a lovely day.-")

  with col8:
      st.empty()


