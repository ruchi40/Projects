import pandas as pd
import streamlit as st
df=pd.read_csv(r"Netflix_Stock_Prediction_Project\NFLX.csv")
df.head()
x=df.iloc[:, [1,2,3,6]]
y=df.iloc[:,4]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
st.header("Netflix Stock Prediction")
st.sidebar.header("User Input")
open=st.sidebar.slider("Price at which stock opened",0, 2000, 150)
high=st.sidebar.slider("Today's High Price",0, 1000, 100)
low=st.sidebar.slider("Today's Low Price",0, 1000, 150)
volume=st.sidebar.slider("Volume of stocks",0,30000000,100000)

y_pred1=lr.predict([[open,high,low,volume]])
y_pred2=lr.predict(x_test)
st.subheader("Predicted Close price adjusted for splits")
st.write(y_pred1)

from sklearn.metrics import r2_score
st.subheader("Accuracy")
st.write(r2_score(y_test,y_pred2))
