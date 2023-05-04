import numpy as np
import pickle
import streamlit as st

mp = pickle.load(open("model_pickle",'rb'))

st.title('BIKE SHARING DEMAND ANALYSIS')
st.image('/home/swathy/Downloads/Bike_Sharing/bike.webp',width=500)
season = st.selectbox('Choose the season',('winter', 'spring', 'summer', 'fall'))
month = st.date_input('Select the month',(datetime.date('jan','feb','mar')))
hour = st.time_input('Set an alarm for', datetime.time(0, 23))
holiday = st.radio('Is today holiday?',('Yes','No'))
weekday = st.radio('Is weekday?',('Yes','No'))
workingday = st.radio('Is workingday?',('yes','No'))
weather = st.selectbox('Choose the weather',('Clear','Mist','Snow','Heavy rain'))
temperature = st.select_slider('Set the temperature',0,1)
humidity = st.select_slider('Set the humidity',0,1)

def predict():
    row = np.array([season,month,hour,holiday,weekday,workingday,weather,temperature,humidity])
    y_pred = model.predict(x_test)

st.button('Predict', on_click=predict)    
