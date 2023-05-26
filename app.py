import datetime
import numpy as np
import pandas as pd
import pickle
import streamlit as st

model = pickle.load(open("model_pickle.pkl",'rb'))

st.title('BIKE SHARING DEMAND ANALYSIS')

season = st.selectbox('Choose the season',('Winter', 'Spring', 'Summer', 'Fall'))
month = st.selectbox(
    'Choose Month',
    ('January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'))
hour = st.slider('Choose Hour', 0, 23, 8)
holiday = st.radio('Is today holiday?',('Yes','No'))
weekday = st.selectbox(
    'Choose Day',
    ('Sunday','Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'))
workingday = st.radio('Is workingday?',('Yes','No'))
weather = st.selectbox('Choose the weather',('Clear','Mist','Light Snow','Heavy Rain'))
temperature = st.slider('Set the temperature',0.0,0.5,1.0)
humidity = st.slider('Set the humidity',0.0,0.5,1.0)

# def predict():
#     row = np.array([season,month,hour,holiday,weekday,workingday,weather,temperature,humidity])
#     y_pred = model.predict([row])

def season_conv():
    if season == 'Winter':
        return 1
    elif season == 'Spring':
        return 2
    elif season == 'Summer':
        return 3
    elif season == 'Fall':
        return 4

def month_conv():
    month_dict = {"January": 0, "February": 1, "March": 2, "April": 3, 
                  "May": 4, "June": 5, "July": 6, "August": 7, 
                  "September": 8, "October": 9, "November": 10, "December": 11}
    return month_dict.get(month.title(), 0)

def week_conv():
    day_dict = {"Sunday": 0, "Monday": 1, "Tuesday": 2, "Wednesday": 3,
                "Thursday": 4, "Friday": 5, "Saturday": 6}
    return day_dict.get(weekday.title(), 0)

def weather_conv():
    weather_dict = {"Clear":1,"Mist":2, "Light Snow":3, "Heavy Rain":4}
    return weather_dict.get(weather.title(), 1)


def one_hot_encoding(data, column):
    data = pd.concat([data, pd.get_dummies(data[column], prefix=column, drop_first=True)], axis=1)
    data = data.drop([column], axis=1)
    return data





if st.button('Predict'):
    df = pd.read_csv('hour.csv')
    df = df.rename(columns={'weathersit':'weather',
                       'yr':'year',
                       'mnth':'month',
                       'hr':'hour',
                       'hum':'humidity',
                       'cnt':'count'})
    df['count'] = np.log(df['count'])
    df_oh = df
    df_oh = df_oh.drop(columns=['windspeed', 'casual', 'registered', 'count', 'instant', 'dteday', 'year'], axis=1)
    
    data = [
        season_conv(),
        month_conv(),
        hour,
        1 if holiday=='Yes' else 0,
        week_conv(),
        1 if workingday=="Yes" else 0,
        weather_conv(),
        temperature,
        humidity
    ]
    df_oh.loc[len(df_oh.index)] = data
    
    def one_hot_encoding(data, column):
        data = pd.concat([data, pd.get_dummies(data[column], prefix=column, drop_first=True)], axis=1)
        data = data.drop([column], axis=1)
        return data

    cols = ['season','month','hour','holiday','weekday','workingday','weather']

    for col in cols:
        df_oh = one_hot_encoding(df_oh, col)

    try:
        df_oh = df_oh.drop(['month_1.0'], axis=1)
    finally:
        y_pred = model.predict([df_oh.loc[len(df_oh.index) -1].values.tolist()])
        st.success(int(np.exp(y_pred[0])))
    
    # row = np.array([
    #     season_conv(),
    #     month_conv(),
    #     hour,
    #     1 if holiday=='Yes' else 0,week_conv(),
    #     1 if workingday=="Yes" else 0,
    #     weather_conv(),
    #     temperature,
    #     humidity])

    # y_pred = model.predict([row])
    # st.success(y_pred)