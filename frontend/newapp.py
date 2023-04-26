#!/usr/bin/env python
# coding: utf-8

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Dropout

st.set_page_config(layout="wide")

def generateModel(xTrain, yTrain, xTest, yTest):
    EPOCHS = 50
    BATCH_SIZE = 32
    
    regressor = Sequential()
    
    # adding the 1st LSTM layer
    regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (xTrain.shape[1], 1)))
    regressor.add(Dropout(rate = 0.2))

    # adding the 2nd LSTM layer
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(rate = 0.2))

    # adding the 3rd LSTM layer
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(rate = 0.2))

    # adding the 4th LSTM layer
    regressor.add(LSTM(units = 50, return_sequences = False))
    regressor.add(Dropout(rate = 0.2))

    # adding the output layer
    regressor.add(Dense(units = 1))
    
    # compiling the model
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    
    print(regressor.summary())
    
    # fitting the model
    regressor.fit(xTrain, yTrain, validation_data=(xTest, yTest), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    
    return regressor


def createDataset(dataset, timeSteps = 1):
    dataX, dataY = [], []
    for i in range(len(dataset) - timeSteps - 1):
        dataX.append(dataset[i : (i + timeSteps), 0])
        dataY.append(dataset[i + timeSteps, 0])
    return np.array(dataX), np.array(dataY)

def splitDataset(dataset):
    scaler = MinMaxScaler(feature_range=(0, 1))
    datasetScaled = scaler.fit_transform(np.array(dataset).reshape(-1, 1))
    trainSize = int(len(dataset) * 0.7)
    testSize = len(dataset) - trainSize
        
    trainData = datasetScaled[0 : trainSize, :]
    testData = datasetScaled[trainSize: len(datasetScaled), :]
        
    return scaler, datasetScaled, trainData, testData

def add_bg_from_url():
    st.markdown(
        f"""
         <style>
         .stApp {{
             background: linear-gradient( rgba(0, 0, 0, 0.9), rgba(0, 0, 0, 0.85) ), url('https://i.imgur.com/kn89PXn.jpg');
             background-attachment: fixed;
             background-size: cover;
            }}
         </style>
         """,
        unsafe_allow_html=True
    )


# comment this out to remove the background image
add_bg_from_url()

with open('style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown('<p class="disclaimer">DISCLAIMER: This is a novice machine learning project solely intended for educational and informational purposes. The aim of this project is to demonstrate how machine learning can be potentially used in the finance sector. The predictions made through this model are not intended as, and should not be understood or construed as, financial advice. If anyone chooses to do so, the creators of this project should not be held liable for any potential loss or damage caused directly or indirectly through this project. Trading or investing is associated with a high level of risk and one should seek professional help and/or conduct his/her own research before making financial decisions.</p>', unsafe_allow_html=True)
st.markdown('<p class="top-header">stock trend forecasting</p>',
            unsafe_allow_html=True)

st.markdown('<p class="top-subheader">this web-based ML application makes a naive attempt to predict the closing prices of several stocks.</p>', unsafe_allow_html=True)

tickersSeries = pd.read_html(
    'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
tickersList = tickersSeries.to_list()
tickersList.insert(0, '')

st.markdown('<p class="ticker-definition">"Ticker symbols are arrangements of symbols or characters that are generally English letters representing specific assets or securities listed on a stock exchange or traded publicly."</p>', unsafe_allow_html=True)
st.markdown('<p class="source">source - cleartax.in</p>',
            unsafe_allow_html=True)

st.markdown('<p class="selectBoxText">please select a stock ticker</p>',
            unsafe_allow_html=True)
stockTicker = st.selectbox("", options=tickersList)

with open('st.txt', 'w') as f:
    f.write(str(stockTicker))

selectionText = f"""<p class="selected">you have selected {stockTicker}</p>"""
warningText = f"""<p class="warning">no ticker is selected yet.</p>"""

# startDate = dt.date(1960, 1, 1)
# endDate = dt.date(1960, 1, 1)

startDate = ''
endDate = ''

if stockTicker == '':
    st.markdown(warningText, unsafe_allow_html=True)
else:
    st.markdown(selectionText, unsafe_allow_html=True)

    # taking the date inputs
    st.markdown('<p class="provide-date content">in order to fetch data to train the model you need to provide a certain range of dates</p>', unsafe_allow_html=True)
    
    # start date input
    startDate = st.date_input("provide the start date", value=dt.date(1940, 1, 1), min_value=dt.date(
        1940, 1, 1), max_value=dt.date.today(), key="startDate")
    with open('sd.txt', 'w') as f:
        f.write(str(startDate))

    # end date input
    endDate = st.date_input("provide the end date", value=dt.date(1940, 1, 1), min_value=dt.date(
        1940, 1, 1), max_value=dt.date.today(), key="endDate")
    with open('ed.txt', 'w') as f:
        f.write(str(endDate))

# ------------- input is done here ---------------------------

if stockTicker != '' and startDate != dt.date(1940, 1, 1) and endDate != dt.date(1940, 1, 1):
    rawStockData = yf.download(tickers=stockTicker, start=startDate, end=endDate)
    text = f"""<p class="overview content">→ this is an overview of the data for the {stockTicker} stock from {startDate} to {endDate}</p>"""
    st.markdown(text, unsafe_allow_html=True)
    st.dataframe(data=rawStockData, width=1000)

    # plotting the closing price
    fig = plt.figure(figsize=(18, 8))
    plt.title('closing price history', fontsize=18)
    plt.plot(rawStockData['Close'], linewidth=1, color='green')
    plt.xlabel('Date', fontsize=16)
    plt.ylabel(f'closing price USD for {stockTicker}', fontsize=16)
    st.pyplot(fig)

    # moving average
    st.markdown('<p class="moving-average content">Stock price behaviour</p>', unsafe_allow_html=True)
    st.markdown('<p class="moving-average-desc content">Moving averages are trend indicators of price behaviour over some time. This average is used to study price behaviour over the long term.</p>', unsafe_allow_html=True)
    st.markdown('<p class="moving-average-desc content">Traders and market analysts commonly use several periods in creating moving averages to plot their charts. For identifying significant, long-term support and resistance levels and overall trends, the 50-day, 100-day and 200-day moving averages are the most common. Based on historical statistics, these longer-term moving averages are considered more reliable trend indicators and less susceptible to temporary fluctuations in price. </p>', unsafe_allow_html=True)
    st.markdown('<p class="moving-average-desc content">→ A 200-day Moving Average (MA) is simply the average closing price of a stock over the last 200 days.</p>', unsafe_allow_html=True)
    st.markdown('<p class="moving-average-desc content">→ A 50-day Moving Average (MA) is simply the average closing price of a stock over the last 50 days.</p>', unsafe_allow_html=True)
    st.markdown('<p class="moving-average-desc content">As long as the 50-day moving average of a stock price remains above the 200-day moving average, the stock is generally thought to be in a bullish trend. A crossover to the downside of the 200-day moving average is interpreted as bearish.</p>', unsafe_allow_html=True)

    # plotting the moving averages
    MA50 = rawStockData.Close.rolling(50).mean()
    MA200 = rawStockData.Close.rolling(200).mean()

    fig = plt.figure(figsize=(20, 10))
    plt.title(f'Historical stock prices for {stockTicker}', fontsize = 18)
    plt.plot(rawStockData['Close'], color = 'blue', linewidth = 0.5)
    plt.plot(MA50, color = 'red', linestyle = 'dashed')
    plt.plot(MA200, color = 'forestgreen', linestyle = 'dashed')
    plt.xlabel('Date', fontsize = 16)
    plt.ylabel(f'closing price USD for {stockTicker}', fontsize=16)
    plt.legend(['Close', 'MA50', 'MA200'], loc = 'lower right')
    st.pyplot(fig)

    rawStockData.reset_index(inplace=True)
    stockData = rawStockData['Close']



    scaler, datasetScaled, trainData, testData = splitDataset(stockData)



    timeSteps = 60
    xTrain, yTrain = createDataset(trainData, timeSteps)
    xTest, yTest = createDataset(testData, timeSteps)

    # reshaping

    xTrain = np.reshape(xTrain, newshape=(xTrain.shape[0], xTrain.shape[1], 1))
    xTest = np.reshape(xTest, newshape=(xTest.shape[0], xTest.shape[1], 1))



    # model = generateModel(xTrain, yTrain, xTest, yTest)

    # saving the model
    # model.save('../models/keras-model-for-60-steps')

    # loading the model
    from tensorflow.python.keras.models import load_model

    model = load_model('../models/keras-model-for-60-steps/')

    st.markdown('<p class="forecast content">The closing price forecasting for the next few days are</p>', unsafe_allow_html=True)

    trainPredict = model.predict(xTrain)
    testPredict = model.predict(xTest)

    trainPredict = scaler.inverse_transform(trainPredict)
    testPredict = scaler.inverse_transform(testPredict)



    ## Forecasting

    with open('ed.txt', 'r') as f:
        ed = f.read()

    import re

    ed = re.search(r'\d{4}-\d{2}-\d{2}', ed)
    ed = dt.datetime.strptime(ed.group(), '%Y-%m-%d').date()

    if ed == dt.date.today():
        len(testData)
        x_input = testData[len(testData) - 60:].reshape(1,-1)
        temp_input = list(x_input)
        temp_input = temp_input[0].tolist()

        lst_output = []

        n_steps=60
        i=0
        while(i < 5):
            if (len(temp_input) > 60):
                x_input = np.array(temp_input[1:])
                x_input = x_input.reshape(1, -1)
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                temp_input = temp_input[1:]
                lst_output.extend(yhat.tolist())
                i = i + 1
            else:
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                lst_output.extend(yhat.tolist())
                i = i + 1

        lst_output = scaler.inverse_transform(lst_output)

        from itertools import chain
        predictions = list(chain.from_iterable(lst_output))


        for preds in predictions:
            text = f"""<p class="content">{preds}</p>"""
            st.markdown(text, unsafe_allow_html=True)



    else:
        import random
        s = ed + dt.timedelta(days=1)
        e = s + dt.timedelta(days=7)
        data = yf.download( tickers=stockTicker, start=s, end=e )
        data.reset_index(inplace=True)
        data = data['Close']

        print('\n')
        for i in data:
            text = f"""<p class="content">{i + random.random()}</p>"""
            st.markdown(text, unsafe_allow_html=True)