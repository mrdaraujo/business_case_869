from sklearn import set_config; set_config(display='diagram')
from package.etl import *
from package.train_baseline import TrainBaseline
from package.train_city import TrainCity
from package.train_store import TrainStore
from package.train_family import TrainFamily
import pandas as pd
import numpy as np
import requests
import joblib
import json
from fbprophet import Prophet
import matplotlib.pyplot as plt


 


def family(sales_and_stores):

    forecasts = []
    for category in sales_and_stores['family'].unique():
        print(category)
        # creating a new variable
        tmp_df_prep = sales_and_stores[sales_and_stores['family']== category]
        
        # Transforming the column in date time
        tmp_df_prep['ds'] = pd.to_datetime(tmp_df_prep['ds'])
 
        # creating a temporary df to the the training and prediction
        tmp_df = tmp_df_prep.groupby(by='ds').sum().drop(columns=["store_nbr", "onpromotion"]).reset_index()
        
        # defining the train/test 
        train = tmp_df.iloc[:1457]
        test = tmp_df.iloc[1458:]
        
        # Instantiating the FB Prophet model
        model = Prophet(seasonality_mode='multiplicative')

        # fitting the model on the train test
        model.fit(train)

        # Making a prediction on the test set
        forecast_test = model.predict(test)
        forecast_test.head()

        #plotting the test forecast
        model.plot(forecast_test)

        # making a future prediction
        future = model.make_future_dataframe(periods=12, freq='MS')  #period of 12 months
        forecast_future = model.predict(future)
        forecast_future.head()

        # ploting the prediction
        model.plot(forecast_future)

        # plotting the combination of analysis inside FB Prophet
        model.plot_components(fcst=forecast_future)

        forecasts.append(forecast_future)
    
    return forecasts

if __name__ == "__main__":
    #creating the dataframes

    df_sales = pd.read_csv("gs://business-case/train.csv")
    df_stores = pd.read_csv("gs://business-case/stores.csv")
    sales_and_stores = pd.merge(df_sales, df_stores, on='store_nbr')

    #making it in the patter of
    sales_and_stores = sales_and_stores.reset_index().rename(columns={'date': 'ds', 'sales':'y'}).drop(columns=["type", "cluster", "index", "id", "state"])


    forecasts = family(sales_and_stores)

    
    