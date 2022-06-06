from re import S
from unittest import result
import pandas as pd
import numpy as np
import joblib

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Union



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return dict(greeting="hello test")

@app.get("/predict-single")
def predict(year,
            month,
            family,
            store_nbr,
            onpromotion):

    # build X ⚠️ beware to the order of the parameters ⚠️
    X = pd.DataFrame(dict(
                year=[int(year)],
                month=[int(month)],
                family=[str(family)],
                store_nbr=[int(store_nbr)],
                onpromotion=[int(onpromotion)]
            ))

    # pipeline = get_model_from_gcp()
    pipe = joblib.load('model_baseline.joblib')

    # make prediction and convert log back to number
    results = pipe.predict(X)
    results = pd.DataFrame(results)
    results = np.exp(results)
    results = results.to_numpy()

    # convert response from numpy to python type
    #pred = float(results[0])
    pred = results[0][0]

    return dict(sales=pred)

@app.get("/predict-baseline-year")
def predict(year: Union[List[str], None] = Query(default=None),
            month: Union[List[str], None] = Query(default=None),
            family: Union[List[str], None] = Query(default=None),
            store_nbr: Union[List[str], None] = Query(default=None),
            onpromotion: Union[List[str], None] = Query(default=None)
        ):

    query_items = {
            'year': year,
            'month': month,
            'family': family,
            'store_nbr': store_nbr,
            'onpromotion':onpromotion
        }

    # build X ⚠️ beware to the order of the parameters ⚠️
    X = pd.DataFrame(dict(
                year=[int(i) for i in query_items['year']],
                month=[int(i) for i in query_items['month']],
                family=[str(i) for i in query_items['family']],
                store_nbr=[int(i) for i in query_items['store_nbr']],
                onpromotion=[int(i) for i in query_items['onpromotion']]
            ))

    #X = pd.DataFrame(query_items)

    # pipeline = get_model_from_gcp()
    pipe = joblib.load('model_baseline.joblib')

    #breakpoint()
    # make prediction
    results = pipe.predict(X).tolist()
    results = [np.around(np.exp(i),decimals=2) for i in results]
    results = np.array(results)

    # convert response from numpy to python type
    pred = dict(enumerate(results.flatten(), 1))

    return dict(sales_baseline_year=pred)

@app.get("/predict-city-year")
def predict(year: Union[List[str], None] = Query(default=None),
            month: Union[List[str], None] = Query(default=None),
            city: Union[List[str], None] = Query(default=None)
        ):

    query_items = {
            'year': year,
            'month': month,
            'city': city
        }

    # build X ⚠️ beware to the order of the parameters ⚠️
    X = pd.DataFrame(dict(
                year=[int(i) for i in query_items['year']],
                month=[int(i) for i in query_items['month']],
                city=[str(i) for i in query_items['city']]
            ))

    #X = pd.DataFrame(query_items)

    # pipeline = get_model_from_gcp()
    pipe = joblib.load('model_city.joblib')

    #breakpoint()
    # make prediction
    results = pipe.predict(X).tolist()
    results = [np.around(np.exp(i),decimals=2) for i in results]
    results = np.array(results)

    # convert response from numpy to python type
    pred = dict(enumerate(results.flatten(), 1))

    return dict(sales_city_year=pred)

@app.get("/predict-store-year")
def predict(year: Union[List[str], None] = Query(default=None),
            month: Union[List[str], None] = Query(default=None),
            city: Union[List[str], None] = Query(default=None),
            store_nbr: Union[List[str], None] = Query(default=None)
        ):

    query_items = {
            'year': year,
            'month': month,
            'city': city,
            'store_nbr': store_nbr,
        }

    # build X ⚠️ beware to the order of the parameters ⚠️
    X = pd.DataFrame(dict(
                year=[int(i) for i in query_items['year']],
                month=[int(i) for i in query_items['month']],
                city=[str(i) for i in query_items['city']],
                store_nbr=[int(i) for i in query_items['store_nbr']]
            ))

    #X = pd.DataFrame(query_items)

    # pipeline = get_model_from_gcp()
    pipe = joblib.load('model_store.joblib')

    #breakpoint()
    # make prediction
    results = pipe.predict(X).tolist()
    results = [np.around(np.exp(i),decimals=2) for i in results]
    results = np.array(results)

    # convert response from numpy to python type
    pred = dict(enumerate(results.flatten(), 1))

    return dict(sales_baseline_year=pred)

@app.get("/predict-family-year")
def predict(year: Union[List[str], None] = Query(default=None),
            month: Union[List[str], None] = Query(default=None),
            city: Union[List[str], None] = Query(default=None),
            store_nbr: Union[List[str], None] = Query(default=None),
            family: Union[List[str], None] = Query(default=None)
        ):

    query_items = {
            'year': year,
            'month': month,
            'city': city,
            'store_nbr': store_nbr,
            'family': family
        }

    # build X ⚠️ beware to the order of the parameters ⚠️
    X = pd.DataFrame(dict(
                year=[int(i) for i in query_items['year']],
                month=[int(i) for i in query_items['month']],
                city=[str(i) for i in query_items['city']],
                store_nbr=[int(i) for i in query_items['store_nbr']],
                family=[str(i) for i in query_items['family']]
            ))

    #X = pd.DataFrame(query_items)

    # pipeline = get_model_from_gcp()
    pipe = joblib.load('model_family.joblib')

    #breakpoint()
    # make prediction
    results = pipe.predict(X).tolist()
    results = [np.around(np.exp(i),decimals=2) for i in results]
    results = np.array(results)

    # convert response from numpy to python type
    pred = dict(enumerate(results.flatten(), 1))

    return dict(sales_baseline_year=pred)
