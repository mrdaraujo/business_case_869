import pandas as pd
import joblib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

@app.get("/predict")
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
    pipe = joblib.load('model.joblib')

    # make prediction
    results = pipe.predict(X)

    # convert response from numpy to python type
    pred = float(results[0])

    return dict(sales=pred)
