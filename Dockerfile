FROM python:3.8.6-buster

COPY api /api
COPY package /package
COPY model_baseline.joblib /model_baseline.joblib
COPY model_city.joblib /model_city.joblib
COPY model_family.joblib /model_family.joblib
COPY model_store.joblib /model_store.joblib
COPY requirements.txt /requirements.txt

RUN pip install -r requirements.txt

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
