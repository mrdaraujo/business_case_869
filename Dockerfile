FROM python:3.8.6-buster

COPY app.py /app.py
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD streamlit run app.py
