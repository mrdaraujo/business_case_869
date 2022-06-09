FROM waguii/python-lewagon

COPY app_test.py /app_test.py
COPY requirements.txt /requirements.txt


RUN pip install --upgrade pip
RUN pip install -r requirements.txt


CMD streamlit run app_test.py