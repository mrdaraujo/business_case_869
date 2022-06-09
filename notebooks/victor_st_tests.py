import streamlit as st
import requests

def ApiTest(year, city, store_nbr, family):
    year=year
    month=['1','2','3','4','5','6','7','8','9','10','11','12']
    city=city
    store_nbr=store_nbr
    family =family

    url = 'https://image-bc869-v2-1-ob6evlacjq-ew.a.run.app/predict-family-year'
    params = {
        'year': year,
        'month': month,
        'city': city,
        'store_nbr': store_nbr,
        'family': family
    }

    response = requests.get(url, params=params)

    return response.json()


year=['2017','2017','2017','2017','2017','2017','2017','2017','2017','2017','2017','2017']
city=['Ambato','Ambato','Ambato','Ambato','Ambato','Ambato','Ambato','Ambato','Ambato','Ambato','Ambato','Ambato']
store_nbr=['23','23','23','23','23','23','23','23','23','23','23','23']
family=family = ['BEAUTY', 'BEAUTY', 'BEAUTY', 'BEAUTY', 'BEAUTY', 'BEAUTY','BEAUTY', 'BEAUTY', 'BEAUTY', 'BEAUTY', 'BEAUTY', 'BEAUTY']

api_return = ApiTest(year, city, store_nbr, family)

# Define text, title and header
st.header(
    api_return
)

api_return
