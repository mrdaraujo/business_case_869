import streamlit as st
import numpy as np
import pandas as pd
import datetime

######### Create the user interface #########
## Use streamlit for the app

# Define text, title and header
st.title('Business Case - Le Wagon 869')
st.header('Prediction and analysis of sales in Favorita stores located in Ecuador')
st.text('Presentation of our project')

#Constante
Const_Store_nbr = 54


# Load csv dataset
data_train = pd.read_csv("/Users/farahboukitab/code/mrdaraujo/business_case_869/business_case_869/data/store-sales-time-series-forecasting/train.csv")
data_stores = pd.read_csv("/Users/farahboukitab/code/mrdaraujo/business_case_869/business_case_869/data/store-sales-time-series-forecasting/stores.csv")
# merge train dataset and stores dataset
data_train_merge_stores = pd.merge(data_train, data_stores, on="store_nbr")
data_train_merge_stores['date'] = pd.to_datetime(data_train_merge_stores['date'])
# Create baseline dataset
data_baseline = data_train_merge_stores
data_baseline['month'] = data_baseline['date'].dt.month
data_baseline['year'] = data_baseline['date'].dt.year


# Create a block on the left - Selection
# 1st column - Selection of information given by the user
with st.sidebar:
    st.header("Please fill up the information")

    #Date selection
    st.subheader('Date selection')
    date_selection = st.date_input("Select a date: ",datetime.date(2013, 1, 1))
    st.write('Selected date:', date_selection)

    #City selection
    st.subheader('City selection')
    city_selection = st.selectbox('Select a city: ',np.sort(data_stores['city'].unique()))
    st.write('Selected city:', city_selection)

    #Retrieve state from city
    state_selection = np.array(data_stores.loc[data_stores['city'] == str(city_selection), 'state'])[0]
    st.write('Your city is in state:', state_selection)

    #Retrieve information from number of store
    array_city_selection_store = np.array(data_stores.loc[data_stores['city'] == city_selection, 'store_nbr'])
    st.write ('Your city has ',len(array_city_selection_store),
    ' out of 54 store')
    percentage = (len(array_city_selection_store)/Const_Store_nbr)*100
    percentage = round(percentage, 2)
    st.write('Your city has ',percentage , '% of stores')

    #Retrieve store list from city and Store selection
    st.subheader('Store selection')
    store_selection = st.selectbox('Select a store from your city: ',array_city_selection_store)
    st.write('Selected store:', store_selection)

    #Family selection
    df_all_family = pd.DataFrame(data_train_merge_stores['family'].unique())
    #df_all_family.columns =['family']
    #df_all_family.index = np.arange(1, len(df_all_family) + 1)
    family_selection = st.multiselect('Select family: ',df_all_family)
    st.write('Selected family:')
    st.table(family_selection)









# Create 2 columns in the middle - Analysis and Prediction
col1, col2 = st.columns(2)
with col1:
    st.markdown("Analysis")
    st.write(city_selection)
    st.image("/Users/farahboukitab/Desktop/Favorita logo.png")

with col2:
    st.markdown("Prediction")


'''
# this slider allows the user to select a number of lines
# to display in the dataframe
# the selected value is returned by st.slider
line_count = st.slider('Select a line count', 1, 100000, 100)
# and used in order to select the displayed lines
display_df = data_baseline.head(line_count)
display_df

#import streamlit as st

# Using object notation
add_selectbox = st.sidebar.selectbox(
    "How would you like to be contacted?",
    ("Email", "Home phone", "Mobile phone")
)
st.write(add_selectbox)

# Using "with" notation
with st.sidebar:
    add_radio = st.radio(
        "Choose a shipping method",
        ("Standard (5-15 days)", "Express (2-5 days)")
    )


st.bar_chart({"data": [1, 5, 2, 6, 2, 1]})

with st.expander("See explanation"):
     st.write("""
         The chart above shows some numbers I picked for you.
         I rolled actual dice for these, so they're *guaranteed* to
         be random.
     """)
     st.image("https://static.streamlit.io/examples/dice.jpg")'''
