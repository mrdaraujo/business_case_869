from sys import base_prefix
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
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
data_baseline = data_baseline[data_baseline['year']<2017]


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
    ' out of 54 stores')
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

# Create a first part for analysis on city and store
st.header("1)Analysis on city and store")
data_selection_bycity = data_baseline[data_baseline['city'] == city_selection]
data_selection_bycity = pd.DataFrame(data_selection_bycity.groupby(['store_nbr'])['sales'].sum()).reset_index()


if len(array_city_selection_store) > 1 :
    # Plot sales of each store for the selected city
    st.text("Sales of each store for the selected city")
    with st.expander("Expand to have more details: "):
        st.write("The chart above shows evolution on sales for city ", city_selection,
                " for the ",len(array_city_selection_store),
                " stores from 2013 to 2016 ")
        fig, ax = plt.subplots()
        ax.bar(data_selection_bycity['store_nbr'].astype("string"),data_selection_bycity['sales'])
        st.pyplot(fig)
elif len(array_city_selection_store) == 1 :
    st.write('There is only one store in the city that has sold ',int(data_selection_bycity['sales'].sum()),
                    ' products from 2013 to 2016')

#Plot sales of each year for the selected city and store
st.text("Sales of each year for the selected city and store")
with st.expander("Expand to have more details: "):
    st.write("The chart above shows evolution on sales for store number ",store_selection, " from year 2013 to 2016")

    data_selection_bycity_bystore = data_baseline[data_baseline['city'] == city_selection]
    data_selection_bycity_bystore = data_selection_bycity_bystore[data_selection_bycity_bystore['store_nbr'] == store_selection]
    data_selection_bycity_bystore = data_selection_bycity_bystore.groupby(['year'])['sales'].sum().reset_index()

    fig, ax = plt.subplots()
    ax.plot(data_selection_bycity_bystore['year'].astype("string"),data_selection_bycity_bystore['sales'])
    st.pyplot(fig)

#Plot sales for the selected year, store and city
st.text("Sales for the selected year and store")
with st.expander("Expand to have more details: "):
    st.write("The chart above shows evolution on sales for store number ",store_selection, " for year ",
                 date_selection.year)

    data_selection_bymonth_bystore = data_baseline[data_baseline['year'] == date_selection.year]

    data_selection_bymonth_bystore = data_selection_bymonth_bystore[
            data_selection_bymonth_bystore['city'] == city_selection]

    data_selection_bymonth_bystore = data_selection_bymonth_bystore[
            data_selection_bymonth_bystore['store_nbr'] == store_selection]

    data_selection_bymonth_bystore = data_selection_bymonth_bystore.groupby(['month'])['sales'].sum().reset_index()

    fig, ax = plt.subplots()
    ax.plot(data_selection_bymonth_bystore['month'].astype("string"),data_selection_bymonth_bystore['sales'])
    st.pyplot(fig)



#Create a second part to analyse family
st.header("2)Analysis on family")

data_selection_byfamily = data_baseline[data_baseline['city'] == city_selection]
data_selection_byfamily = pd.DataFrame(data_selection_byfamily.groupby(['family'])['sales'].sum()).reset_index()
# Plot sales of each family for the selected city
st.text("Sales of each family for the selected city")
with st.expander("Expand to have more details: "):
    st.write("The chart above shows evolution on sales for 33 families from 2013 to 2016 ")
    fig, ax = plt.subplots()
    plt.xticks(rotation=90)
    ax.bar(data_selection_byfamily['family'],data_selection_byfamily['sales'])
    st.pyplot(fig)


#Plot family of each year for the selected city and store
st.text("Sales for the selected city and store for all families each year")
with st.expander("Expand to have more details: "):
    #4 plots for 2013, 2014, 2015 and 2016 for all families for 1 city and 1 store
    st.write("The charts above show evolution on sales for store number ",store_selection, " for year 2013,2014,2015 and 2016")

    #2013
    data_selection_fam_2013_bycity_bystore = data_baseline[data_baseline['year'] == 2013]
    data_selection_fam_2013_bycity_bystore = data_selection_fam_2013_bycity_bystore[
        data_selection_fam_2013_bycity_bystore['city'] == city_selection]
    data_selection_fam_2013_bycity_bystore = data_selection_fam_2013_bycity_bystore[
        data_selection_fam_2013_bycity_bystore['store_nbr'] == store_selection]
    data_selection_fam_2013_bycity_bystore = data_selection_fam_2013_bycity_bystore.groupby(['family'])['sales'].sum().reset_index()

    #2014
    data_selection_fam_2014_bycity_bystore = data_baseline[data_baseline['year'] == 2014]
    data_selection_fam_2014_bycity_bystore = data_selection_fam_2014_bycity_bystore[
        data_selection_fam_2014_bycity_bystore['city'] == city_selection]
    data_selection_fam_2014_bycity_bystore = data_selection_fam_2014_bycity_bystore[
        data_selection_fam_2014_bycity_bystore['store_nbr'] == store_selection]
    data_selection_fam_2014_bycity_bystore = data_selection_fam_2014_bycity_bystore.groupby(['family'])['sales'].sum().reset_index()

   #2015
    data_selection_fam_2015_bycity_bystore = data_baseline[data_baseline['year'] == 2015]
    data_selection_fam_2015_bycity_bystore = data_selection_fam_2015_bycity_bystore[
        data_selection_fam_2015_bycity_bystore['city'] == city_selection]
    data_selection_fam_2015_bycity_bystore = data_selection_fam_2015_bycity_bystore[
        data_selection_fam_2015_bycity_bystore['store_nbr'] == store_selection]
    data_selection_fam_2015_bycity_bystore = data_selection_fam_2015_bycity_bystore.groupby(['family'])['sales'].sum().reset_index()

  #2016
    data_selection_fam_2016_bycity_bystore = data_baseline[data_baseline['year'] == 2016]
    data_selection_fam_2016_bycity_bystore = data_selection_fam_2016_bycity_bystore[
        data_selection_fam_2016_bycity_bystore['city'] == city_selection]
    data_selection_fam_2016_bycity_bystore = data_selection_fam_2016_bycity_bystore[
        data_selection_fam_2016_bycity_bystore['store_nbr'] == store_selection]
    data_selection_fam_2016_bycity_bystore = data_selection_fam_2016_bycity_bystore.groupby(['family'])['sales'].sum().reset_index()

    fig, axs = plt.subplots(4)
    #fig.suptitle('Vertically stacked subplots')
    axs[0].bar(data_selection_fam_2013_bycity_bystore['family'], data_selection_fam_2013_bycity_bystore['sales'])
    axs[1].bar(data_selection_fam_2014_bycity_bystore['family'], data_selection_fam_2014_bycity_bystore['sales'])
    axs[2].bar(data_selection_fam_2015_bycity_bystore['family'], data_selection_fam_2015_bycity_bystore['sales'])
    axs[3].bar(data_selection_fam_2016_bycity_bystore['family'], data_selection_fam_2016_bycity_bystore['sales'])
    st.pyplot(fig)
