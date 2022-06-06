##from sys import base_prefix
##from turtle import width
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import plotly.express as px

######### Create the user interface #########
## Use streamlit for the app

# Define text, title and header
st.header('Business Case - Le Wagon 869')
st.subheader('How to analyse and predict sales in a company with different stores, cities and product types')
st.text('Case study: 54 stores in 22 cities in Ecuador for a commercial company')

#Constante
Const_Store_nbr = 54
api_token = px.set_mapbox_access_token('pk.eyJ1IjoibXJkYXJhdWpvIiwiYSI6ImNsM3hsY2c2NzAzcHEzYm1oYmliZHc5aXoifQ.1E3p2I8p8bEkHSPJDzUXWQ')

# Load csv dataset
data_train = pd.read_csv("/Users/farahboukitab/code/mrdaraujo/business_case_869/business_case_869/data/store-sales-time-series-forecasting/train.csv")
data_stores = pd.read_csv("/Users/farahboukitab/code/mrdaraujo/business_case_869/business_case_869/data/store-sales-time-series-forecasting/stores.csv")

df_heatmap = pd.read_csv("/Users/farahboukitab/code/mrdaraujo/business_case_869/business_case_869/data/store-sales-time-series-forecasting/Heatmap.csv")
df_heatmap.rename(columns = {'Unnamed: 0':'city', 'Unnamed: 1':'Lat', 'Unnamed: 2':'Lon', 'Unnamed: 3':'Weight'}, inplace = True)
df_heatmap.drop([0], axis=0, inplace=True)

# merge train dataset and stores dataset
data_train_merge_stores = pd.merge(data_train, data_stores, on="store_nbr")
data_train_merge_stores['date'] = pd.to_datetime(data_train_merge_stores['date'])
#Drop year 2017 because it is not full
data_train_merge_stores = data_train_merge_stores.drop(data_train_merge_stores[data_train_merge_stores['year']==2017].index, axis=0, inplace=False)

#merge train dataset and stores dataset with heatmap
map_base = pd.merge(data_train_merge_stores, df_heatmap, on='city').drop(columns=['Weight', 'type', 'cluster', 'state', 'onpromotion'])
map_base['date'] = pd.to_datetime(map_base['date'])
map_base['month'] = map_base['date'].dt.month
map_base['year'] = map_base['date'].dt.year
#map_base = map_base.drop(map_base[map_base['year']==2017].index, axis=0, inplace=False)
map_base['Lat'] = pd.to_numeric(map_base['Lat'])
map_base['Lon'] = pd.to_numeric(map_base['Lon'])
sales_city_year = pd.DataFrame(map_base.groupby(['city', 'Lat', 'Lon'])['sales'].sum()).reset_index().sort_values(by='sales', ascending=False)

# Create baseline dataset
data_baseline = data_train_merge_stores
data_baseline['month'] = data_baseline['date'].dt.month
data_baseline['year'] = data_baseline['date'].dt.year
#data_baseline = data_baseline[data_baseline['year']<2017]

#Create dataframe for top 5 cities
map_base_top_five = map_base[map_base['city'].isin(['Quito', 'Guayaquil', 'Cuenca', 'Ambato', 'Santo Domingo'])]
map_base_top_five = pd.DataFrame(map_base_top_five.groupby(['city', 'year'])['sales'].sum()).reset_index().sort_values(by=[ 'year','sales', 'city' ], ascending=[True,False , True])
map_base_top_five['year'] = map_base_top_five['year'].astype('string')

#Create dataframe with cities and number of stores in each city
df_stores_city = pd.DataFrame(data_stores.groupby(['city'])['store_nbr'].count())
df_stores_city = df_stores_city.reset_index().sort_values(by='store_nbr', ascending=False)


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

st.header("1) Data visualization")

# Create a first part for analysis on cities
with st.expander("1.1) General analysis on sales and cities"):
    #col1, col2 = st.columns(2)
    #with col1:
    st.caption("a) Map with cities and sales")
    px.set_mapbox_access_token('pk.eyJ1IjoibXJkYXJhdWpvIiwiYSI6ImNsM3hsY2c2NzAzcHEzYm1oYmliZHc5aXoifQ.1E3p2I8p8bEkHSPJDzUXWQ')
    df = px.data.election_geojson()
    fig = px.scatter_mapbox(data_frame=sales_city_year, lat="Lat", lon="Lon", color="city", size="sales",
                  color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=5, mapbox_style="carto-darkmatter")
    st.write(fig)
    #with col2:
    st.caption("b) Sales for top 5 cities")
    fig = px.bar(map_base_top_five, x='city', y='sales', color='year')
    st.write(fig)
    st.caption("c) Prediction based on the city")
    st.write("Let's predict the number of sales for the city ", city_selection, " for 12 months of the year ", date_selection.year)


#Create a second part for analysis on stores
with st.expander("1.2)General analysis on sales and stores") :
    st.caption("a) Number on stores on each city")
    fig = px.bar(data_frame=df_stores_city, x='city', y='store_nbr')
    st.write(fig)



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

    #fig, axs = plt.subplots(4)
    ##fig.suptitle('Vertically stacked subplots')
    #axs[0].plot(data_selection_fam_2013_bycity_bystore['family'], data_selection_fam_2013_bycity_bystore['sales'])
    #axs[1].plot(data_selection_fam_2014_bycity_bystore['family'], data_selection_fam_2014_bycity_bystore['sales'])
    #axs[2].plot(data_selection_fam_2015_bycity_bystore['family'], data_selection_fam_2015_bycity_bystore['sales'])
    #axs[3].plot(data_selection_fam_2016_bycity_bystore['family'], data_selection_fam_2016_bycity_bystore['sales'])
    #st.pyplot(fig)


    fig, ax = plt.subplots()

    # Plotting the curves in the same graph
    plt.plot(data_selection_fam_2013_bycity_bystore['family'],
             data_selection_fam_2013_bycity_bystore['sales'], color='r', label='2013')

    plt.plot(data_selection_fam_2014_bycity_bystore['family'],
             data_selection_fam_2014_bycity_bystore['sales'], color='g', label='2014')

    plt.plot(data_selection_fam_2015_bycity_bystore['family'],
             data_selection_fam_2015_bycity_bystore['sales'], color='blue', label='2015')

    plt.plot(data_selection_fam_2016_bycity_bystore['family'],
             data_selection_fam_2016_bycity_bystore['sales'], color='black', label='2016')

    plt.xticks(rotation=90)
    plt.xlabel("Family")
    plt.ylabel("Sales")
    plt.title("Comparison of sales for selected store")
    plt.legend()

    st.pyplot(fig)