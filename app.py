import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import plotly.express as px
import seaborn as sns
import datetime
import requests
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from PIL import Image
from fbprophet import Prophet


#Constants
Const_Store_nbr = 54
api_token = px.set_mapbox_access_token('pk.eyJ1IjoibXJkYXJhdWpvIiwiYSI6ImNsM3hsY2c2NzAzcHEzYm1oYmliZHc5aXoifQ.1E3p2I8p8bEkHSPJDzUXWQ')
#Const_LocalPath = "/Users/farahboukitab/code/mrdaraujo/business_case_869/business_case_869/data/store-sales-time-series-forecasting/"
Const_LocalPath = 'https://storage.googleapis.com/business-case/Production%20files/'
#Const_LocalPath = "gs://business-case/Production files/"
Const_url_predict_city = 'https://image-bc869-v2-1-ob6evlacjq-ew.a.run.app/predict-city-year'
Const_url_predict_store = 'https://image-bc869-v2-1-ob6evlacjq-ew.a.run.app/predict-store-year'
Const_url_predict_family = 'https://image-bc869-v2-1-ob6evlacjq-ew.a.run.app/predict-family-year'
Const_month_predict = ['1','2','3','4','5','6','7','8','9','10','11','12']

#Load image
Image_LeWagon_Red = Image.open('images/LeWagon_Red.png')
#Image_LeWagon_White = Image.open(Const_LocalPath + 'LeWagon_White.png')
#Image_Store= Image.open(Const_LocalPath + 'Store.jpeg')
#Image_Favorita = Image.open(Const_LocalPath + 'Favorita.png')
Image_Data_visualization = Image.open('images/Data visualization.jpeg')

#Set titles for all pages
st.set_page_config(layout="wide")
col1, col_ = st.columns([1,5])
with col1:
    st.image(Image_LeWagon_Red, caption="")
with col_:
    title_AnalysisPrediction = '<p style="font-family:arial; \
    background-color:red;text-align:center; \
    color:white; font-size: 255%;">Analysis and Prediction</p>'
    title_SalesStoreCompanies = '<p style="font-family:arial; \
    background-color:red;text-align:center; \
    color:white; font-size: 255%;">of sales for grocery retailer</p>'
    st.markdown(title_AnalysisPrediction,unsafe_allow_html=True )
    st.markdown(title_SalesStoreCompanies,unsafe_allow_html=True )

row1_1, row1_2 = st.columns((2, 3))
with row1_1:
    text_Product = '<td style="font-family: arial; font-size: 15px; \
        line-height: 21px; color:blue; text-align: center; vertical-align: middle">\
            <p style="margin: 0 0 0 0; font-size: 17px; line-height: 23px; border-color:red\
  "><strong>Our Product : <br> An interactive tool that presents an analysis of our client business\
  on all different levels  \
           </strong></p></td>'
    st.markdown(text_Product,unsafe_allow_html=True )
with row1_2:
    text_CaseStudy = '<td style="font-family: arial; font-size: 15px; \
        line-height: 21px; color: #3C3F44; text-align: center; vertical-align: middle">\
            <p style="margin: 0 0 0 0; font-size: 17px; line-height: 23px; border-color:#3C3F44\
  "><em>Case study : <br> Examining sales from 2013 to 2016 for a company located in Ecuador. \
           The company is in 22 cities \
    with 54 stores and sell 33 categories of products</em></p></td>'
    st.markdown(text_CaseStudy,unsafe_allow_html=True )

# Load csv dataset - Check the cache
# Initialization
if 'count' not in st.session_state :
    st.session_state.count = 0
    @st.cache()
    def getting_csv():
        # Getting all csv files that are not going to change
        df_heatmap = pd.read_csv(Const_LocalPath + "Heatmap.csv")
        data_train_merge_stores = pd.read_parquet("data_train_merge_stores.parquet")
        map_base = pd.read_parquet("map_base.parquet")
        sales_city_year = pd.read_csv(Const_LocalPath +"sales_city_year.csv")
        map_base_top_five = pd.read_csv(Const_LocalPath + "map_base_top_five.csv")
        df_stores_city = pd.read_csv(Const_LocalPath + "df_stores_city.csv")
        df_MaxSales_BiggestStore = pd.read_csv(Const_LocalPath + "df_MaxSales_BiggestStore.csv")
        df_MaxSales_BiggestStore['store_nbr'] = df_MaxSales_BiggestStore['store_nbr'].astype('string')
        df_stores_top_five = pd.read_csv(Const_LocalPath + "df_stores_top_five.csv")
        df_stores_top_five['store_nbr'] = df_stores_top_five['store_nbr'].astype('string')
        data_family_peryear = pd.read_csv(Const_LocalPath + "data_family_peryear.csv")
        data_family_peryear['year'] = data_family_peryear['year'].astype('string')
        data_family_allyear = pd.read_csv(Const_LocalPath + "data_family_allyear.csv")
        data_top5family = pd.read_csv(Const_LocalPath + "data_top5family.csv")
        return df_heatmap, data_train_merge_stores,map_base,sales_city_year,map_base_top_five,df_stores_city,df_MaxSales_BiggestStore, df_stores_top_five,data_family_peryear, data_family_allyear, data_top5family
    st.session_state.df_heatmap, st.session_state.data_train_merge_stores,st.session_state.map_base,st.session_state.sales_city_year,st.session_state.map_base_top_five,st.session_state.df_stores_city,st.session_state.df_MaxSales_BiggestStore, st.session_state.df_stores_top_five,st.session_state.data_family_peryear, st.session_state.data_family_allyear, st.session_state.data_top5family = getting_csv()

if len(st.session_state) != 0:
    # Create a block on the left - Selection
    # 1st column - Selection of information given by the user

    list_page_name = ['Data visualization', 'Sales prediction based on a store', 'Sales prediction based on a category']
    selected_page = st.sidebar.selectbox("Select a page", list_page_name)

    with st.sidebar:
        with st.expander("Fill up the form for your prediction"):

            #Date selection
            date_selection_prevision = st.date_input("Select a date: ",datetime.datetime.strptime('2017-01', '%Y-%m'))

            #City selection
            city_selection = st.selectbox('Select a city: ',np.sort(st.session_state.data_train_merge_stores['city'].unique()))

            #Retrieve store list from city and Store selection
            array_city_selection_store = np.array(st.session_state.data_train_merge_stores.loc[st.session_state.data_train_merge_stores['city'] == city_selection, 'store_nbr'])
            array_city_selection_store = pd.DataFrame(array_city_selection_store)
            array_city_selection_store = array_city_selection_store.drop_duplicates(subset=0).sort_values(by = 0, ascending=True)
            store_selection = st.selectbox('Select a store from your city: ',array_city_selection_store)

            #Family selection for prediction
            df_all_family = pd.DataFrame(st.session_state.data_train_merge_stores['family'].unique())
            family_prediction_selection = st.selectbox('Select a category: ',df_all_family)

    def main_page ():
        col_,col1, col2,col__ = st.columns((1,1,3,2))
        with col1:
                st.write("")
                new_image = Image_Data_visualization.resize((250, 250))
                st.markdown('')
                st.image(new_image,use_column_width=True)
        with col2:
            st.write("-------")
            st.markdown("<h1 style='text-align: center; color: red;font-family: arial; font-size: 200%;\
                        vertical-align: middle ; line-height: 21px\
                    '><strong>Data Visualization</strong></h1>", unsafe_allow_html=True)
            st.write("-------")

        #Plot the map
        st.markdown("<h1 style='text-align: left; color: black;font-family: arial; font-size: 150%;\
            vertical-align: middle;\
            '><strong>I) Where do you sell the most ? 🌍</strong></h1>", unsafe_allow_html=True)
        px.set_mapbox_access_token('pk.eyJ1IjoibXJkYXJhdWpvIiwiYSI6ImNsM3hsY2c2NzAzcHEzYm1oYmliZHc5aXoifQ.1E3p2I8p8bEkHSPJDzUXWQ')
        df = px.data.election_geojson()
        fig = px.scatter_mapbox(data_frame=st.session_state.sales_city_year, lat="Lat", lon="Lon", color="city", size="sales",
                 hover_name="city",
                 color_continuous_scale=px.colors.cyclical.IceFire, size_max=50, zoom=5.5, mapbox_style="open-street-map")
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)

        #Plot the top 5
        st.markdown("<h1 style='text-align: left; color: black;font-family: arial; font-size: 150%;\
            vertical-align: middle;\
            '><strong>II) Your Top 5 ? 🖐 </strong></h1>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns((1,1,1))
        with col1:
            st.markdown("<h1 style='text-align: center; color: blue;font-family: arial; font-size: 130%;\
            vertical-align: middle;\
            '><strong>Top 5 cities 🌆</strong></h1>", unsafe_allow_html=True)
            fig_top5_Cities = px.bar(st.session_state.map_base_top_five, x='city', y='sales', color='city')
            st.plotly_chart(fig_top5_Cities, use_container_width=True)
        with col2:
            st. markdown("<h1 style='text-align: center; color: blue;font-family: arial; \
                         font-size: 130%;'><strong>Top 5 stores 🏪 </strong></h1>", unsafe_allow_html=True)
            fig_top5_Stores = px.bar(st.session_state.df_stores_top_five, x='store_nbr', y='sales', color = 'city')
            st.plotly_chart(fig_top5_Stores, use_container_width=True)
        with col3:
            st.text("")
            st. markdown("<h1 style='text-align: center; color: blue;font-family: arial; \
                        font-size: 130%;'><strong>Top 5 categories 🛍</strong></h1>", unsafe_allow_html=True)
            fig_top5_Family = px.bar(data_frame=st.session_state.data_top5family, x='category', y='sales', color='category')
            st.plotly_chart(fig_top5_Family, use_container_width=True)

        #Plot the stores
        st.markdown("<h1 style='text-align: left; color: black;font-family: arial; font-size: 150%;\
            vertical-align: middle;\
            '><strong>III) How many stores do you have ? 🧐   </strong></h1>", unsafe_allow_html=True)

        col1, col2 = st.columns((1,1))
        with col1:
            fig = px.pie(st.session_state.df_stores_city, values = 'store_nbr', names = 'city')
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            with st.expander("Best store on each city ? 🥇") :
                st.markdown(
                    """
                <style>
                .streamlit-expanderHeader {
                    font-size: x-large;
                    text-align: center;
                    font-family: arial;
                    color: blue
                }
                </style>
                """,
                    unsafe_allow_html=True,
                )
                fig = px.bar(st.session_state.df_MaxSales_BiggestStore, x='sales', y='store_nbr', color = 'city', orientation='h')
                fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
                st.plotly_chart(fig, use_container_width=True)

        #Plot the family=category
        st.markdown("<h1 style='text-align: left; color: black;font-family: arial; font-size: 150%;\
            vertical-align: middle;\
            '><strong>IV) Which categories do you sell ? 🛒  </strong></h1>", unsafe_allow_html=True)
        data_family_allyear_plot = st.session_state.data_family_allyear
        data_family_allyear_plot_From1to10 = data_family_allyear_plot.iloc[0:10,:]
        #data_family_allyear_plot_From12to21 = data_family_allyear_plot.iloc[11:22,:]
        data_family_allyear_plot_From23to33 = data_family_allyear_plot.iloc[23:33,:]

        col1, col2 = st.columns((1,1))
        with col1:
            st.markdown("<h1 style='text-align: center; color: blue;font-family: arial; font-size: 130%;\
            vertical-align: middle;\
            '><strong>Top 10 categories 👍</strong></h1>", unsafe_allow_html=True)
            fig1 = px.bar(data_family_allyear_plot_From1to10, x='category', y='sales', color='category')
            fig1.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            st.markdown("<h1 style='text-align: center; color: blue;font-family: arial; font-size: 130%;\
            vertical-align: middle;\
            '><strong>Bottom 10 categories 👎</strong></h1>", unsafe_allow_html=True)
            st.text(" ")
            fig3 = px.funnel(data_family_allyear_plot_From23to33, x='category', y='sales', color='category')
            fig3.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig3, use_container_width=True)

    def page2():
        st.markdown("Prediction based on the city")
        st.write("Let's predict the number of sales for the city ", city_selection, " for 12 months of the year ", date_selection_prevision.year)


       # st.caption("d) Sales for all stores on selected city")
       # data_selection_bycity = st.session_state.data_train_merge_stores[st.session_state.data_train_merge_stores['city'] == city_selection]
       # data_selection_bycity = pd.DataFrame(data_selection_bycity.groupby(['store_nbr'])['sales'].sum()).reset_index()
        #data_selection_bycity['store_nbr'] = data_selection_bycity['store_nbr'].astype('string')
        #data_selection_bycity = data_selection_bycity.sort_values('sales',ascending= False)
        #if len(array_city_selection_store) > 1 :
         #   fig = px.bar(data_selection_bycity, x ='store_nbr',y='sales', color = 'store_nbr')
          #  st.write(fig)
        #elif len(array_city_selection_store) == 1 :
         #   st.write('There is only one store in the city that has sold ',int(data_selection_bycity['sales'].sum()),
          #              ' products from 2013 to 2016')

        #st.caption("e) Sales on selected store")
        #data_selection_bystore = st.session_state.data_train_merge_stores[st.session_state.data_train_merge_stores['store_nbr'] == store_selection]
        #data_selection_bystore = data_selection_bystore.groupby(['year'])['sales'].sum().reset_index()
        #data_selection_bystore['year'] = data_selection_bystore['year'].astype('string')
        #fig = px.line(data_selection_bystore, x ='year',y='sales')
        #st.write(fig)

        #st.caption("f) Sales on selected store for 12 months")
        #st.write("Analysis for year ", date_selection_analysis.year)
        #data_selection_bymonth_bystore = st.session_state.data_train_merge_stores[st.session_state.data_train_merge_stores['store_nbr'] == store_selection]
        #data_selection_bymonth_bystore = data_selection_bymonth_bystore[data_selection_bymonth_bystore['year'] == date_selection_analysis.year]
        #data_selection_bymonth_bystore = data_selection_bymonth_bystore.groupby(['month'])['sales'].sum().reset_index()
        #data_selection_bymonth_bystore['month'] = data_selection_bymonth_bystore['month'].astype('string')
        #fig = px.line(data_selection_bymonth_bystore, x ='month',y='sales')
        #st.write(fig)

        #st.caption("d) Sales for selected families on all stores")
        #family_selection_df = st.session_state.data_family_peryear[st.session_state.data_family_peryear['family'].isin(family_selection)]
        #family_selection_df['year'] = family_selection_df['year'].astype('string')
        #fig = plt.figure(figsize=(12,10))
        #sns.lineplot(x='year', y='sales', data=family_selection_df, hue = 'family')
        #plt.xticks(rotation=90)
        #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        #st.pyplot(fig)

        #st.caption("e) Sales for selected families on selected store")
        #store_family_selection_df = st.session_state.data_train_merge_stores[st.session_state.data_train_merge_stores['store_nbr']==store_selection]
        #store_family_selection_df = store_family_selection_df[store_family_selection_df['family'].isin(family_selection)]
        #fig = px.bar(store_family_selection_df, x='family', y='sales', color='family')
        #st.write(fig)


        def API_City(year, city):
            year_list=[year]*12
            city_list=[city]*12
            params = {
                'year': year_list,
                'month': Const_month_predict,
                'city': city_list
            }
            response = requests.get(Const_url_predict_city, params=params)
            return response.json()

        city_return = pd.DataFrame(API_City(date_selection_prevision.year, city_selection)).reset_index()
        city_return.rename(columns = {'index':'month', 'sales_city_year':'sales'}, inplace = True)
        city_return['month'] = city_return['month'].astype('int')
        city_return = city_return.sort_values(by = 'month', ascending=True)
        st.write("Dataframe Prediction 12 months in year ", date_selection_prevision.year, "for city " , city_selection)
        st.dataframe(city_return)
        fig = px.line(city_return, x ='month',y='sales')
        st.write("Plot Prediction for 12 months in year ", date_selection_prevision.year, "for city " , city_selection)
        st.write(fig)


    if st.session_state.count == 0:
        print('Rodando o Facebook Prophet')
        @st.cache()
        def store(sales_and_stores):

            sales_and_stores = sales_and_stores.rename(columns={'date': 'ds', 'sales':'y'}).drop(columns=["month", "year", "city"])
            forecasts = {}
            for store in sales_and_stores['store_nbr'].unique():

                print(store)

                # creating a new variable
                tmp_df_prep = sales_and_stores[sales_and_stores['store_nbr']== store]

                # Transforming the column in date time
                tmp_df_prep['ds'] = pd.to_datetime(tmp_df_prep['ds'])

                # creating a temporary df to the the training and prediction
                tmp_df = tmp_df_prep.groupby(by='ds').sum().reset_index()


                # defining the train/test
                train = tmp_df.iloc[:1457]
                test = tmp_df.iloc[1458:]


                # Instantiating the FB Prophet model
                model = Prophet(seasonality_mode='multiplicative')

                # fitting the model on the train test
                model.fit(train)

                forecasts[store]= model

            return forecasts

        st.session_state.model_store = store(st.session_state.data_train_merge_stores)



    def page3():
        st.markdown("Prediction based on the store")
        st.write("Let's predict the number of sales in the city", city_selection,
                 " for the store ", store_selection, " for 12 months of the year ", date_selection_prevision.year)

        def API_Store(year, city, store_nbr):
            year_list=[year]*12
            city_list=[city]*12
            store_list=[store_nbr]*12

            params = {
                'year': year_list,
                'month': Const_month_predict,
                'city': city_list,
                'store_nbr': store_list
            }
            response = requests.get(Const_url_predict_store, params=params)
            return response.json()

        store_return = pd.DataFrame(API_Store(date_selection_prevision.year, city_selection, store_selection)).reset_index()
        store_return.rename(columns = {'index':'month', 'sales_store_year':'sales'}, inplace = True)
        store_return['month'] = store_return['month'].astype('int')
        store_return = store_return.sort_values(by = 'month', ascending=True)
        st.write("Dataframe Prediction 12 months in year ", date_selection_prevision.year, "for city " , city_selection, " and store ",
                 store_selection)
        st.dataframe(store_return)
        fig = px.line(store_return, x ='month',y='sales')
        st.write("Plot Prediction for 12 months in year ", date_selection_prevision.year, "for city " , city_selection, " and store ",
                 store_selection)
        st.write(fig)

        future_store = st.session_state.model_store[store_selection].make_future_dataframe(periods=12, freq='MS')  #period of 12 months
        forecast_future_store = st.session_state.model_store[store_selection].predict(future_store)
        forecast_future_store.head()

        fig = st.session_state.model_store[store_selection].plot(forecast_future_store)
        st.write(fig)

        fig = st.session_state.model_store[store_selection].plot_components(fcst=forecast_future_store)
        st.write(fig)

    if st.session_state.count == 0:
        print('Rodando o Facebook Prophet')
        @st.cache()
        def family(sales_and_stores):

            sales_and_stores = sales_and_stores.rename(columns={'date': 'ds', 'sales':'y'}).drop(columns=["month", "year", "city"])

            forecasts = {}
            for category in sales_and_stores['family'].unique():

                print(category)
                # creating a new variable
                tmp_df_prep_2 = sales_and_stores[sales_and_stores['family']== category]

                # Transforming the column in date time
                tmp_df_prep_2['ds'] = pd.to_datetime(tmp_df_prep_2['ds'])

                # creating a temporary df to the the training and prediction
                tmp_df = tmp_df_prep_2.groupby(by='ds').sum().drop(columns=["store_nbr"]).reset_index()


                # defining the train/test
                train = tmp_df.iloc[:1457]
                test = tmp_df.iloc[1458:]


                # Instantiating the FB Prophet model
                model = Prophet(seasonality_mode='multiplicative')

                # fitting the model on the train test
                model.fit(train)

                forecasts[category]= model

            return forecasts

        st.session_state.model_family = family(st.session_state.data_train_merge_stores)

    #def page4():
      #  st.markdown("Prediction based on store and family")
       # st.write("Let's predict the number of sales in the city ", city_selection, " for the store ", store_selection,
        #         " for 12 months of the year ", date_selection_prevision.year,
         #           "for the family ", family_prediction_selection)

        #def API_Family(year, city, store_nbr, family):
         #   year_list=[year]*12
          #  city_list=[city]*12
           # store_list=[store_nbr]*12
            #family_list = [family]*12

           # params = {
            #    'year': year_list,
             #   'month': Const_month_predict,
              #  'city': city_list,
               # 'store_nbr': store_list,
                #'family': family_list
           # }
            #response = requests.get(Const_url_predict_family, params=params)
            #return response.json()

        #family_return = pd.DataFrame(API_Family(date_selection_prevision.year, city_selection, store_selection, family_prediction_selection)).reset_index()
        #family_return.rename(columns = {'index':'month', 'sales_family_year':'sales'}, inplace = True)
        #family_return['month'] = family_return['month'].astype('int')
        #family_return = family_return.sort_values(by = 'month', ascending=True)
        #st.write("Dataframe Prediction 12 months in year ", date_selection_prevision.year, "for city " , city_selection, " and store ",
        #         store_selection, " and family ", family_prediction_selection)
        #st.dataframe(family_return)
        #fig = px.line(family_return, x ='month',y='sales')
        #st.write("Plot Prediction for 12 months in year ", date_selection_prevision.year, "for city " , city_selection, " and store ",
         #        store_selection, " and family ", family_prediction_selection)
        #st.write(fig)


    #Code to have different page on streamlit
    page_names_to_funcs = {
        "Data visualization" : main_page,
        "Sales prediction based on a store": page2,
        "Sales prediction based on a category": page3,
    }

    page_names_to_funcs[selected_page]()

#Session States
st.session_state.count += 1
st.write('Count = ', st.session_state.count)
