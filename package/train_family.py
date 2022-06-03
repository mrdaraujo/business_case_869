import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
import joblib

# set for fetching and merging data
def DataMerging(csv_left, csv_stores):
    ''' function for merging stores dataset to train/test '''
    csv_stores = csv_stores[['store_nbr','city']]
    df = pd.merge(csv_left, csv_stores, how='left', on='store_nbr')
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df = df.drop(columns=['date', 'id', 'onpromotion'])
    df = df.groupby(["year","month","city",'store_nbr','family']).sum().reset_index()

    return df

def DataTrain():
    ''' function for for getting X and y train sets'''
    path = 'gs://business-case/'
    csv_left = pd.read_csv(path+'train.csv')
    csv_stores = pd.read_csv(path+"stores.csv")
    df = DataMerging(csv_left, csv_stores)
    df.loc[df['sales']==0,'sales'] = 0.1
    df['sales'] = np.log(df['sales'])
    X_train = df.drop(columns='sales')
    y_train = df['sales']

    return X_train, y_train

def DataPredict():
    ''' function for prediction set'''
    path = 'gs://business-case/'
    csv_left = pd.read_csv(path+'test.csv')
    csv_stores = pd.read_csv(path+"stores.csv")
    X_predict = DataMerging(csv_left, csv_stores)

    return X_predict



# full pipeline set
def Preproc():
    ''' function for preprocessing'''
    # num_transformer = Pipeline([
        # ('scaler', StandardScaler())
    # ])

    cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)

    preprocessor = ColumnTransformer([
            # ('num_tr', num_transformer, ['onpromotion']),
            ('cat_tr', cat_transformer, ['year', 'month', 'city', 'store_nbr', 'family'])
        ],remainder='passthrough'
    )

    return preprocessor

def Pipe(X_train, y_train):
    ''' function for training model'''

    preprocessor = Preproc()
    pipeline = make_pipeline(preprocessor, LinearRegression())
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, 'model_family.joblib')

    return pipeline, preprocessor

def TrainFamily():
    X_train, y_train = DataTrain()
    X_predict = DataPredict()
    pipe, preprocessor = Pipe(X_train, y_train)

    return X_train, y_train, X_predict, pipe, preprocessor

if __name__ == '__main__':
    X_train, y_train = DataTrain()
    X_predict = DataPredict()
    pipe, preprocessor = Pipe(X_train, y_train)
