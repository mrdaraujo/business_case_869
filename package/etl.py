import pandas as pd
import numpy as np

# set for fetching and merging data
def DataMerging(csv_left, csv_stores, drop, features):
    ''' function for merging stores dataset to train/test '''
    csv_stores = csv_stores[['store_nbr','city']]
    df = pd.merge(csv_left, csv_stores, how='left', on='store_nbr')
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    drop.extend(['date', 'id'])
    df = df.drop(columns=drop)
    df = df.groupby(features).sum().reset_index()

    return df

def DataTrain(drop, features):
    ''' function for for getting X and y train sets'''
    path = 'gs://business-case/'
    csv_left = pd.read_csv(path+'train.csv')
    csv_stores = pd.read_csv(path+"stores.csv")
    df = DataMerging(csv_left, csv_stores, drop, features)
    df.loc[df['sales']==0,'sales'] = 0.1
    df['sales'] = np.log(df['sales'])
    X_train = df.drop(columns='sales')
    y_train = df['sales']

    return X_train, y_train

def DataPredict(drop, features):
    ''' function for prediction set'''
    path = 'gs://business-case/'
    csv_left = pd.read_csv(path+'test.csv')
    csv_stores = pd.read_csv(path+"stores.csv")
    X_predict = DataMerging(csv_left, csv_stores, drop, features)

    return X_predict

if __name__ == '__main__':
    X_train, y_train = DataTrain(drop=[], features=[])
    X_predict = DataPredict(drop=[], features=[])
