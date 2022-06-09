from package.etl import *
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
import joblib



# full pipeline set
def Preproc():
    ''' function for preprocessing'''
    # num_transformer = Pipeline([
        # ('scaler', StandardScaler())
    # ])

    cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)

    preprocessor = ColumnTransformer([
            # ('num_tr', num_transformer, ['onpromotion']),
            ('cat_tr', cat_transformer, ['year', 'month', 'city', 'store_nbr'])
        ],remainder='passthrough'
    )

    return preprocessor

def Pipe(X_train, y_train):
    ''' function for training model'''

    preprocessor = Preproc()
    pipeline = make_pipeline(preprocessor, LinearRegression())
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, 'model_store.joblib')

    return pipeline, preprocessor

def TrainStore():
    drop = ['onpromotion', 'family']
    features = ["year","month",'city', 'store_nbr']
    X_train, y_train = DataTrain(drop, features)
    X_predict = DataPredict(drop, features)
    pipe, preprocessor = Pipe(X_train, y_train)

    return X_train, y_train, X_predict, pipe, preprocessor

if __name__ == '__main__':
    drop = ['onpromotion', 'family']
    features = ["year","month",'city', 'store_nbr']
    X_train, y_train = DataTrain(drop, features)
    X_predict = DataPredict(drop, features)
    pipe, preprocessor = Pipe(X_train, y_train)
