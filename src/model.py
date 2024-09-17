import pandas as pd
import lightgbm as lgb

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline, Pipeline

# FunctionTransformer
def average_rides_last_4_weeks(X: pd.DataFrame) -> pd.DataFrame:
    '''Add column for average rides last 4 weeks for given hour/day'''
    X['average_rides_last_4_weeks'] = 0.25*(
        X[f'rides_previous_{7*24}_hour'] + \
        X[f'rides_previous_{2*7*24}_hour'] + \
        X[f'rides_previous_{3*7*24}_hour'] + \
        X[f'rides_previous_{4*7*24}_hour']
    )
    return X

# temporal features
class TemporalFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        return
    def transform(self, X, y = None):
        # since we are dropping the pickup hour datetime
        X_ = X.copy()

        X_['hour'] = X_['pickup_hour'].dt.hour
        X_['day'] = X_['pickup_hour'].dt.dayofweek

        # lightGBM does not handle datetime objects
        return X_.drop(columns = ['pickup_hour'])
    
def get_pipeline(**hyperparams) -> Pipeline:
    '''Function to access the pipeline'''
    # function transformer initializer
    add_average_rides_last_4_weeks = FunctionTransformer(add_average_rides_last_4_weeks,
                                                         validate=False)
    
    # adding temporal features to pipeline
    add_temporal_features = TemporalFeatures()

    # create the pipeline
    pipeline = make_pipeline(
        add_average_rides_last_4_weeks,
        add_temporal_features,
        lgb.LGBMRegressor(**hyperparams)
    )

    