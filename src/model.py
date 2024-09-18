import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline
import lightgbm as lgb

def average_rides_last_4_weeks(X: pd.DataFrame) -> pd.DataFrame:
    X['average_rides_last_4_weeks'] = 0.25 * (
        X[f'rides_previous_{7*24}_hour'] +
        X[f'rides_previous_{2*7*24}_hour'] +
        X[f'rides_previous_{3*7*24}_hour'] +
        X[f'rides_previous_{4*7*24}_hour']
    )
    return X

class TemporalFeaturesEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_ = X.copy()
        X_["hour"] = X_['pickup_hour'].dt.hour
        X_["day_of_week"] = X_['pickup_hour'].dt.dayofweek
        return X_.drop(columns=['pickup_hour'])

def get_pipeline(**hyperparams) -> Pipeline:
    print("Creating pipeline with hyperparameters:", hyperparams)
    
    add_feature_average_rides_last_4_weeks = FunctionTransformer(
        average_rides_last_4_weeks, validate=False)
    print("Added feature transformer for average rides")

    add_temporal_features = TemporalFeaturesEngineer()
    print("Added temporal features engineer")

    pipeline = make_pipeline(
        add_feature_average_rides_last_4_weeks,
        add_temporal_features,
        lgb.LGBMRegressor(**hyperparams)
    )
    print("Pipeline created:")
    return pipeline
