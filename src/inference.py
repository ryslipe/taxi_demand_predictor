from datetime import datetime, timedelta

import hopsworks
import hopsworks.project
from hsfs.feature_store import FeatureStore
import pandas as pd 
import numpy as np

import src.config as config

# connect to our project
def get_hopsworks_project() -> hopsworks.project.Project:
    return hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY
    )

# get our feature store
def get_feature_store() -> FeatureStore:
    project = get_hopsworks_project()
    return project.feature_store()

# get our model
def get_model_predictions(model, features: pd.DataFrame) -> pd.DataFrame:
    '''Generate Predictions based on recent batch of features.'''
    predictions = model.predict(features)

    results = pd.DataFrame
    results['pickup_location_id'] = features['pickup_location_id'].values
    results['predicted_demand'] = predictions.round(0)

    return results

# load a batch of features from our feature store
def load_batch_of_features_from_store(
        current_date: datetime
) -> pd.DataFrame:
    
    # connect to our feature store
    feature_store = get_feature_store()

    n_features = config.N_FEATURES

    # get batch of data depending on our current date
    # fetch data until an hour ago starting from 28 days ago
    fetch_data_to = current_date - timedelta(hours = 1)
    fetch_data_from = current_date - timedelta(days = 28)
    print(f'Fetching rides from {fetch_data_from} to {fetch_data_to}')

    # connect to our feature view
    feature_view = feature_store.get_feature_view(
        name = config.FEATURE_VIEW_NAME,
        version=config.FEATURE_VIEW_VERSION
    )

    # set up time series data - include a day before and a day after to ensure we have no missing data
    ts_data = feature_view.get_batch_data(
        start_time= (fetch_data_from - timedelta(days=1)),
        end_time= (fetch_data_to  + timedelta(days=1))
    )

    # filter data to the time period we are interested in
    pickup_ts_from = int(fetch_data_from.timestamp() * 1000)
    pickup_ts_to = int(fetch_data_to.timestamp() * 1000)
    ts_data = ts_data[ts_data['pickup_ts'].between(pickup_ts_from, pickup_ts_to)]

    # sort data by location and time
    ts_data.sort_values(by=['pickup_location_id', 'pickup_hour'], inplace=True)

    # validate we are not missing data in the feature store
    location_ids = ts_data['pickup_location_id'].unique()

    # we should have 24*28 (28 days with 24 hours) for all locations
    assert len(ts_data) == config.N_FEATURES * len(location_ids), \
        "Time-series data is not complete. Make sure your feature pipeline is up and runnning."
    
    # transpose the data
    x = np.ndarray(shape = (len(location_ids), n_features), dtype = np.float32)
    for i, location_id in enumerate(location_ids):
        ts_data_i = ts_data.loc[ts_data['pickup_location_id'] == location_id, :]
        ts_data_i = ts_data_i.sort_values(by=['pickup_hour'])
        x[i, :] = ts_data_i['rides'].values

    # numpy arrays to Pandas dataframes
    features = pd.DataFrame(
        x,
        columns=[f'rides_previous_{i+1}_hour' for i in reversed(range(n_features))]
    )
    features['pickup_hour'] = current_date
    features['pickup_location_id'] = location_ids
    features.sort_values(by=['pickup_location_id'], inplace=True)

    return features

# load the model
def load_model_from_registry():

    import joblib
    from pathlib import Path

    project = get_hopsworks_project()
    model_registry = project.get_model_registry()

    model = model_registry.get_model(
        name = config.MODEL_NAME,
        version=config.MODEL_VERSION
    )

    # download to local directory
    model_dir = model.download()
    # load the model from that local directory
    model = joblib.load(Path(model_dir) / 'model.pkl')

    return model






    




