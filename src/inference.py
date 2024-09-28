# streamlit app will need these functions
from datetime import datetime, timedelta
import src.config as config

import hopsworks
import hopsworks.project
from hsfs.feature_store import FeatureStore
import pandas as pd
import numpy as np

# function to connect to our project 
def get_hopsworks_project() -> hopsworks.project.Project:

    return hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY
    )

# get the feature store of our project
def get_feature_store() -> FeatureStore:
    # sign in to project
    project = get_hopsworks_project()
    # return the feature store
    return project.get_feature_store()

# use a model and features for predictions
def get_model_predictions(model, features) -> pd.DataFrame:
    # make predictions on the features
    predictions = model.predict(features)

    # dataframe creation
    results = pd.DataFrame()
    # convert to numpy array using .values()
    results['pickup_location_id'] = features['pickup_location_id'].values()
    results['predicted_demand'] = predictions.round()

    return results


def load_batch_of_features_from_store(
    current_date: pd.Timestamp,    
) -> pd.DataFrame:
    """Fetches the batch of features used by the ML system at `current_date`

    Args:
        current_date (datetime): datetime of the prediction for which we want
        to get the batch of features

    Returns:
        pd.DataFrame: 4 columns:
            - `pickup_hour`
            - `rides`
            - `pickup_location_id`
            - `pickpu_ts`
    """
    feature_store = get_feature_store()
    n_features = config.N_FEATURES

    # fetch data from the feature store go from 28 days ago until last hour
    fetch_data_from = current_date - timedelta(days=28)
    fetch_data_to = current_date - timedelta(hours=1)
    print(f'Fetching data from {fetch_data_from} to {fetch_data_to}.')
    
    # access feature view
    feature_view = feature_store.get_feature_view(
        name=config.FEATURE_VIEW_NAME,
        version=config.FEATURE_VIEW_VERSION
    )
    # get a batch of data from the feature_view
    ts_data = feature_view.get_batch_data(
        start_time=fetch_data_from - timedelta(days=1),
        end_time=fetch_data_to + timedelta(days=1)
    )
    # 

    # filter data to the time period we are interested in
    pickup_ts_from = int(fetch_data_from.timestamp() * 1000)
    pickup_ts_to = int(fetch_data_to.timestamp() * 1000)
    ts_data = ts_data[ts_data.pickup_ts.between(pickup_ts_from, pickup_ts_to)]

    # sort data by location and time
    ts_data.sort_values(by=['pickup_location_id', 'pickup_hour'], inplace=True)

    # validate we are not missing data in the feature store
    location_ids = ts_data['pickup_location_id'].unique()
    assert len(ts_data) == config.N_FEATURES * len(location_ids), \
        "Time-series data is not complete. Make sure your feature pipeline is up and runnning."

    # transpose time-series data as a feature vector, for each `pickup_location_id`
    x = np.ndarray(shape=(len(location_ids), n_features), dtype=np.float32)
    for i, location_id in enumerate(location_ids):
        ts_data_i = ts_data.loc[ts_data.pickup_location_id == location_id, :]
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
    
