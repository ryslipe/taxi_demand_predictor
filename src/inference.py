# streamlit app will need these functions
from datetime import datetime, timdelta
import src.config as config

import hopsworks
import hopsworks.project
from hsfs.feature_store import FeatureStore
import pandas as pd
import numpy

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


    
