import os
from dotenv import load_dotenv

from src.paths import PARENT_DIR

# load the variable from the .env file
load_dotenv(PARENT_DIR / '.env')

# project name
HOPSWORKS_PROJECT_NAME = 'taxi_demand_rs'

# try to get the api key and if there is no ap key created, tell us
try:
    # note the [], not () after os.environ
    HOPSWORKS_API_KEY = os.environ['HOPSWORKS_API_KEY']
except:
    print('Create .env file in PARENT_DIR with API key named HOPSWORKS_API_KEY.')

FEATURE_GROUP_NAME = 'time_series_hourly_feature_group'
FEATURE_GROUP_VERSION = 1

FEATURE_VIEW_NAME = 'time_series_hourly_feature_view'
FEATURE_VIEW_VERSION = 1