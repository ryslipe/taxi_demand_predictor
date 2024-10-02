import zipfile
from datetime import datetime, timezone
import requests
import numpy as np
import pandas as pd

# plotting libraries
import streamlit as st
import geopandas as gpd
import pydeck as pdk

from src.inference import (
    load_batch_of_features_from_store,
    get_model_predictions,
    load_model_from_registry
)

# store data
from src.paths import DATA_DIR
from src.plot import plot_one_sample

# set wide layout for streamlit
st.set_page_config(layout='wide')

# detect the current time and print it as the title
current_date = pd.to_datetime(datetime.now(timezone.utc)).floor('H')
current_date = current_date.replace(tzinfo=None)
st.title('Taxi Demand Predictions 🚖')
st.header(f'{current_date}')

# Add a header with a gear emoji
progress_bar = st.sidebar.header('⚙️ Working Progress')
progress_bar = st.sidebar.progress(0)
N_STEPS = 7

# load external file with taxi zone coordinates
def load_shape_data_file()-> gpd.geodataframe.GeoDataFrame:
    """
    Fetches remote file with shape data, that we later use to plot the
    different pickup_location_ids on the map of NYC.

    Raises:
        Exception: when we cannot connect to the external server where
        the file is.

    Returns:
        GeoDataFrame: columns -> (OBJECTID	Shape_Leng	Shape_Area	zone	LocationID	borough	geometry)
    """
    # download the file
    URL = 'https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip'
    # get URL
    response = requests.get(URL)
    # directory to save to 
    path = DATA_DIR / f'taxi_zones.zip'

    # try to download
    # if the status code is successful read the content
    if response.status_code == 200:
        open(path, 'wb').write(response.content)
    else: 
        raise Exception(f'{URL} is not available.')
    
    # unzip the taxi zones file
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR / 'taxi_zones')

    # load and return the shape file
    # 'epsg:4326' specifies the EPSG code for the WGS 84 coordinate system, 
    # which is a common geographic coordinate system used for latitude and longitude.
    return gpd.read_file(DATA_DIR / 'taxi_zones/taxi_zones.shp').to_crs('epsg:4326')

with st.spinner(text='Downloading the shape file to plot taxi zones...'):
    # call the function
    geo_df = load_shape_data_file()
    st.sidebar.write('✅ Shape file was downloaded.')
    progress_bar.progress(1/N_STEPS)

# access the feature store for most recent batch of data from github ations
with st.spinner(text='Downloading recent batch of data...'):
    features = load_batch_of_features_from_store(current_date)
    st.sidebar.write('✅ Batch of data downloaded.')
    progress_bar.progress(2/N_STEPS)
    print(f'{features}')





