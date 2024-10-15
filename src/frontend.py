import zipfile
from datetime import datetime
import pytz

import requests
import numpy as np
import pandas as pd

# plotting libraries
import streamlit as st
import geopandas as gpd
import pydeck as pdk

# modeling libraries
from src.inference import(
    load_batch_of_features_from_store,
    load_model_from_registry,
    get_model_predictions
)

from src.paths import DATA_DIR
from src.plot import plot_one_sample

st.set_page_config(layout='wide')

# set title to be the current time 
# Get the current date and time in UTC
current_date = pd.to_datetime(datetime.now(pytz.utc)).floor('H')
st.title('NYC Taxi Ride Predictions 🚕')
st.header(current_date)

# plotting the progress bar
progress_bar = st.sidebar.header('⚙️ Working Progress')
# start at no progress
progress_bar = st.sidebar.progress(0)
# useful when steps are completed
N_STEPS = 7



def load_shape_data_file() -> gpd.geodataframe.GeoDataFrame:
    """
    Fetches remote file with shape data, that we later use to plot the
    different pickup_location_ids on the map of NYC.

    Raises:
        Exception: when we cannot connect to the external server where
        the file is.

    Returns:
        GeoDataFrame: columns -> (OBJECTID	Shape_Leng	Shape_Area	zone	LocationID	borough	geometry)
    """
    # download zip file
    URL = 'https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip'
    response = requests.get(URL)
    path = DATA_DIR / f'taxi_zones.zip'

    # if sucessful write the contents
    if response.status_code == 200:
        open(path, "wb").write(response.content)
    else:
        raise Exception(f'{URL} is not available')

    # unzip file and write to the data directory under the name taxi_zones
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR / 'taxi_zones')

    # load and return shape file
    return gpd.read_file(DATA_DIR / 'taxi_zones/taxi_zones.shp').to_crs('epsg:4326')

# call the funciton to store the shape file into a geo_df
with st.spinner(text="Downloading shape file to plot taxi zones"):
    geo_df = load_shape_data_file()
    st.sidebar.write('✅ Shape file was downloaded ')
    progress_bar.progress(1/N_STEPS)

st.write('This is the shape file that will be used for \
         graphing the locations and getting the location name.')
st.write(geo_df)


# connect to feature store for most recent batch of features
with st.spinner(text='Fetching recent batch of data.'):
    # loads a month of data up til last hour
    features = load_batch_of_features_from_store(current_date)
    
    st.sidebar.write('✅ Features received.')
    progress_bar.progress(2/N_STEPS)

st.write('These are the current features')
st.write(features)


# load the model from the model registry
with st.spinner(text = 'Loading the model from model registry'):
    model = load_model_from_registry()
    st.sidebar.write('✅ The model has been loaded.')
    progress_bar.progress(3/N_STEPS)

st.write('These are some details about how our model works.')
st.write(model)


# get predictions
with st.spinner(text = 'Generating model predictions'):
    results = get_model_predictions(model, features)
    st.sidebar.write('✅ Predictions have been generated.')
    progress_bar.progress(4/N_STEPS)

st.write('Here are the predictions.')
st.write(results)

with st.spinner(text="Preparing data to plot"):

    def pseudocolor(val, minval, maxval, startcolor, stopcolor):
        """
        Convert value in the range minval...maxval to a color in the range
        startcolor to stopcolor. The colors passed and the the one returned are
        composed of a sequence of N component values.

        Credits to https://stackoverflow.com/a/10907855
        """
        f = float(val-minval) / (maxval-minval)
        return tuple(f*(b-a)+a for (a, b) in zip(startcolor, stopcolor))
        
    df = pd.merge(geo_df, results,
                  right_on='pickup_location_id',
                  left_on='LocationID',
                  how='inner')
    
    BLACK, GREEN = (0, 0, 0), (0, 255, 0)
    df['color_scaling'] = df['predicted_demand']
    max_pred, min_pred = df['color_scaling'].max(), df['color_scaling'].min()
    df['fill_color'] = df['color_scaling'].apply(lambda x: pseudocolor(x, min_pred, max_pred, BLACK, GREEN))
    progress_bar.progress(5/N_STEPS)

with st.spinner(text="Generating NYC Map"):

    INITIAL_VIEW_STATE = pdk.ViewState(
        latitude=40.7831,
        longitude=-73.9712,
        zoom=11,
        max_zoom=16,
        pitch=45,
        bearing=0
    )

    geojson = pdk.Layer(
        "GeoJsonLayer",
        df,
        opacity=0.25,
        stroked=False,
        filled=True,
        extruded=False,
        wireframe=True,
        get_elevation=10,
        get_fill_color="fill_color",
        get_line_color=[255, 255, 255],
        auto_highlight=True,
        pickable=True,
    )

    tooltip = {"html": "<b>Zone:</b> [{LocationID}]{zone} <br /> <b>Predicted rides:</b> {predicted_demand}"}

    r = pdk.Deck(
        layers=[geojson],
        initial_view_state=INITIAL_VIEW_STATE,
        tooltip=tooltip
    )

    st.pydeck_chart(r)
    progress_bar.progress(6/N_STEPS)


    with st.spinner(text="Plotting time-series data"):
        predictions_df = df
        row_indices = np.argsort(predictions_df['predicted_demand'].values)[::-1]
        n_to_plot = 15

        for row_id in row_indices[:n_to_plot]:
            # title
            location_id = predictions_df['pickup_location_id'].iloc[row_id]
            location_name = predictions_df['zone'].iloc[row_id]
            st.header(f'Location ID: {location_id} - {location_name}')
            fig = plot_one_sample(
                features=features,
                targets=predictions_df['predicted_demand'],
                example_id= row_id,
                predictions= pd.Series(results['predicted_demand'])

            )
            st.plotly_chart(fig, theme='streamlit', use_container_width=True, width=1000)
        progress_bar.progress(7/N_STEPS)


        