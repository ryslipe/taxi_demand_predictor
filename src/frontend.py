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

# load the model from the model registry
with st.spinner(text='Loading model registry...'):
    model = load_model_from_registry()
    st.sidebar.write('✅ Loaded model from model registry.')
    progress_bar.progress(3/N_STEPS)

# make model predictions
with st.spinner(text='Generating Predictions...'):
    results = get_model_predictions(model, features)
    st.sidebar.write('✅ Predictions have been generated.')
    progress_bar.progress(4/N_STEPS)

# prepare color coding for plot
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

# generate the map
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
    # display information when hovering over a zone
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
    n_to_plot = 10

    # plot each time-series with the prediction
    for row_id in row_indices[:n_to_plot]:

        # title
        location_id = predictions_df['pickup_location_id'].iloc[row_id]
        location_name = predictions_df['zone'].iloc[row_id]
        st.header(f'Location ID: {location_id} - {location_name}')

        # plot predictions
        prediction = predictions_df['predicted_demand'].iloc[row_id]
        st.metric(label="Predicted demand", value=int(prediction))
        
        # plot figure
        # generate figure
        fig = plot_one_sample(
            example_id=row_id,
            features=features,
            targets=predictions_df['predicted_demand'],
            predictions=pd.Series(predictions_df['predicted_demand']),
            display_title=False,
        )
        st.plotly_chart(fig, theme="streamlit", use_container_width=True, width=1000)
        
    progress_bar.progress(7/N_STEPS)



