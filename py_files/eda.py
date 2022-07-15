import plotly.express as px
import numpy as np
# import plotly.io as pio
# pio.renderers.default = "notebook"


def plot_tanzania(subdf):
    # The geographic extents of Tanzania based on the location of the wells

    lat_extent = [-11.64944, -0.998464]
    lon_extent = [29.607122, 40.345193]

    center = (np.mean(lat_extent), np.mean(lon_extent))

    zoom_level = 5.5

    fig = px.scatter_mapbox(subdf, lat="latitude", lon="longitude",
                            zoom=zoom_level, height=800,
                            color="status_group",
                            color_discrete_map={'functional': 'green',
                                                'non functional': 'red',
                                                'functional needs repair': 'yellow'},
                            hover_data=['lga', 'ward', 'region'],
                            opacity=1.0,
                            width=1000,
                            )

    fig.update_layout(mapbox_style="stamen-terrain",
                      mapbox_zoom=zoom_level,
                      mapbox_center_lat=center[0],
                      mapbox_center_lon=center[1],
                      margin={"r": 0, "t": 0, "l": 0, "b": 0})

    fig.show()


def plot_bar(df, col):
    for type_ in df[col].unique():
        data = df[df[col] == type_].status_group.value_counts().reset_index()
        fig = px.bar(data,
                     x='index',
                     y='status_group',
                     title=type_,
                     width=800,
                     color='index',
                     color_discrete_map={'functional': 'green',
                                         'non functional': 'red',
                                         'functional needs repair': 'yellow'}
                     )
        fig.show()

