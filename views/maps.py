#!./.venv/bin/python
# -*- coding: utf-8 -*-

import math

from dash import dcc
import pandas as pd
import plotly.graph_objects as go

from utils import get_plot_config


def get_station_map(df_meta, resp=None, preds=None):

    df_meta["text"] = (
        "Name: " + df_meta["name"] + "<br>"
        "ID: " + df_meta["triplet"] + "<br>"
        "Elev.: " + df_meta["elevation"].astype(str) + "'" + "<br>"
        "Installed: " + df_meta["beginDate"].str.replace(" 00:00:00", "").str.replace(" 00:00", "")
    )
    df_resp = df_meta[df_meta["triplet"] == resp]
    df_preds = df_meta[df_meta["triplet"].isin(preds)]
    triplets_in_use = tuple(set([i for i in (resp,) + preds if i]))
    df_others = df_meta[~df_meta["triplet"].isin(triplets_in_use)]
    
    # Always define selected_site_meta
    selected_site_meta = df_meta[df_meta["triplet"] == resp] if resp else pd.DataFrame()

    colorscale = "Hot"
    fig = go.Figure(
        data=[
            go.Scattermapbox(
                name="Nearby Sites",
                lon=df_others["longitude"],
                lat=df_others["latitude"],
                hovertext=df_others["text"],
                mode="markers",
                marker=dict(
                    color=df_others["elevation"],
                    size=15,
                    colorscale=colorscale,
                    colorbar=dict(
                        thickness=20,
                        separatethousands=True,
                        title_text="Elevation (ft)",
                        ticksuffix=" ft",
                    ),
                    coloraxis="coloraxis",
                    opacity=0.7,
                ),
            ),
            go.Scattermapbox(
                name="Predictor Site(s)",
                lon=df_preds["longitude"],
                lat=df_preds["latitude"],
                hovertext=df_preds["text"],
                mode="markers",
                marker=dict(
                    color=df_preds["elevation"],
                    size=23,
                    colorscale=colorscale,
                    coloraxis="coloraxis",
                    showscale=False,
                    opacity=0.9,
                ),
            ),
            go.Scattermapbox(
                name="Modeled Site",
                lon=df_resp["longitude"],
                lat=df_resp["latitude"],
                hovertext=df_resp["text"],
                mode="markers",
                marker=dict(
                    color=df_resp["elevation"],
                    size=30,
                    colorscale=colorscale,
                    coloraxis="coloraxis",
                    showscale=False,
                ),
            ),
        ],
    )

    lat_spread = abs(max(df_meta["latitude"]) - min(df_meta["latitude"]))
    lon_spread = abs(max(df_meta["longitude"]) - min(df_meta["longitude"]))
    max_bound = max(lat_spread, lon_spread) * 111
    zoom = 11.5 - math.log(max_bound)

    fig.update_layout(
        title="Filtered Sites",
        margin=dict(t=35, b=5, l=5, r=5),
        hovermode="closest",
        height=700,
        mapbox=dict(
            style="white-bg",
            zoom=zoom,
            center=dict(
                lat=df_meta["latitude"].mean(),
                lon=df_meta["longitude"].mean()
            ),
            layers=[
                {
                    "below": 'traces',
                    "sourcetype": "raster",
                    "sourceattribution": "U.S. Geological Survey",
                    "source": [
                        "https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}"
                    ]
                }
            ]
        ),
        showlegend=True,
        legend={"orientation": "h", "title": ""},
    )

    if not selected_site_meta.empty:
        fig.update_layout(
            mapbox_center=dict(
                lon=selected_site_meta["longitude"].values[0],
                lat=selected_site_meta["latitude"].values[0],
            )
        )

    site_map = dcc.Graph(
        figure=fig,
        config=get_plot_config(),
        style={"height": "75vh"},  # Set explicit height for map rendering
    )

    return site_map


if __name__ == "__main__":
    print("I do nothing, just non dynamic components that take up space...")
