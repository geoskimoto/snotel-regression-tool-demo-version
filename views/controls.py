# -*- coding: utf-8 -*-
from datetime import date
from dateutil.relativedelta import relativedelta

from dash import dcc, html
import dash_bootstrap_components as dbc

from views.static import tidbits

INTIAL_TRAINING_YRS = 10


def get_initial_begin_date(initial_training_years=INTIAL_TRAINING_YRS):
    return date.today() - relativedelta(years=initial_training_years)


PARAMETER_OPTIONS = [
    {"label": "Snow Water Equivalent", "value": "WTEQ"},
    {"label": "Accumulative Precipitation", "value": "PREC"},
    # {"label": "Precipitation", "value": "PRCP"},
    {"label": "Snow Depth", "value": "SNWD"},
    {"label": "Average Temperature", "value": "TAVG"},
    {"label": "Accumulative WTEQ (use to estimate Accum Prec)", "value": "WTEQ - Accumulative"},
    # {"label": "Observed Temperature", "value": "TOBS"},
    # {"label": "Max Temperature", "value": "TMAX"},
    # {"label": "Min Temperature", "value": "TMIN"},
    {"label": "Soil Moisture Percent", "value": "SMS"},
    {"label": "Soil Temperature Observed", "value": "STO"}
]

MODEL_OPTIONS = [
    {"label": "Linear", "value": "Linear"},
    {"label": "Lasso", "value": "Lasso"},
    {"label": "Huber", "value": "Huber"},
    {"label": "SVM w/ rbf kernel", "value": "SVM"},
    {"label": "Random Forest", "value": "Random Forest"},
    {"label": "AdaBoost", "value": "AdaBoost"},
    {"label": "XGBoost", "value": "XGBoost"},
]


def get_control_view(newest_begin_date=get_initial_begin_date()):
    return dbc.Card(
            id="input-card",
            className="p-1",
            children=[
                dbc.Form(
                    id="input-form",
                    children=[
                        html.Div(
                            children=[
                                dbc.Label(
                                    "Filter by nearest sites:",
                                    html_for="station-selector",
                                ),
                                dcc.Slider(
                                    id="station-selector",
                                    min=0,
                                    max=100,
                                    step=10,
                                    value=10,
                                    tooltip={
                                        "placement": "bottom",
                                        "always_visible": True,
                                    },
                                    className="my-1",
                                ),
                                dbc.Label(
                                    "Station/element to model:",
                                    html_for="response-station",
                                ),
                                dcc.Dropdown(
                                    id="response-station",
                                    options=[],
                                    value="301:CA:SNTL",
                                    placeholder="Select a station",
                                ),
                                dbc.Select(
                                    id="response-parameter",
                                    options=PARAMETER_OPTIONS,
                                    value="WTEQ",
                                    placeholder="Select an parameter",
                                ),
                            ]
                        ),
                        html.Div(
                            children=[
                                dbc.Label(
                                    "Station/element(s) for regression:",
                                    html_for="predictor-station1",
                                ),
                                dbc.RadioItems(
                                    options=[
                                        {"label": "Alphabetical", "value": "name"},
                                        {"label": "Proximity", "value": "proximity"},
                                    ],
                                    value="name",
                                    id="sort-pred-selectors",
                                    inline=True,
                                ),
                                dbc.Select(
                                    id="predictor-station1",
                                    options=[],
                                    value="301:CA:SNTL",
                                    placeholder="Select a station",
                                ),
                                dbc.Select(
                                    id="predictor-parameter1",
                                    options=PARAMETER_OPTIONS,
                                    value="SNWD",
                                    placeholder="Select a parameter",
                                ),
                                dbc.Select(
                                    id="predictor-station2",
                                    options=[],
                                    value="391:CA:SNTL",
                                    className="mt-2",
                                    placeholder="Select a station",
                                ),
                                dbc.Select(
                                    id="predictor-parameter2",
                                    options=PARAMETER_OPTIONS,
                                    value="WTEQ",
                                    placeholder="Select a parameter",
                                ),
                                dbc.Select(
                                    id="predictor-station3",
                                    options=[],
                                    className="mt-2",
                                    placeholder="Select a station",
                                ),
                                dbc.Select(
                                    id="predictor-parameter3",
                                    options=PARAMETER_OPTIONS,
                                    value=None,
                                    placeholder="Select a parameter",
                                ),
                                dbc.Select(
                                    id="predictor-station4",
                                    options=[],
                                    className="mt-2",
                                    placeholder="Select a station",
                                ),
                                dbc.Select(
                                    id="predictor-parameter4",
                                    options=PARAMETER_OPTIONS,
                                    value=None,
                                    placeholder="Select a parameter",
                                ),
                            ],
                            className="my-2",
                        ),
                        html.Div(
                            children=[
                                dbc.Label(
                                    "Date range to train model on:",
                                    html_for="train-model-date-grp",
                                ),
                                html.Div(
                                    id="train-model-date-grp",
                                    children=[
                                        dbc.Label("Start:", html_for="startdate_picker"),
                                        dbc.Input(
                                            id="startdate_picker",
                                            type="date",
                                            min="1950-10-01",
                                            value=str(newest_begin_date),
                                        ),
                                        dbc.Label("End:", html_for="enddate_picker"),
                                        dbc.Input(
                                            id="enddate_picker",
                                            type="date",
                                            min="1950-10-01",
                                            value=str(date.today() - relativedelta(days=15)),
                                        ),
                                    ],
                                ),
                                dbc.Label(
                                    "Regressor type:", html_for="model_selection"
                                ),
                                dbc.Select(
                                    id="model_selection",
                                    options=MODEL_OPTIONS,
                                    value="Lasso",
                                ),
                                dbc.Label(
                                    "Train then predict:",
                                    html_for="submit-button-training",
                                ),
                                html.Div(
                                    [                                      
                                        dbc.Button(
                                            id="submit-button-training",
                                            children="Train Model",
                                            color="secondary",
                                            #className="",    
                                        ),
                                    ]
                                ),
                            ],
                            className="my-2",
                        ),
                        html.Div(
                            [
                                dbc.Label(
                                    "Run well trained model:",
                                    html_for="run-model-date-grp",
                                ),
                                html.Div(
                                    id="run-model-date-grp",
                                    children=[
                                        dbc.Label("Start:", html_for="predict_startdate_picker"),
                                        dbc.Input(
                                            id="predict_startdate_picker",
                                            type="date",
                                            value=str(date.today() - relativedelta(months=1)),
                                        ),
                                        dbc.Label("End:", html_for="predict_enddate_picker"),
                                        dbc.Input(
                                            id="predict_enddate_picker",
                                            type="date",
                                            value=str(date.today()),
                                        ),
                                    ],
                                ),
                            ],
                            className="my-2",
                        ),
                        html.Div(
                            [
                                dbc.Button(
                                    id="submit-button-predictions",
                                    children="Run Predictions",
                                    color="secondary",
                                    className="",
                                )
                            ],
                            className="my-2",
                        ),
                        html.Div(
                            html.Pre(id='text box', children=[])),
                        html.Details(children=tidbits, className="my-2"),
                    ],
                )
            ],
        )


if __name__ == "__main__":
    print("I do nothing, just dynamic components that take up space...")
