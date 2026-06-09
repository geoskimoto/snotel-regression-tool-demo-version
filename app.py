#!./.venv/bin/python
# coding: utf-8

from datetime import datetime as dt
# import os
import pandas as pd
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

import dbs
from utils import get_plot_config, add_null_option
from views.controls import get_control_view, get_initial_begin_date
from views.maps import get_station_map
import views.regression as RegressionTool
# import sqlite3 as sql
# from flask_sqlalchemy import SQLAlchemy

app = dbs.app
server = app.server


app.layout = dbc.Container(
    [
        dcc.Location(id="url"),
        html.Div(
            [
                html.P(
                    [
                        html.H4(dbs.APP_TITLE),
                        html.H5(
                            "Estimate missing or bad data using regression",
                        ),
                    ],
                    style={"text-align": "center"},
                ),
            ],
            className="my-2",
        ),
        dbc.Row(
            [
                dbc.Col(get_control_view(), width=3),
                dbc.Col(
                    children=[
                        dbc.Tabs(
                            children=[
                                dbc.Tab(
                                    label="Training",
                                    children=[
                                        dbc.Row(
                                            dbc.Col(html.Div(id="traintest-plots")),
                                        ),
                                        dbc.Row(
                                            dbc.Col(html.Div(id="modelfit-plots")),
                                        ),
                                    ],
                                ),
                                dbc.Tab(
                                    label="Map",
                                    children=[
                                        dbc.Row(
                                            dbc.Col(
                                                html.Div(
                                                    id="station-map", className="p-2"
                                                ),
                                            ),
                                        ),
                                    ],
                                ),
                                dbc.Tab(
                                    label="Results",
                                    children=[
                                        dbc.Row(
                                            dbc.Col(html.Div(id="pred-plots")),
                                        ),
                                    ],
                                ),
                                dbc.Tab(
                                    label="Database Management",
                                    children=[
                                        dbc.Row(
                                            dbc.Col(
                                                children=[
                                                    html.Br(),
                                                    dbc.ButtonGroup(
                                                        [
                                                            dbc.Button(
                                                                id="view-datatable-button",
                                                                children="View Database",
                                                                color="secondary",
                                                                # className="",
                                                            ),
                                                            dbc.Button(
                                                                id="save-to-db-button",
                                                                children="Save Model to DB",
                                                                color="secondary",
                                                                # className="",
                                                            ),
                                                            dbc.Button(
                                                                id='update-db-button',
                                                                children="Save changes to DB",
                                                                color="secondary",
                                                                # className="",
                                                            ),
                                                            # dbc.Modal(
                                                            #     [
                                                            #         dbc.ModalHeader(dbc.ModalTitle("Header")),
                                                            #         dbc.ModalBody(id='save-message', children=[]),
                                                            #         dbc.ModalFooter(
                                                            #             dbc.Button(
                                                            #                 "Close", id="close", className="ms-auto", n_clicks=0
                                                            #             )
                                                            #         ),
                                                            #     ],
                                                            # id="modal",
                                                            # is_open=False,
                                                            # ),
                                                        ]
                                                    ),
                                                    html.Div(
                                                        id="save-message", children=[]
                                                    ),
                                                    html.Div(
                                                        id="update-message", children=[]
                                                    ),
                                                    # dcc.Interval(id='interval_pg', interval=86400000*7, n_intervals=0),  # activated once/week or when page refreshed
                                                    html.Div(
                                                        id="view-datatable", children=[]
                                                    )
                                                    # html.Div(id='Save-df-to-db',
                                                    #         children = [
                                                    #             dbc.Button(id='Save-df-to-db-button'),
                                                    #             ]
                                                    #         ),
                                                ]
                                            ),
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
            className="my-2",
        ),
    ],
    fluid=True,
)


def parse_url_args(arg_str):
    if not arg_str:
        return {}
    try:
        args = arg_str[1:].split("&")
        args = {i.split("=")[0]: i.split("=")[1] for i in args}
        return args
    except Exception as err:
        print(f"Error parsing url args, ignoring... - {err}")
        return {}


def nearest_sites(df_meta, resp_site=None, pred_sites=None, num_sites=10):
    try:
        num_sites = int(num_sites)
    except ValueError:
        num_sites = 10
    
    tx = df_meta[df_meta["triplet"] == resp_site]["latitude"]
    ty = df_meta[df_meta["triplet"] == resp_site]["longitude"]
    
    # Check if we have a valid response site with coordinates
    if tx.empty or ty.empty:
        # No response site selected, return first num_sites stations
        df_nearest = df_meta.head(num_sites).copy()
    else:
        def calc_dist(row, tx=tx.iloc[0], ty=ty.iloc[0]):
            return (row["latitude"] - tx) ** 2 + (row["longitude"] - ty) ** 2

        df_meta = df_meta.copy()
        df_meta["proximity"] = df_meta.apply(calc_dist, axis=1)
        df_meta.sort_values(by="proximity", inplace=True)
        df_nearest = df_meta.head(num_sites + 1).copy()
    
    triplets_in_use = tuple(set([i for i in (resp_site,) + pred_sites if i]))
    df_in_use = df_meta[df_meta["triplet"].isin(triplets_in_use)]
    df = pd.concat([df_nearest, df_in_use]).drop_duplicates(ignore_index=True)
    return df


@app.callback(
    [
        Output("response-station", "options"),
        Output("predictor-station1", "options"),
        Output("predictor-station2", "options"),
        Output("predictor-station3", "options"),
        Output("predictor-station4", "options"),
        Output("station-selector", "value"),
        Output("station-map", "children"),
        Output("startdate_picker", "value"),
    ],
    [
        Input("station-selector", "value"),
        Input("url", "search"),
        Input("station-selector", "value"),
        Input("sort-pred-selectors", "value"),
        Input("response-station", "value"),
        Input("predictor-station1", "value"),
        Input("predictor-station2", "value"),
        Input("predictor-station3", "value"),
        Input("predictor-station4", "value"),
    ],
)
def populate_dropdowns(filter_by, url_args, num_sites, sortby, resp_site, *pred_sites):
    args = parse_url_args(url_args)
    num_sites = args.get("num_sites", num_sites)
    df_meta = pd.read_sql("SELECT * FROM meta", dbs.db.engine)
    df_nearest = nearest_sites(df_meta, resp_site, pred_sites, num_sites=num_sites)
    
    # Show all stations on the map, not just nearest
    site_map = get_station_map(df_meta, resp=resp_site, preds=pred_sites)

    # Handle proximity sorting specially since it needs to be calculated
    if sortby == 'proximity':
        if resp_site:
            df_meta_sorted = df_meta.copy()
            tx = df_meta[df_meta["triplet"] == resp_site]["latitude"]
            ty = df_meta[df_meta["triplet"] == resp_site]["longitude"]
            if not tx.empty and not ty.empty:
                def calc_dist(row, tx=tx.iloc[0], ty=ty.iloc[0]):
                    return (row["latitude"] - tx) ** 2 + (row["longitude"] - ty) ** 2
                df_meta_sorted["proximity"] = df_meta_sorted.apply(calc_dist, axis=1)
                df_meta_sorted = df_meta_sorted.sort_values(by="proximity")
            else:
                df_meta_sorted = df_meta.sort_values(by="name")
        else:
            # No response station selected yet, default to alphabetical
            df_meta_sorted = df_meta.sort_values(by="name")
    else:
        df_meta_sorted = df_meta.sort_values(by=sortby)
    
    all_stations = add_null_option()
    for i, row in df_meta_sorted.iterrows():
        all_stations.append({"label": row["label"], "value": row["triplet"]})

    # Predictor stations should also show all stations, not just nearest
    nearest_stations = all_stations
    preds_begin_dates = df_meta[df_meta["triplet"].isin(pred_sites)][
        "beginDate"
    ].tolist()
    if preds_begin_dates:
        # Handle both old format (YYYY-MM-DD HH:mm:ss) and new format (YYYY-MM-DD HH:mm)
        parsed_dates = []
        for date_str in preds_begin_dates:
            try:
                parsed_dates.append(dt.strptime(date_str, "%Y-%m-%d %H:%M:%S"))
            except ValueError:
                try:
                    parsed_dates.append(dt.strptime(date_str, "%Y-%m-%d %H:%M"))
                except ValueError:
                    # Fallback: try just the date part
                    parsed_dates.append(dt.strptime(date_str.split()[0], "%Y-%m-%d"))
        max_begin_date = max(parsed_dates)
    else:
        max_begin_date = get_initial_begin_date()
    return (
        all_stations,
        nearest_stations,
        nearest_stations,
        nearest_stations,
        nearest_stations,
        num_sites,
        site_map,
        max_begin_date.strftime("%Y-%m-%d"),
    )


@app.callback(
    [Output("traintest-plots", "children"), Output("modelfit-plots", "children")],
    Input("submit-button-training", "n_clicks"),
    [
        State("response-station", "value"),
        State("response-parameter", "value"),
        State("predictor-station1", "value"),
        State("predictor-parameter1", "value"),
        State("predictor-station2", "value"),
        State("predictor-parameter2", "value"),
        State("predictor-station3", "value"),
        State("predictor-parameter3", "value"),
        State("predictor-station4", "value"),
        State("predictor-parameter4", "value"),
        State("startdate_picker", "value"),
        State("enddate_picker", "value"),
        State("model_selection", "value"),
        State("predict_startdate_picker", "value"),
        State("predict_enddate_picker", "value"),
    ],
)
def train_test_figures(
    n_clicks,
    responsestation,
    responseparameter,
    predictorstation1,
    predictorparameter1,
    predictorstation2,
    predictorparameter2,
    predictorstation3,
    predictorparameter3,
    predictorstation4,
    predictorparameter4,
    startdate,
    enddate,
    modelselection,
    predict_startdate,
    predict_enddate,
):

    station_param_pairs = [
        (responsestation, responseparameter),
        (predictorstation1, predictorparameter1),
        (predictorstation2, predictorparameter2),
        (predictorstation3, predictorparameter3),
        (predictorstation4, predictorparameter4),
    ]

    station_param_pairs[:] = [i for i in station_param_pairs if i[0] and i[1]]
    if len(station_param_pairs) < 2:
        return (html.Div(), html.Div())

    model = RegressionTool.RegressionFun(
        station_param_pairs, str(startdate), str(enddate)
    )

    model.train_model(modelselection, 0.3)
    model.make_predictions(predict_startdate, predict_enddate)

    traintest_graph = (
        html.Div(
            children=[
                dcc.Graph(
                    figure=model.traintest_fig,
                    responsive=True,
                    config=get_plot_config(),
                ),
            ],
        ),
    )
    modelfit_graphs = html.Div(
        children=[
            dcc.Graph(
                figure=model.modelfit_fig,
                responsive=True,
                style={"height": "95vh"},
                config=get_plot_config(),
            ),
        ],
    )

    return [traintest_graph, modelfit_graphs]


@app.callback(
    [
        Output("pred-plots", "children"),
    ],
    [Input("submit-button-predictions", "n_clicks")],
    [
        State("response-station", "value"),
        State("response-parameter", "value"),
        State("predictor-station1", "value"),
        State("predictor-parameter1", "value"),
        State("predictor-station2", "value"),
        State("predictor-parameter2", "value"),
        State("predictor-station3", "value"),
        State("predictor-parameter3", "value"),
        State("predictor-station4", "value"),
        State("predictor-parameter4", "value"),
        State("startdate_picker", "value"),
        State("enddate_picker", "value"),
        State("model_selection", "value"),
        State("predict_startdate_picker", "value"),
        State("predict_enddate_picker", "value"),
    ],
    prevent_initial_call=True,
)
def train_pred_figures(
    n_clicks,
    responsestation,
    responseparameter,
    predictorstation1,
    predictorparameter1,
    predictorstation2,
    predictorparameter2,
    predictorstation3,
    predictorparameter3,
    predictorstation4,
    predictorparameter4,
    startdate,
    enddate,
    modelselection,
    predict_startdate,
    predict_enddate,
):

    station_param_pairs = [
        (responsestation, responseparameter),
        (predictorstation1, predictorparameter1),
        (predictorstation2, predictorparameter2),
        (predictorstation3, predictorparameter3),
        (predictorstation4, predictorparameter4),
    ]
    station_param_pairs[:] = [i for i in station_param_pairs if i[0] and i[1]]
    if len(station_param_pairs) < 2:
        ## TODO: make a canned response to a empty predictor set
        return (html.Div(),)

    # Run the regression analysis
    model_Pr = RegressionTool.RegressionFun(
        station_param_pairs, str(startdate), str(enddate)
    )

    model_Pr.train_model(modelselection, 0.01)
    model_Pr.make_predictions(predict_startdate, predict_enddate)

    predictions_graph = (
        html.Div(
            children=[


                dcc.Graph(
                    figure=model_Pr.predictions_fig,
                    responsive=True,
                    config=get_plot_config(),
                ),


                dash_table.DataTable(
                    id="predictions-datatable",
                    columns=[
                        {
                            "name": str(column),
                            "id": str(column),
                            "deletable": False,
                        }
                        for column in model_Pr.predict_data.columns
                    ],
                    data=model_Pr.predict_data.to_dict("records"),
                    editable=False,
                    row_deletable=False,
                    filter_action="none",
                    sort_action="native",
                    sort_mode="none",
                    page_action="none",
                    style_table={"height": "500px", "overflowY": "auto", "overflowX": "auto"},
                    style_cell={
                        "textAlign": "center",
                    #     "minWidth": "20px",
                    #     "width": "40px",
                    #     "maxWidth": "100px",
                        "overflow": "native",
                    #     "textOverflow": "ellipsis",
                    },
                ),
            ]
        ),
    )

    return (predictions_graph)




# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------


#Callback to show in a Dash data_table the table regression_models from the regr_models.db


@app.callback(
    Output("view-datatable", "children"),
    # Input("interval-pg", "n_intervals"),
    Input("view-datatable-button", "n_clicks"),
    prevent_initial_call=False,
)
def datatable(n_clicks):

    df = pd.read_sql_query(
        "SELECT * from regression_models", con=dbs.db.engines["regr_models"]
    )
    return [
        dash_table.DataTable(
            id="regr-models-datatable",
            columns=[
                {
                    "name": str(x),
                    "id": str(x),
                    "deletable": False,
                }
                for x in df.columns
            ],
            data=df.to_dict("records"),
            editable=False,
            row_deletable=True,
            filter_action="native",
            sort_action="native",  # give user capability to sort columns
            sort_mode="single",  # sort across 'multi' or 'single' columns
            page_action="none",  # render all of the data at once. No paging.
            style_table={"height": "300px", "overflowY": "auto"},
            style_cell={
                "textAlign": "left",
                "minWidth": "100px",
                "width": "100px",
                "maxWidth": "100px",
                "overflow": "hidden",
                "textOverflow": "ellipsis",
            },
        ),
    ]


# -----------------------------------------------------------------------------------------------------------------


# Callback to save the model to the database


@app.callback(
    # Output('save-message', 'is_open'),
    Output("save-message", "children"),
    Input("save-to-db-button", "n_clicks"),
    [
        State("response-station", "value"),
        State("response-parameter", "value"),
        State("predictor-station1", "value"),
        State("predictor-parameter1", "value"),
        State("predictor-station2", "value"),
        State("predictor-parameter2", "value"),
        State("predictor-station3", "value"),
        State("predictor-parameter3", "value"),
        State("predictor-station4", "value"),
        State("predictor-parameter4", "value"),
        State("startdate_picker", "value"),
        State("enddate_picker", "value"),
        State("model_selection", "value"),
        # State("save_message", "is_open")
    ],
    prevent_initial_call=True,
)
def save_model(
    n_clicks,
    responsestation,
    responseparameter,
    predictorstation1,
    predictorparameter1,
    predictorstation2,
    predictorparameter2,
    predictorstation3,
    predictorparameter3,
    predictorstation4,
    predictorparameter4,
    startdate,
    enddate,
    modelselection,
    # is_open
):

    station_param_pairs = [
        (responsestation, responseparameter),
        (predictorstation1, predictorparameter1),
        (predictorstation2, predictorparameter2),
        (predictorstation3, predictorparameter3),
        (predictorstation4, predictorparameter4),
    ]

    station_param_pairs[:] = [i for i in station_param_pairs if i[0] and i[1]]

    # Run the regression analysis
    model = RegressionTool.RegressionFun(
        station_param_pairs, str(startdate), str(enddate)
    )
    model.train_model(modelselection, 0.3)

    try:
        data = dbs.Regression_Models(
            model.stationparameterpairs[0][0],
            str(model.stationparameterpairs),
            model.begindate,
            model.enddate,
            model.regressor_type,
            model.RMSE_train,
            model.RMSE_test,
            str(model.regr_data_string),
        )

        dbs.db.session.add(data)
        dbs.db.session.commit()
        dbs.db.session.close()

        # if n_clicks:
        #     return not is_open
        # return is_open
        return html.H1('Save successful!')
    # except #con.Error as err:
    #     print('Some shit went down')
    # return {err}
    # return html.H1(f'Error occured while saving model {err}')

    finally:
        dbs.db.session.close()
        # return '<h1> Hello </h1>'

    # df = pd.read_sql('regression_models', con=conn)


# --------------------------------------------------------------------------------------------------


#  Callback to save changes made in dash data_table to the regression db

@app.callback(
    Output('update-message', 'children'),
    Input('update-db-button', 'n_clicks'),
    [
         State('regr-models-datatable', 'data')
      
     ]
    
    )

def update_database(n_clicks, datatable):

    #convert data table with the changes to a df:   
    df_datatable = pd.dataFrame(datatable)
    print(df_datatable.iloc[0,0])
    #retrieve the unchanged table from the db:   
    con=dbs.db.engines["regr_models"]
    
    # df_db = pd.read_sql_query(
    #     "SELECT * from regression_models", con=dbs.db.get_engine(bind="regr_models")
    # )
    
    df_datatable.to_sql("regression_table", con=con, if_exists='replace', index=False)
    # for index, row in df.iterrows():
    #     query = f'update regression_models set column = {} where column = {}'
    
    
    return html.H1('Nada mae')
























# --------------------------------------------------------------------------------------------------


# # def toggle_modal(n1, n2, is_open):
# #     if n1 or n2:
# #         return not is_open
# #     return is_open

#     except con.Error as err:
#         return f'<h1> Error occured while saving model: {err} </h1>'

#     finally:
#         con.close()






if __name__ == "__main__":

    import argparse

    cli_desc = """
    Run the SNOTEL Regression Tool locally for development
    """
    parser = argparse.ArgumentParser(description=cli_desc)
    parser.add_argument(
        "-V", "--version", help="show program version", action="store_true"
    )
    parser.add_argument("-d", "--debug", help="run in debug mode", action="store_true")
    parser.add_argument("--port", help="set port to deploy on", default=5000)
    args = parser.parse_args()

    app.run_server(
        port=args.port,
        debug=args.debug,
    )
