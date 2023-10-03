import threading
import time as time

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from astropy.io import fits
from dash import Input, Output, State, dcc, html
from dash.dependencies import Input, Output, State
from ecallisto_ng.data_processing.utils import (
    elimwrongchannels,
    subtract_constant_background,
    subtract_rolling_background,
)
from ecallisto_ng.plotting.plotting import plot_spectogram
from ecallisto_ng.plotting.utils import fill_missing_timesteps_with_nan
from flask import Flask, send_file

from dash_utils import (
    generate_datetime_picker,
    generate_download_guide,
    generate_ecallisto_info,
    generate_intro,
    generate_load_button,
    generate_nav_bar,
    generate_user_guide,
)
from database_functions import (
    check_if_table_has_data_between_dates_sql,
    get_table_names_sql,
    get_table_names_with_data_between_dates_sql,
    sql_result_to_df,
    timebucket_values_from_database_sql,
)
from database_utils import get_table_names_sql
from logging_utils import GLOBAL_LOGGER as LOGGER
from plotting_utils import timedelta_to_sql_timebucket_value
from rest_api import return_header_from_newest_spectogram

# Use a Bootstrap stylesheet
app = dash.Dash(
    __name__,
    url_base_pathname="/ecallisto_dashboard/",
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)
server = Flask(__name__, static_folder="assets")
navbar = generate_nav_bar()

# Define some constants
RESOLUTION_WIDTH = 2000
INTERVAL = 5 * 1000  # 5 seconds
# Get LOGGER
LOGGER = app.logger

# for gunicorn
server = app.server


# Generate some constants
def generate_options_instrument(list_of_instruments):
    if len(list_of_instruments) == 0:
        return []
    options_instrument = [{"label": i, "value": i} for i in list_of_instruments]
    # Sort by label
    options_instrument = sorted(options_instrument, key=lambda k: k["label"])
    # Add Top 3 instruments
    options_instrument.insert(0, {"label": "All", "value": "all"})
    return options_instrument


## Global variables
options_instrument = generate_options_instrument(get_table_names_sql())
dfs_query = {}
graphs = []

# Define the layout of the app
app.layout = html.Div(
    [
        dcc.Interval(id="graph-update-interval", interval=INTERVAL, n_intervals=0),
        navbar,
        dbc.Col(
            dbc.NavbarBrand(
                "eCallisto radiospectrograms",
                className="ml-2",
                style={"font-size": "3em"},
            )
        ),  # Increase font size
        dbc.Row(generate_intro()),
        html.H3("User Usage"),  # Add title for user-focused section
        dbc.Row(generate_user_guide()),
        html.H4("Download"),  # Add title for developer-focused section
        dbc.Row(generate_download_guide()),
        html.H5("Ecallisto NG"),
        dbc.Row(generate_ecallisto_info()),
        html.Div(
            [
                generate_datetime_picker(),
                generate_load_button(options_instrument),
            ]
        ),
        html.Div(id="graphs-container", children=[]),
        html.Div(id="dummy-output", style={"display": "none"}),
        dcc.Store(id="store-update-state", data={"n": 0}),
    ]
)


@app.callback(
    [
        Output("instrument-dropdown", "options"),
        Output("instrument-loading-state", "data"),
    ],
    [
        Input("start-datetime-picker", "value"),
        Input("end-datetime-picker", "value"),
    ],
)
def update_instrument_dropdown_options(start_datetime, end_datetime):
    # initially set loading state to True
    loading_state = False
    options = []
    if start_datetime and end_datetime:
        available_instruments = get_table_names_with_data_between_dates_sql(
            start_datetime, end_datetime
        )
        options = generate_options_instrument(available_instruments)
    else:
        options = generate_options_instrument(get_table_names_sql())
    loading_state = False
    return options, loading_state


# This is the callback function that updates the graphs whenever the date range or instrument is changed by the user.
@app.callback(
    [
        Output("dummy-output", "children"),
    ],
    [
        Input("load-data-button", "n_clicks"),
    ],
    [
        State("instrument-dropdown", "value"),
        State("start-datetime-picker", "value"),
        State("end-datetime-picker", "value"),
        State("background-sub-dropdown", "value"),
        State("elim-wrong-channels-checklist", "value"),
        State("color-scale-dropdown", "value"),
    ],
    prevent_initial_call=True,
)
def start_data_fetching(
    n_clicks,
    instruments,
    start_date,
    end_date,
    backsub_option,
    elim_option,
    color_scale,
):
    # Make sure that the graphs are empty
    graphs.clear()
    
    if n_clicks < 1:
        pass
    else:
        start_datetime = pd.to_datetime(start_date)
        end_datetime = pd.to_datetime(end_date)
        time_delta = (end_datetime - start_datetime) / RESOLUTION_WIDTH
        timebucket_value = timedelta_to_sql_timebucket_value(time_delta)

        if isinstance(instruments, str):
            instruments = [instruments]
        if "all" in instruments:
            instruments = get_table_names_sql()

        threading.Thread(
            target=_generate_plots,
            args=(
                instruments,
                start_datetime,
                end_datetime,
                timebucket_value,
                backsub_option,
                elim_option,
                color_scale,
            ),
        ).start()
    return dash.no_update


@app.callback(
    Output("graphs-container", "children", allow_duplicate=True),
    [Input("graph-update-interval", "n_intervals")],
    [State("graphs-container", "children")],
    prevent_initial_call=True,
)
def update_graph_display(n, current_children):
    global graphs
    if current_children is None:
        current_children = []
    if graphs is None:
        graphs = []
    if isinstance(current_children, dict):
        print(current_children.keys())
    if isinstance(graphs, dict):
        print(graphs.keys())
    new_children = current_children + graphs

    # Clear graphs
    graphs.clear()
    return new_children


def _generate_plots(
    instruments,
    start_datetime,
    end_datetime,
    timebucket_value,
    backsub_option,
    elim_option,
    color_scale,
):
    global graphs
    global dfs_query
    dfs_query = {}
    for instrument in instruments:
        try:
            query = {
                "table": instrument,
                "start_datetime": start_datetime.strftime("%Y-%m-%d %H:%M:%S.%f"),
                "end_datetime": end_datetime.strftime("%Y-%m-%d %H:%M:%S.%f"),
                "timebucket": f"{timebucket_value}",
                "agg_function": "MAX",
            }
            LOGGER.info(f"Query: {query}")
            # Check if table has data between dates
            if (
                check_if_table_has_data_between_dates_sql(
                    query["table"], query["start_datetime"], query["end_datetime"]
                )
                == False
            ):
                warning_message = f"No data available for instrument '{instrument}' within the specified time frame."
                warning_div = html.Div(
                    warning_message, style={"color": "red", "margin-top": "10px"}
                )
                graphs.append(warning_div)
                continue
            # Create data
            sql_result = timebucket_values_from_database_sql(**query)
            df = sql_result_to_df(sql_result)
            LOGGER.info(f"DataFrame shape: {df.shape}")
            # Get header information
            try:
                meta_data = return_header_from_newest_spectogram(df, instrument)
            except Exception as e:
                LOGGER.error(f"Could not get header information: {e}")
                meta_data = {}

            # Add header information to DataFrame
            for key, value in meta_data.items():
                df.attrs[key] = value
            # Store unedited data
            dfs_query[instrument] = query
            # Create download button with a unique ID
            download_link = html.A(
                "Download",
                id={"type": "download", "index": instrument},
                className="download-button",  # Optional, for styling like a button
                href=f"/ecallisto_dashboard/download/{instrument}",
            )

            # Sync time axis between plots
            df = fill_missing_timesteps_with_nan(df)
            # Background subtraction
            if elim_option == ["elim"]:
                df = elimwrongchannels(df)
            if backsub_option == "constant":
                df = subtract_constant_background(df)
            elif backsub_option == "rolling":
                df = subtract_rolling_background(df, window=10)
            # Plotting
            df = fill_missing_timesteps_with_nan(df, start_datetime, end_datetime)
            fig = plot_spectogram(
                df, instrument, start_datetime, end_datetime, color_scale=color_scale
            )
            fig_style = {"display": "block"} if fig else {"display": "none"}
            graph = dcc.Graph(
                id={"type": "live-graph", "index": instrument},
                figure=fig,
                config={
                    "displayModeBar": True,
                    "staticPlot": False if len(instruments) <= 1 else True,
                    "modeBarButtonsToRemove": [
                        "zoom2d",
                        "pan2d",
                        "zoomIn2d",
                        "zoomOut2d",
                        "autoScale2d",
                        "hoverClosestCartesian",
                        "hoverCompareCartesian",
                    ],
                },
                style=fig_style,
                responsive=True if len(instruments) <= 1 else False,
            )
            LOGGER.info(f"Setting button ID with instrument: {instrument}")
            graphs.append(html.Div([graph, download_link]))
        except Exception as e:
            LOGGER.error(f"Error: {e}")
            continue


# This is the callback function that updates the graphs whenever the date range or instrument is changed by the user.
@server.route("/ecallisto_dashboard/download/<instrument>")
def download(instrument):
    query = dfs_query[instrument]  # Get DataFrame

    # Create data
    sql_result = timebucket_values_from_database_sql(**query)
    df = sql_result_to_df(sql_result)

    # Transpose
    df = df.T

    # Convert DataFrame to 2D NumPy array (assumes df is already 2D)
    data_array = df.to_numpy()

    # Create header
    header = fits.Header()
    for key, value in df.attrs.items():
        header[key] = value

    # Create ImageHDU
    hdu = fits.ImageHDU(data=data_array, header=header)

    # Write to FITS file
    file_path = f"./_tmp/temp_{instrument}.fits"
    hdu.writeto(file_path, overwrite=True)

    return send_file(file_path, as_attachment=True, download_name=f"{instrument}.fits")


# Run the app
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8051, dev_tools_props_check=False)
