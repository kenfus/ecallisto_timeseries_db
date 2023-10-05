import os
import time as time
import uuid
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Queue
from typing import List

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import torch
from astropy.io import fits
from dash import Input, Output, State, dcc, html
from dash.dependencies import Input, Output, State
from ecallisto_ng.combine_antennas.combine import (
    match_spectrograms,
    preprocess_data,
    sync_spectrograms,
)
from ecallisto_ng.combine_antennas.utils import round_col_to_nearest_bin
from ecallisto_ng.data_processing.utils import (
    elimwrongchannels,
    subtract_constant_background,
    subtract_rolling_background,
)
from ecallisto_ng.plotting.plotting import (
    fill_missing_timesteps_with_nan,
    plot_spectogram,
)
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
INTERVAL = 1 * 1000  # in milliseconds
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
graph_queue = Queue()
query_meta_data = {
    "instruments": [],
    "completed": 0,
    "status": "idle",
    "combine_antennas_method": "none",
}

# Define the layout of the app
app.layout = html.Div(
    [
        dcc.Interval(id="interval-update", interval=INTERVAL, max_intervals=0),
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
        dcc.Store(id="progress-store", data={"completed": 0}),
        dbc.Progress(
            id="progress-bar",
            label="0%",
            value=0,
            striped=True,
            animated=True,
            style={"margin-top": "20px", "margin-bottom": "20px", "display": "none"},
        ),
        html.Div(id="graphs-container", children=[]),
        dcc.Store(id="store-graphs", data=[]),
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
        Output("graphs-container", "children", allow_duplicate=True),
        Output("load-data-button", "disabled"),
        Output('interval-update', 'max_intervals', allow_duplicate=True),
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
        State("combine-antennas-method", "value"),
        State("combine-antennas-quantile", "value"),
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
    combine_antennas_method,
    combine_antennas_quantile,
    color_scale,
):
    if n_clicks < 1:
        return dash.no_update, dash.no_update, dash.no_update
    else:
        start_datetime = pd.to_datetime(start_date)
        end_datetime = pd.to_datetime(end_date)
        time_delta = (end_datetime - start_datetime) / RESOLUTION_WIDTH
        timebucket_value = timedelta_to_sql_timebucket_value(time_delta)

        if isinstance(instruments, str):
            instruments = [instruments]
        if "all" in instruments:
            instruments = get_table_names_with_data_between_dates_sql(
                start_datetime, end_datetime
            )
        # Update meta data
        query_meta_data["instruments"] = instruments
        query_meta_data["completed"] = 0
        query_meta_data["combine_antennas_method"] = combine_antennas_method
        query_meta_data["status"] = "processing"
        executor = ThreadPoolExecutor(max_workers=os.cpu_count())
        executor.submit(
            _generate_plots,
            instruments,
            start_datetime,
            end_datetime,
            timebucket_value,
            backsub_option,
            elim_option,
            combine_antennas_method,
            combine_antennas_quantile,
            color_scale,
        )
        print("Done")
        print("Returning max_intervals now")
        return [
            html.Div(
                id="force-re-render", style={"display": "none"}, children=str(uuid.uuid4())
            ),  # This is a hack to force a re-render of the graphs
            True,
            -1
        ]


def _generate_plots(
    instruments,
    start_datetime,
    end_datetime,
    timebucket_value,
    backsub_option,
    elim_option,
    combine_antennas_method,
    combine_antennas_quantile,
    color_scale,
):
    dfs = []
    # Fetch data
    for instrument in instruments:
        try:
            df = _fetch_data(
                instrument,
                start_datetime,
                end_datetime,
                timebucket_value,
            )
            if df is None:
                warning_message = f"No data available for instrument '{instrument}' within the specified time frame."
                warning_div = html.Div(
                    warning_message, style={"color": "red", "margin-top": "10px"}
                )
                graph_queue.put(warning_div)
                continue
            # Update meta data
            query_meta_data["completed"] += 1
            # Store unedited data if we want to combine antennas
            if combine_antennas_method != "none":
                df = round_col_to_nearest_bin(df, 0.2)
                dfs.append(df)
                continue
            LOGGER.info(f"DataFrame shape: {df.shape}")

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
            _create_graph_from_raw_data(
                df,
                instrument,
                start_datetime,
                end_datetime,
                color_scale,
                static_plot=True if len(instruments) > 1 else False,
                add_download_button=True,
            )

        except Exception as e:
            LOGGER.error(f"Error: {e}")
            continue
    # Combine antennas
    if combine_antennas_method != "none":
        # Preprocess data
        dfs = preprocess_data(dfs)
        # Match data, in time and frequency
        matched_data = match_spectrograms(dfs)
        # Sync data in time (e.g. if some clocks of antennas are not synced). This only works if the data does not have any nans.
        synced_data, _ = sync_spectrograms(matched_data)
        # To torch, for fast processingtorch_shifted = torch.stack([torch.from_numpy(df.values) for df in synced_data])
        ## Calcualte quantile
        torch_synced = torch.stack([torch.from_numpy(df.values) for df in synced_data])
        torch_quantile = torch.nanquantile(
            torch_synced, combine_antennas_quantile, dim=0
        )
        # Plotting
        data_quantile_df = pd.DataFrame(
            torch_quantile, columns=synced_data[1].columns, index=synced_data[1].index
        )
        LOGGER.info(f"DataFrame shape: {data_quantile_df.shape}")
        LOGGER.info("Plotting combined antennas")
        _create_graph_from_raw_data(
            data_quantile_df,
            "E-Callisto Virtual Antenna",
            start_datetime,
            end_datetime,
            color_scale,
            static_plot=False,
            add_download_button=True,
        )
        LOGGER.info("Done plotting combined antennas")
    # Update meta data
    query_meta_data["status"] = "done"


def _fetch_data(
    instrument,
    start_datetime,
    end_datetime,
    timebucket_value,
):
    q = {
        "table": instrument,
        "start_datetime": start_datetime.strftime("%Y-%m-%d %H:%M:%S.%f"),
        "end_datetime": end_datetime.strftime("%Y-%m-%d %H:%M:%S.%f"),
        "timebucket": f"{timebucket_value}",
        "agg_function": "MAX",
    }
    LOGGER.info(f"Query: {q}")
    dfs_query[instrument] = q
    # Check if table has data between dates
    if (
        check_if_table_has_data_between_dates_sql(
            q["table"], q["start_datetime"], q["end_datetime"]
        )
        == False
    ):
        return None
    # Create data
    sql_result = timebucket_values_from_database_sql(**q)
    df = sql_result_to_df(sql_result)
    return df


def _create_graph_from_raw_data(
    df,
    instrument,
    start_datetime,
    end_datetime,
    color_scale,
    static_plot=False,
    add_download_button=True,
):
    fig = plot_spectogram(
        df, instrument, start_datetime, end_datetime, color_scale=color_scale
    )
    fig_style = {"display": "block"} if fig else {"display": "none"}
    graph = dcc.Graph(
        id={"type": "live-graph", "index": instrument},
        figure=fig,
        config={
            "displayModeBar": True,
            "staticPlot": True,
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
        responsive=True,
    )
    if add_download_button:
        # Create download button with a unique ID
        download_link = html.A(
            "Download",
            id={"type": "download", "index": instrument},
            className="download-button",  # Optional, for styling like a button
            href=f"/ecallisto_dashboard/download/{instrument}",
        )
    else:
        # Create message with download link not available
        download_link = html.Div("Download not available", className="download-button")
    LOGGER.info(f"Graph created for {instrument}")
    graph_queue.put(html.Div([graph, download_link]))


# Storing of graphs etc
@app.callback(Output("store-graphs", "data"), [Input("interval-update", "n_intervals")])
def update_stored_graphs(n):
    new_graphs = []
    while not graph_queue.empty():
        new_graphs.append(graph_queue.get())
    if len(new_graphs) > 0:
        return new_graphs
    else:
        dash.no_update


@app.callback(
    Output("graphs-container", "children"),
    [Input("store-graphs", "data")],
    [State("graphs-container", "children")],
)
def update_graph_display(stored_graphs, current_children):
    if current_children is None:
        current_children = []
    if stored_graphs is None:
        stored_graphs = []
    # For some reason, sometimes current_children is a dictionary, so we need to check for that
    if not isinstance(current_children, List):
        current_children = [current_children]
    return current_children + stored_graphs


# This is the function for downloading the data
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

    header = return_header_from_newest_spectogram(df, instrument)

    # Create header
    header = fits.Header()
    for key, value in header.items():
        header[key] = value

    # Create ImageHDU
    hdu = fits.ImageHDU(data=data_array, header=header)

    # Write to FITS file
    file_path = f"./_tmp/temp_{instrument}.fits"
    hdu.writeto(file_path, overwrite=True)

    return send_file(file_path, as_attachment=True, download_name=f"{instrument}.fits")


# Boring and wayy to complex prograss bar stuff. That has to be simpler, no?
@app.callback(
    Output("progress-store", "data"),
    Input("interval-update", "n_intervals"),
    State("progress-store", "data"),
)
def update_progress(n, progress_data):
    progress_data["completed"] = len(dfs_query)
    return progress_data


@app.callback(
    [
        Output("progress-bar", "value"),
        Output("progress-bar", "label"),
        Output("progress-bar", "style"),
        Output("load-data-button", "disabled", allow_duplicate=True),
        Output("interval-update", "max_intervals", allow_duplicate=True),
    ],
    [Input("progress-store", "data"), Input("progress-bar", "style"), Input("interval-update", "max_intervals")],
    prevent_initial_call=True,
)
def update_progress_bar(data, style, max_intervals):
    total_instruments = len(query_meta_data["instruments"])
    completed = query_meta_data["completed"]
    percent_complete = (
        (completed / total_instruments) * 100 if total_instruments > 0 else 0
    )
    percent_complete = round(percent_complete, 0)
    if (
        query_meta_data["combine_antennas_method"] != "none"
        and query_meta_data["status"] != "done"
    ):
        percent_complete = min(percent_complete, 99)
    # Max interval
    if query_meta_data["status"] == "done":
        percent_complete = 100
        style["display"] = "none"
        loading_state = False
        max_intervals = 0
    elif query_meta_data["status"] == "idle":
        percent_complete = 0
        style["display"] = "none"
        loading_state = False
        # Don't change max_intervals
        max_intervals = max_intervals
    elif query_meta_data["status"] == "processing":
        style["display"] = "block"
        loading_state = True
        max_intervals = -1
    else:
        print("Unknown status")
    print(query_meta_data)
    print(max_intervals)
    return percent_complete, f"{int(percent_complete)}%", style, loading_state, max_intervals


# Run the app
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8051, dev_tools_props_check=False)
