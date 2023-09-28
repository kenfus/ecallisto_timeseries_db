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

from bulk_load_to_database_between_dates import (
    get_paths,
)  # TODO: Move this get_paths to another utils file.
from dash_utils import generate_datetime_picker, generate_load_button, generate_nav_bar
from database_functions import (
    check_if_table_has_data_between_dates_sql,
    get_table_names_sql,
    get_table_names_with_data_between_dates_sql,
    sql_result_to_df,
    timebucket_values_from_database_sql,
)
from database_utils import (
    get_last_spectrogram_from_paths_list,
    get_table_names_sql,
    instrument_name_to_glob_pattern,
)
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
    options_instrument.insert(0, {"label": "Top 5 instruments", "value": "top5"})
    return options_instrument


options_instrument = generate_options_instrument(get_table_names_sql())

# Define the layout of the app
app.layout = html.Div(
    [
        navbar,
        dbc.Col(
            dbc.NavbarBrand(
                "eCallisto radiospectrograms",
                className="ml-2",
                style={"font-size": "3em"},
            )
        ),  # Increase font size
        dbc.Row(
            [
                dbc.Col(
                    html.P(
                        [
                            "The solar radio spectrograms that can be retrieved through this interface are provided by the network e-Callisto. ",
                            html.Br(),
                            "This network consists of a common receiver, a CALLISTO spectrometer, that are installed on radio antennas spread around the globe. ",
                            html.Br(),
                            "They all observe the full Sun from diverse latitudes and longitudes. Due to the spreading, the network reaches a 24/7 observing time coverage.",
                        ],
                        style={"font-size": "1em", "margin-top": "10px"},
                    ),  # Add description
                ),
            ]
        ),
        html.H3("User Usage"),  # Add title for user-focused section
        dbc.Row(
            [
                dbc.Col(
                    html.P(
                        [
                            "When selecting `Top 5 instruments`, the five instruments with the highest signal-to-noise ratio are selected. To download the image, please click on the camera icon in the top right corner of the plot. ",
                        ],
                        style={"font-size": "1em", "margin-top": "10px"},
                    ),  # Add user usage information
                ),
            ]
        ),
        html.H4("Download"),  # Add title for developer-focused section
        dbc.Row(
            [
                dbc.Col(
                    html.P(
                        [
                            "To download the image, please click on the camera icon in the top right corner of the plot. To download the fits-file, please click on the download button below the plot.",
                        ],
                        style={"font-size": "1em", "margin-top": "10px"},
                    ),  # Add user usage information
                ),
            ]
        ),
        html.Div(
            [
                generate_datetime_picker(),
                generate_load_button(options_instrument),
            ]
        ),
        html.Div(id="graphs-container", children=[]),
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
    loading_state = True
    options = []
    if start_datetime and end_datetime:
        available_instruments = get_table_names_with_data_between_dates_sql(
            start_datetime, end_datetime
        )
        options = generate_options_instrument(available_instruments)
        loading_state = False  # set loading state to False after options are loaded
    return options, loading_state


# This is the callback function that updates the graphs whenever the date range or instrument is changed by the user.
@app.callback(
    [Output("graphs-container", "children"), Output("load-data-loading-state", "data")],
    [Input("load-data-button", "n_clicks")],
    [
        State("instrument-dropdown", "value"),
        State("start-datetime-picker", "value"),
        State("end-datetime-picker", "value"),
        State("background-sub-dropdown", "value"),
        State("elim-wrong-channels-checklist", "value"),
        State("color-scale-dropdown", "value"),  # Add this line
    ],
)
def update_graph(
    n_clicks,
    instruments,
    start_date,
    end_date,
    backsub_option,
    elim_option,
    color_scale,
):
    # Clear graphs
    loading_state = False  # initially set loading state to False
    if n_clicks < 1:
        return [], loading_state
    else:
        start_datetime = pd.to_datetime(start_date)
        end_datetime = pd.to_datetime(end_date)
        time_delta = (end_datetime - start_datetime) / RESOLUTION_WIDTH
        timebucket_value = timedelta_to_sql_timebucket_value(time_delta)

        if isinstance(instruments, str):
            instruments = [instruments]
        if instruments == ["top5"]:
            instruments = [
                "australia_assa_62",
                "glasgow_01",
                "austria_unigraz_01",
                "germany_dlr_63",
                "alaska_anchorage_01",
            ]  # Replace in future with top 5 instruments with highest signal-to-noise ratio

        graphs = []
        loading_state = True  # initially set loading state to True
        global dfs
        dfs = {}
        for instrument in instruments:
            query = {
                "table": instrument,
                "start_datetime": start_datetime.strftime("%Y-%m-%d %H:%M:%S.%f"),
                "end_datetime": end_datetime.strftime("%Y-%m-%d %H:%M:%S.%f"),
                "timebucket": f"{timebucket_value}",
                "agg_function": "MAX",
            }
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

            # Get header information
            meta_data = return_header_from_newest_spectogram(df, instrument)

            # Add header information to DataFrame
            for key, value in meta_data.items():
                df.attrs[key] = value
            # Store unedited data
            dfs[instrument] = df
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
            print(f"Setting button ID with instrument: {instrument}")
            graphs.append(html.Div([graph, download_link]))
        loading_state = False  # set loading state to False after options are loaded
        return graphs, loading_state


# For meta data
def return_header_from_newest_spectogram(df, instrument_name):
    """
    Add the header from the newest spectrogram (based on the datetime inside the df)
    to the dataframe.
    """
    df = df.copy()
    # Get last day from df
    last_day = df.index.max().date()
    # Get glob pattern
    glob_pattern = instrument_name_to_glob_pattern(instrument_name)
    # Get paths
    paths = get_paths(last_day, last_day, glob_pattern)
    # Get last spectrogram
    last_spectrogram = get_last_spectrogram_from_paths_list(paths)
    dict_ = {}
    # Add metadata
    for key, value in last_spectrogram.header.items():
        dict_[key] = value

    del last_spectrogram
    return dict_


# This is the callback function that updates the graphs whenever the date range or instrument is changed by the user.
@server.route("/ecallisto_dashboard/download/<instrument>")
def download(instrument):
    df = dfs[instrument]  # Get DataFrame

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
