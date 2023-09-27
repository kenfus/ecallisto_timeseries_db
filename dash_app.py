import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html
from database_functions import (
    timebucket_values_from_database_sql,
    sql_result_to_df,
    get_table_names_sql,
    check_if_table_has_data_between_dates_sql,
    get_table_names_with_data_between_dates_sql,
)
from plotting_utils import timedelta_to_sql_timebucket_value
from ecallisto_ng.plotting.utils import fill_missing_timesteps_with_nan
from ecallisto_ng.plotting.plotting import plot_spectogram
from ecallisto_ng.data_processing.utils import (
    subtract_rolling_background,
    elimwrongchannels,
    subtract_constant_background,
)
import pandas as pd
from datetime import datetime, timedelta
import dash_bootstrap_components as dbc
from dash_utils import generate_nav_bar

# Use a Bootstrap stylesheet
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
navbar = generate_nav_bar()

# Define some constants
RESOLUTION_WIDTH = 1080

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
                            "When selecting `Top 5 instruments`, the five instruments with the highest signal-to-noise ratio are selected."
                        ],
                        style={"font-size": "1em", "margin-top": "10px"},
                    ),  # Add user usage information
                ),
            ]
        ),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Input(
                            id="start-datetime-picker",
                            type="datetime-local",
                            value=datetime.now()
                            .replace(hour=0, minute=0, second=0, microsecond=0)
                            .strftime("%Y-%m-%dT%H:%M"),
                        ),
                        dcc.Input(
                            id="end-datetime-picker",
                            type="datetime-local",
                            value=datetime.now().replace(
                                hour=0, minute=0, second=0, microsecond=0
                            )
                            + timedelta(days=1),
                        ),
                    ],
                    style={"width": "100%", "display": "block", "margin-top": "60px"},
                ),
                html.Div(
                    id="instrument-and-load-button-container",  # This is the new Div container
                    children=[
                        dcc.Store(
                            id="instrument-loading-state", data=False
                        ),  # initially not loading
                        dcc.Loading(
                            id="loading-instrument-dropdown",
                            type="default",
                            children=[
                                html.Div(
                                    [
                                        dcc.Dropdown(
                                            id="instrument-dropdown",
                                            options=options_instrument,
                                            value="top5",
                                            multi=True,
                                        ),
                                    ],
                                    style={"width": "100%", "display": "block"},
                                ),
                                html.Div(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            html.H5(
                                                                "Background Subtraction"
                                                            ),  # Title
                                                            dcc.Dropdown(
                                                                id="background-sub-dropdown",
                                                                options=[
                                                                    {
                                                                        "label": "None",
                                                                        "value": "none",
                                                                    },
                                                                    {
                                                                        "label": "Constant BackSub",
                                                                        "value": "constant",
                                                                    },
                                                                    {
                                                                        "label": "Rolling BackSub",
                                                                        "value": "rolling",
                                                                    },
                                                                ],
                                                                value="none",  # default value
                                                                multi=False,
                                                            ),
                                                        ]
                                                    ),
                                                    width={"size": 6, "offset": 0},
                                                ),
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            html.H5(
                                                                "Channel Operations"
                                                            ),  # Title
                                                            dcc.Checklist(
                                                                id="elim-wrong-channels-checklist",
                                                                options=[
                                                                    {
                                                                        "label": "Eliminate Wrong Channels",
                                                                        "value": "elim",
                                                                    },
                                                                ],
                                                                value=[],  # default value, empty means not selected
                                                            ),
                                                        ]
                                                    ),
                                                    width={"size": 6, "offset": 0},
                                                ),
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            html.H5(
                                                                "Plotting Options"
                                                            ),  # New Title
                                                            dcc.Dropdown(
                                                                id="color-scale-dropdown",
                                                                options=[
                                                                    {
                                                                        "label": "Plasma",
                                                                        "value": "Plasma",
                                                                    },
                                                                    {
                                                                        "label": "Viridis",
                                                                        "value": "Viridis",
                                                                    },
                                                                    {
                                                                        "label": "Cividis",
                                                                        "value": "Cividis",
                                                                    },
                                                                ],
                                                                value="Plasma",  # default value
                                                                multi=False,
                                                            ),
                                                        ]
                                                    ),
                                                    width={"size": 4, "offset": 0},
                                                ),
                                            ],
                                            className="mb-4",
                                        ),  # Margin bottom
                                        html.Div(
                                            id="load-data-button-container",
                                            children=[
                                                dcc.Store(
                                                    id="load-data-loading-state",
                                                    data=False,
                                                ),  # initially not loading
                                                dcc.Loading(
                                                    id="loading-button",
                                                    type="default",  # other types: 'circle', 'cube', 'dot'
                                                    children=[
                                                        html.Button(
                                                            "Load Data",
                                                            id="load-data-button",
                                                            n_clicks=0,
                                                        )
                                                    ],
                                                ),
                                            ],
                                        ),
                                    ],
                                    className="mt-4",
                                ),  # Margin top
                            ],
                        ),
                    ],
                    style={"display": "block"},  # Initially, the container is visible
                ),
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
    [Input("date-picker-range", "start_date"), Input("date-picker-range", "end_date")],
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
    loading_state = False  # initially set loading state to False
    if n_clicks > 0:
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
        loading_state = True  # initially set loading state to Tr
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
                id=f"live-graph-{instrument}",
                figure=fig,
                config={
                    "displayModeBar": True,
                    "modeBarButtonsToRemove": [
                        "zoom2d",
                        "pan2d",
                        "zoomIn2d",
                        "zoomOut2d",
                        "autoScale2d",
                        "resetScale2d",
                        "hoverClosestCartesian",
                        "hoverCompareCartesian",
                    ],
                },
                style=fig_style,
                responsive=True,
            )
            graphs.append(graph)
        loading_state = False  # set loading state to False after options are loaded
        return graphs, loading_state
    return [], loading_state


# Run the app
if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8051)
