from datetime import datetime, timedelta

import dash_bootstrap_components as dbc
import dash_daq as daq
import plotly.express as px
from dash import dcc, html


def generate_datetime_picker():
    return html.Div(
        [
            dcc.Input(
                id="start-datetime-picker",
                type="datetime-local",
                value=(datetime.now() - timedelta(days=1)).replace(
                    minute=0, second=0, microsecond=0
                ),
            ),
            dcc.Input(
                id="end-datetime-picker",
                type="datetime-local",
                value=datetime.now().replace(minute=0, second=0, microsecond=0),
            ),
        ],
        style={"width": "100%", "display": "block", "margin-top": "60px"},
    )


def generate_plotting_options():
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.H5("Background Subtraction"),  # Title
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
                                    value="constant",  # default value
                                    multi=False,
                                ),
                            ]
                        ),
                        width={"size": 3, "offset": 0},
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.H5("Channel Operations"),  # Title
                                dcc.Checklist(
                                    id="elim-wrong-channels-checklist",
                                    options=[
                                        {
                                            "label": "Eliminate Wrong Channels",
                                            "value": "elim",
                                        },
                                    ],
                                    value=["elim"],
                                ),
                            ]
                        ),
                        width={"size": 3, "offset": 0},
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.H5(
                                    "Virtual Antenna by combining antennas"
                                ),  # Title
                                html.P(
                                    [
                                        "When fetching more than one antenna, you can combine them to a virtual antenna. To know how this don, please have a",
                                        html.A(
                                            " look at this notebook.",
                                            href="https://github.com/i4Ds/ecallisto_ng/blob/main/example/combination_of_signals_quantile_crosscorrelation.ipynb",
                                        ),
                                    ],
                                ),
                                html.H6("Method for combination:"),
                                dcc.Dropdown(
                                    id="combine-antennas-method",
                                    options=[
                                        {
                                            "label": "Don't combine",
                                            "value": "none",
                                        },
                                        {
                                            "label": "Quantile",
                                            "value": "quantile",
                                        },
                                    ],
                                    value="none",  # default value
                                    multi=False,
                                ),
                                # Slider for quantile value
                                html.H6("Quantile value:"),  # Title
                                dcc.Slider(
                                    id="combine-antennas-quantile",
                                    min=0.1,
                                    max=0.9,
                                    step=0.1,
                                    value=0.5,
                                ),
                            ]
                        ),
                        width={"size": 3, "offset": 0},
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.H5("Plotting Options"),
                                html.P("Select your color scale:"),
                                dcc.Dropdown(
                                    id="color-scale-dropdown",
                                    options=px.colors.named_colorscales(),
                                    value="inferno",  # default value
                                    multi=False,
                                ),
                                html.A(
                                    "See color scales",
                                    href="https://plotly.com/python/builtin-colorscales/",
                                    target="_blank",
                                ),
                            ]
                        ),
                        width={"size": 3, "offset": 0},
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
    )


def generate_intro():
    return [
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


def generate_ecallisto_info():
    return [
        dbc.Col(
            html.P(
                [
                    "We strongly recommend to use the new version of the ",
                    html.A(
                        "software, which is called eCallisto NG",
                        href="https://pypi.org/project/ecallisto-ng/",
                    ),
                ],
                style={"font-size": "1em", "margin-top": "10px"},
            ),
        ),
    ]


def generate_user_guide():
    return [
        dbc.Col(
            html.P(
                [
                    "To download the image, please click on the camera icon in the top right corner of the plot. If you encounter any problems, please contact vincenzo.timmel@fhnw.ch",
                ],
                style={"font-size": "1em", "margin-top": "10px"},
            ),  # Add user usage information
        ),
    ]


def generate_download_guide():
    return [
        dbc.Col(
            html.P(
                [
                    "To download the image, please click on the camera icon in the top right corner of the plot. To download the fits-file, please click on the download button below the plot.",
                ],
                style={"font-size": "1em", "margin-top": "10px"},
            ),  # Add user usage information
        ),
    ]


def generate_load_button(options_instrument):
    return html.Div(
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
                                value="all",
                                multi=True,
                            ),
                        ],
                        style={"width": "100%", "display": "block"},
                    ),
                    generate_plotting_options(),  # Margin top
                ],
            ),
        ],
        style={"display": "block"},  # Initially, the container is visible
    )


def generate_nav_bar():
    return dbc.Navbar(
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.Img(
                                src="./assets/logo_ecallisto_website.png", height="50px"
                            ),
                            className="p-0",
                        ),
                        dbc.Col(
                            dbc.Nav(
                                [
                                    dbc.DropdownMenu(
                                        nav=True,
                                        in_navbar=True,
                                        label="Help",
                                        children=[
                                            dbc.DropdownMenuItem("Option 1", href="#"),
                                            dbc.DropdownMenuItem("Option 2", href="#"),
                                            dbc.DropdownMenuItem("Option 3", href="#"),
                                        ],
                                    ),
                                    dbc.DropdownMenu(
                                        nav=True,
                                        in_navbar=True,
                                        label="Similar products",
                                        children=[
                                            dbc.DropdownMenuItem("Option 1", href="#"),
                                            dbc.DropdownMenuItem("Option 2", href="#"),
                                            dbc.DropdownMenuItem("Option 3", href="#"),
                                        ],
                                    ),
                                    dbc.DropdownMenu(
                                        nav=True,
                                        in_navbar=True,
                                        label="Related products",
                                        children=[
                                            dbc.DropdownMenuItem("Option 1", href="#"),
                                            dbc.DropdownMenuItem("Option 2", href="#"),
                                            dbc.DropdownMenuItem("Option 3", href="#"),
                                        ],
                                    ),
                                    dbc.DropdownMenu(
                                        nav=True,
                                        in_navbar=True,
                                        label="Latest",
                                        children=[
                                            dbc.DropdownMenuItem("Option 1", href="#"),
                                            dbc.DropdownMenuItem("Option 2", href="#"),
                                            dbc.DropdownMenuItem("Option 3", href="#"),
                                        ],
                                    ),
                                    dbc.DropdownMenu(
                                        nav=True,
                                        in_navbar=True,
                                        label="Query",
                                        children=[
                                            dbc.DropdownMenuItem("Option 1", href="#"),
                                            dbc.DropdownMenuItem("Option 2", href="#"),
                                            dbc.DropdownMenuItem("Option 3", href="#"),
                                        ],
                                    ),
                                ],
                                className="ml-auto",
                            ),
                        ),
                    ],
                    align="center",
                    className="g-0",  # equivalent to no_gutters
                ),
            ]
        ),
        color="dark",
        dark=True,
        className="mb-5",
    )
