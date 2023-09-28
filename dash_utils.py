from datetime import datetime, timedelta

import dash_bootstrap_components as dbc
import plotly.express as px
from dash import dcc, html


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


def generate_datetime_picker():
    return html.Div(
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
                value=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                + timedelta(days=1),
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
                        width={"size": 4, "offset": 0},
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
                        width={"size": 4, "offset": 0},
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.H5("Plotting Options"),
                                html.P("Select your color scale:"),
                                dcc.Dropdown(
                                    id="color-scale-dropdown",
                                    options=px.colors.named_colorscales(),
                                    value="haline",  # default value
                                    multi=False,
                                ),
                                html.A(
                                    "See color scales",
                                    href="https://plotly.com/python/builtin-colorscales/",
                                    target="_blank",
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
    )


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
                                value="top5",
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
