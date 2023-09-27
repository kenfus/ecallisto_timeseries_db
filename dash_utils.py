import dash_bootstrap_components as dbc
from dash import html


def generate_nav_bar():
    return dbc.Navbar(
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.Img(
                                src="/assets/logo_ecallisto_website.png", height="50px"
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
