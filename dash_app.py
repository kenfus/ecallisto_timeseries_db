import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html
from database_functions import timebucket_values_from_database_sql, sql_result_to_df, get_table_names_sql, check_if_table_has_data_between_dates_sql, get_table_names_with_data_between_dates_sql
from plotting_utils import timedelta_to_sql_timebucket_value
from ecallisto_ng.plotting.utils import plot_spectogram, fill_missing_timesteps_with_nan
from ecallisto_ng.data_processing.utils import subtract_rolling_background, elimwrongchannels
import pandas as pd
from datetime import datetime, timedelta
import dash_bootstrap_components as dbc
from dash_utils import generate_nav_bar

# Use a Bootstrap stylesheet
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
navbar = generate_nav_bar()

# Define some constants
RESOLUTION_WIDTH = 720

# for gunicorn
server = app.server

# Generate some constants
def generate_options_instrument(list_of_instruments):
    if len(list_of_instruments) == 0:
        return []
    options_instrument = [{'label': i, 'value': i} for i in list_of_instruments]
    # Sort by label
    options_instrument = sorted(options_instrument, key=lambda k: k['label'])
    # Add Top 3 instruments
    options_instrument.insert(0, {'label': 'Top 5 instruments', 'value': 'top5'})
    return options_instrument

options_instrument = generate_options_instrument(get_table_names_sql())

# Define the layout of the app
app.layout = html.Div([
    navbar,
    dbc.Col(dbc.NavbarBrand("eCallisto radiospectrograms", className="ml-2", 
                            style={'font-size': '3em'})),  # Increase font size
    dbc.Row([
        dbc.Col(
            html.P(["The solar radio spectrograms that can be retrieved through this interface are provided by the network e-Callisto. ",
                    html.Br(), 
                   "This network consists of a common receiver, a CALLISTO spectrometer, that are installed on radio antennas spread around the globe. ",
                    html.Br(), 
                   "They all observe the full Sun from diverse latitudes and longitudes. Due to the spreading, the network reaches a 24/7 observing time coverage.",
                   ], style={'font-size': '1em', 'margin-top': '10px'}),  # Add description
        ),
    ]),
    html.H3("User Usage"),  # Add title for user-focused section
    dbc.Row([
        dbc.Col(
            html.P(["When selecting `Top 5 instruments`, the five instruments with the highest signal-to-noise ratio are selected."
                ], style={'font-size': '1em', 'margin-top': '10px'}),  # Add user usage information
        ),
    ]),
    html.Div([
        html.Div([
            dcc.DatePickerRange(
                id='date-picker-range',
                start_date=datetime.now() - timedelta(days=5),
                end_date=datetime.now()
            ),
        ], style={'width': '100%', 'display': 'block', 'margin-top': '60px'}),

        html.Div(id='instrument-and-load-button-container',  # This is the new Div container
            children=[
                dcc.Store(id='instrument-loading-state', data=False),  # initially not loading
                dcc.Loading(
                    id="loading-instrument-dropdown",
                    type="default",
                    children=[
                        html.Div([
                            dcc.Dropdown(
                                id='instrument-dropdown',
                                options=options_instrument,
                                value='top5',
                                multi=True
                            ),
                        ], style={'width': '100%', 'display': 'block'}),

                        html.Div(id='load-data-button-container',
                                children=[
                                    html.Button('Load Data', id='load-data-button', n_clicks=0)
                                ],
                                ),
                    ]
                ),
            ],
            style={'display': 'block'}  # Initially, the container is visible
        ),
    ]),
    html.Div(
        id='graphs-container',
        children=[]
    )
])
@app.callback(
    [Output('instrument-dropdown', 'options'),
     Output('instrument-loading-state', 'data')],
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_instrument_dropdown_options(start_datetime, end_datetime):
    # initially set loading state to True
    loading_state = True
    options = []
    if start_datetime and end_datetime:
        available_instruments = get_table_names_with_data_between_dates_sql(start_datetime, end_datetime)
        options = generate_options_instrument(available_instruments)
        loading_state = False  # set loading state to False after options are loaded
    return options, loading_state
# This is the callback function that updates the graphs whenever the date range or instrument is changed by the user.
@app.callback(
    Output('graphs-container', 'children'),
    [Input('load-data-button', 'n_clicks')],
    [State('instrument-dropdown', 'value'),
     State('date-picker-range', 'start_date'),
     State('date-picker-range', 'end_date')]
)
def update_graph(n_clicks, instruments, start_date, end_date):
    if n_clicks > 0:
        start_datetime = pd.to_datetime(start_date)
        end_datetime = pd.to_datetime(end_date)
        time_delta = (end_datetime - start_datetime) / RESOLUTION_WIDTH
        timebucket_value = timedelta_to_sql_timebucket_value(time_delta)

        if isinstance(instruments, str):
            instruments = [instruments]
        if instruments == ['top5']:
            instruments = ['australia_assa_62', 'mongolia_ub_01', 'austria_unigraz_01', 'germany_dlr_63', 'alaska_anchorage_01'] # Replace in future with top 5 instruments with highest signal-to-noise ratio

        graphs = []
        for instrument in instruments:
            query = {
                "table": instrument,
                "start_datetime": start_datetime.strftime("%Y-%m-%d %H:%M:%S.%f"),
                "end_datetime": end_datetime.strftime("%Y-%m-%d %H:%M:%S.%f"),
                "timebucket": f"{timebucket_value}",
                "agg_function": "MAX"
            }
            if check_if_table_has_data_between_dates_sql(query['table'], query['start_datetime'], query['end_datetime']) == False:
                warning_message = f"No data available for instrument '{instrument}' within the specified time frame."
                warning_div = html.Div(warning_message, style={'color': 'red', 'margin-top': '10px'})
                graphs.append(warning_div)
                continue
            # Create data
            sql_result = timebucket_values_from_database_sql(**query)
            df = sql_result_to_df(sql_result, datetime_col='datetime')
            df = fill_missing_timesteps_with_nan(df)
            # Background subtraction
            df = elimwrongchannels(df)
            df = subtract_rolling_background(df, window_size=30)
            fig = plot_spectogram(df, instrument, start_datetime, end_datetime)
            fig_style = {'display': 'block'} if fig else {'display': 'none'}
            graph = dcc.Graph(
                id=f'live-graph-{instrument}',
                figure=fig,
                config={'displayModeBar': False},
                style=fig_style,
                responsive=True
            )
            graphs.append(graph)

        return graphs
    return []

# Run the app
if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=8050)
