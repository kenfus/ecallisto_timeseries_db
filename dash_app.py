import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html
from database_functions import timebucket_values_from_database_sql, sql_result_to_df, fill_missing_timesteps_with_nan, get_table_names_sql, check_if_table_has_data_between_dates_sql
from plotting_utils import timedelta_to_sql_timebucket_value, plot_spectogram
import pandas as pd
from datetime import datetime, timedelta
import dash_bootstrap_components as dbc
from dash_utils import generate_nav_bar

# Use a Bootstrap stylesheet
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

navbar = generate_nav_bar()

# Define some constants
RESOLUTION_WIDTH = 720

# Generate some constants
options_instrument = [{'label': i, 'value': i} for i in get_table_names_sql()]
# Sort by label
options_instrument = sorted(options_instrument, key=lambda k: k['label'])
# Add Top 3 instruments
options_instrument.insert(0, {'label': 'Top 5 instruments', 'value': 'top5'})

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
                   html.Br(), 
                   "When selecting `Top 5 instruments` the three instruments with the highest signal-to-noise ratio are selected."
                   ], style={'font-size': '1em', 'margin-top': '10px'}),  # Add description
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

        html.Div([
            dcc.Dropdown(
                id='instrument-dropdown',
                options=options_instrument,
                value='top5',
                multi=True
            ),
        ], style={'width': '100%', 'display': 'block'}),

        html.Div([
            html.Button('Show Data', id='show-data-button', n_clicks=0),
        ], style={'width': '100%', 'display': 'block', 'margin-top': '10px'}),
    ]),
    html.Div(
        id='graphs-container',
        children=[]
    )
])

# This is the callback function that updates the graphs whenever the date range or instrument is changed by the user.
@app.callback(
    Output('graphs-container', 'children'),
    [Input('show-data-button', 'n_clicks')],
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
            instruments = ['australia_assa_62', 'triest_60', 'austria_unigraz_01', 'swiss_landschlacht_63', 'alaska_anchorage_01'] # Replace in future with top 5 instruments with highest signal-to-noise ratio

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
            sql_result = timebucket_values_from_database_sql(**query)
            df = sql_result_to_df(sql_result, datetime_col='datetime')
            df = fill_missing_timesteps_with_nan(df)
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
    app.run_server(debug=True)
