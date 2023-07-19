import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html
import plotly.express as px
from database_functions import timebucket_values_from_database_sql, sql_result_to_df, fill_missing_timesteps_with_nan, get_table_names_sql
from plotting_utils import timedelta_to_sql_timebucket_value, plot_spectogram
import pandas as pd
from datetime import datetime, timedelta

app = dash.Dash(__name__)

# Define some constants
RESOLUTION_WIDTH = 480

# Create an empty list to keep track of the zoom levels.
figures_zoom = []

# Define the layout of the app
app.layout = html.Div([
    html.Div([
        dcc.DatePickerRange(
            id='date-picker-range',
            start_date=datetime.now() - timedelta(days=14),
            end_date=datetime.now()
        ),
    ], style={'width': '48%', 'display': 'inline-block'}),

    html.Div([
        dcc.Dropdown(
            id='instrument-dropdown',
            options=[{'label': i, 'value': i} for i in get_table_names_sql()],
            value='austria_unigraz_01',
        ),
    ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),

    html.Button('Show Data', id='show-data-button', n_clicks=0),

    dcc.Graph(
        id='live-graph',
        style={'display': 'none'}
    ),
    html.Button('Zoom Out', id='zoom-out-button', n_clicks=0, style={'display': 'none'}),
    html.Div(id='hidden-div', style={'display': 'none'}),
    dcc.Store(id='zoom-history')  # hidden store to keep track of zoom history
])

# This is the callback function that updates the graph whenever the date range, 
# instrument or zoom level is changed by the user.
@app.callback(
    [Output('live-graph', 'figure'), Output('live-graph', 'style'),
     Output('zoom-out-button', 'style'), Output('zoom-history', 'data')],
    [Input('show-data-button', 'n_clicks'),
     Input('live-graph', 'relayoutData'),
     Input('zoom-out-button', 'n_clicks')],
    [State('instrument-dropdown', 'value'),
     State('date-picker-range', 'start_date'),
     State('date-picker-range', 'end_date'),
     State('zoom-history', 'data')]
)

def update_graph(n_clicks, rangeselector, zoom_clicks, instrument, start_date, end_date, zoom_history):
    # The context provides information about which Input triggered the callback.
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # If the "Show Data" button is clicked, we update the graph based on the chosen date range
    # and the selected instrument.
    if trigger_id == 'show-data-button' and n_clicks > 0:
        start_datetime = pd.to_datetime(start_date)
        end_datetime = pd.to_datetime(end_date)

    # If the user changed the zoom level, we update the graph accordingly. The new start and end
    # dates are taken directly from the range selected by the user.
    elif trigger_id == 'live-graph' and 'xaxis.range[0]' in rangeselector:
        start_datetime = pd.to_datetime(rangeselector['xaxis.range[0]'])
        end_datetime = pd.to_datetime(rangeselector['xaxis.range[1]'])

    # If the "Zoom Out" button is clicked, we restore the previous zoom level.
    # This is achieved by keeping a history of zoom levels (figures) in a list. The most recent 
    # figure is popped out from the list, thus restoring the previous state.
    elif trigger_id == 'zoom-out-button' and zoom_clicks > 0:  # New condition
        if zoom_history and len(zoom_history) > 1:
            zoom_history.pop()
            return zoom_history[-1], {'display': 'block'}, {'display': 'block'}, zoom_history  # return the latest figure
        else:
            # If there is no zoom history or it contains only a single figure, we do nothing.
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    else:
        return dash.no_update

    # We calculate the difference between the start and end times and use it to fetch data
    # with the appropriate resolution from the database.
    time_delta = (end_datetime - start_datetime) / RESOLUTION_WIDTH
    timebucket_value = timedelta_to_sql_timebucket_value(time_delta)

    # Fetch new data based on the start and end dates
    query = {
        "table": instrument,
        "start_datetime": start_datetime.strftime("%Y-%m-%d %H:%M:%S.%f"),
        "end_datetime": end_datetime.strftime("%Y-%m-%d %H:%M:%S.%f"),
        "timebucket": f"{timebucket_value}",
        "agg_function": "MAX"
    }

    # Fetch new data based on the start and end dates
    sql_result = timebucket_values_from_database_sql(**query)
    df = sql_result_to_df(sql_result, datetime_col='datetime')
    df = fill_missing_timesteps_with_nan(df)

    # Create figure
    fig = plot_spectogram(df, instrument, start_datetime, end_datetime)

    # Add the new figure to the history
    zoom_history = zoom_history or []
    zoom_history.append(fig)

    # Determine the style of the figure and the zoom button
    fig_style = {'display': 'block'} if fig else {'display': 'none'}
    zoom_button_style = {'display': 'block'} if zoom_history and len(zoom_history) > 1 else {'display': 'none'}

    return fig, fig_style, zoom_button_style, zoom_history

if __name__ == '__main__':
    app.run_server(debug=True)