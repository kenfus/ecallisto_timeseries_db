import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html
import plotly.express as px
from database_functions import timebucket_values_from_database_sql, sql_result_to_df, fill_missing_timesteps_with_nan, get_table_names_sql
from plotting_utils import timedelta_to_sql_timebucket_value
import pandas as pd

app = dash.Dash(__name__)

# Assuming df is your DataFrame with the initial data.
RESOLUTION_WIDTH = 720

# Create list with figures to allow for auto zoom
figures_zoom = []

app.layout = html.Div([
    html.Div([
        dcc.DatePickerRange(
            id='date-picker-range',
            start_date=pd.to_datetime('2023-06-16 04:30:00'),
            end_date=pd.to_datetime('2023-06-30 23:30:00')
        ),
    ], style={'width': '48%', 'display': 'inline-block'}),

    html.Div([
        dcc.Dropdown(
            id='instrument-dropdown',
            options=[{'label': i, 'value': i} for i in get_table_names_sql()],
            value='austria_unigraz_1',
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
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger_id == 'show-data-button' and n_clicks > 0:
        start_datetime = pd.to_datetime(start_date)
        end_datetime = pd.to_datetime(end_date)
    elif trigger_id == 'live-graph' and 'xaxis.range[0]' in rangeselector:
        start_datetime = pd.to_datetime(rangeselector['xaxis.range[0]'])
        end_datetime = pd.to_datetime(rangeselector['xaxis.range[1]'])
    elif trigger_id == 'zoom-out-button' and zoom_clicks > 0:  # New condition
        if zoom_history and len(zoom_history) > 1:
            zoom_history.pop()
            return zoom_history[-1], {'display': 'block'}, {'display': 'block'}, zoom_history  # return the latest figure
        else:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    else:
        return dash.no_update

    # Time difference
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
    sql = timebucket_values_from_database_sql(**query)
    df = sql_result_to_df(sql, datetime_col='datetime')
    df = fill_missing_timesteps_with_nan(df)

    # Create figure
    fig = px.imshow(df.T.iloc[::-1])
    zoom_history = zoom_history or []
    zoom_history.append(fig)

    # Determine the style of the figure and the zoom button
    fig_style = {'display': 'block'} if fig else {'display': 'none'}
    zoom_button_style = {'display': 'block'} if zoom_history and len(zoom_history) > 1 else {'display': 'none'}

    return fig, fig_style, zoom_button_style, zoom_history

if __name__ == '__main__':
    app.run_server(debug=True)