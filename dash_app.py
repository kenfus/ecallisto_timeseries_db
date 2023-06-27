import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html
import plotly.express as px
from database_functions import timebucket_values_from_database_sql, sql_result_to_df, fill_missing_timesteps_with_nan
from plotting_utils import timedelta_to_sql
import pandas as pd

app = dash.Dash(__name__)

# Assuming df is your DataFrame with the initial data.
RESOLUTION_WIDTH = 720
query = {
    "table": "austria_unigraz_01",
    "start_datetime": '2021-03-01 04:30:00',
    "end_datetime": '2021-03-14 23:30:00',
    "timebucket": "1h",
    "agg_function": "MAX"
}  # Also the "base query"

# Fetch initial data
sql = timebucket_values_from_database_sql(**query)
df = sql_result_to_df(sql, datetime_col='datetime')
df = fill_missing_timesteps_with_nan(df)

fig = px.imshow(df.T.iloc[::-1])

app.layout = html.Div([
    dcc.Graph(
        id='live-graph',
        figure=fig
    ),
    html.Div(id='hidden-div', style={'display': 'none'})
])

@app.callback(
    Output('live-graph', 'figure'),
    [Input('live-graph', 'relayoutData')],
    [State('live-graph', 'figure')]
)
def update_graph(rangeselector, current_figure):
    if rangeselector is not None:
        if 'xaxis.range[0]' in rangeselector:
            start_datetime = pd.to_datetime(rangeselector['xaxis.range[0]'])
            end_datetime = pd.to_datetime(rangeselector['xaxis.range[1]'])

            print(f"Start: {start_datetime}, End: {end_datetime}")

            # Time difference
            time_delta = (end_datetime - start_datetime) / RESOLUTION_WIDTH
            timebucket_value = timedelta_to_sql(time_delta)

            print(timebucket_value)

            # To string
            start_datetime = start_datetime.strftime("%Y-%m-%d %H:%M:%S.%f")
            end_datetime = end_datetime.strftime("%Y-%m-%d %H:%M:%S.%f")

            # Fetch new data based on the start and end dates
            query = {
                "table": "austria_unigraz_01",
                "start_datetime": start_datetime,
                "end_datetime": end_datetime,
                "timebucket": f"{timebucket_value}",
                "agg_function": "MAX"
            }
            sql = timebucket_values_from_database_sql(**query)
            df = sql_result_to_df(sql, datetime_col='datetime')
            df = fill_missing_timesteps_with_nan(df)

            # Update figure
            current_figure['data'] = px.imshow(df.T.iloc[::-1]).data

            return current_figure
        
        elif 'xaxis.autorange' in rangeselector:
            return fig

    return dash.no_update

if __name__ == '__main__':
    app.run_server(debug=True)
