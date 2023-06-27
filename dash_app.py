import dash
from dash.dependencies import Input, Output
from dash import dcc, html
import plotly.express as px
from database_functions import timebucket_values_from_database_sql, values_from_database_sql, sql_result_to_df, fill_missing_timesteps_with_nan
from plotting_utils import calculate_timedelta_from_strings, timedelta_to_sql
import pandas as pd

app = dash.Dash(__name__)

# Assuming df is your DataFrame with the initial data.
RESOLUTION_WIDTH = 720
query = {
    "table": "austria_unigraz_01",
    "start_datetime": '2021-03-10 06:30:00',
    "end_datetime": '2021-03-14 23:30:00',
    "timebucket": "30m",
    "agg_function": "MAX"
} # Also the "base query"

# 1 version alle und liste.
# Todo: timezone pro instrument. Tageslänge pro lat/lon.
sql = timebucket_values_from_database_sql(**query)
df = sql_result_to_df(sql, datetime_col='datetime')
df = fill_missing_timesteps_with_nan(df)

fig = px.imshow(df.T.iloc[::-1])


app.layout = html.Div([
    dcc.Graph(
        id='live-graph',
        figure=fig
    ),
    html.Div(id='hidden-div', style={'display':'none'})
])

@app.callback(
    Output('live-graph', 'figure'),
    [Input('live-graph', 'relayoutData')])
def update_graph(rangeselector):
    if rangeselector is not None:
        if 'xaxis.range[0]' in rangeselector:
            start_datetime = rangeselector['xaxis.range[0]']
            end_datetime = rangeselector['xaxis.range[1]']

            # Time difference
            start_datetime = pd.to_datetime(start_datetime)
            end_datetime = pd.to_datetime(end_datetime)
            time_delta = calculate_timedelta_from_strings(start_datetime, end_datetime) / RESOLUTION_WIDTH
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
            print(df.head())

            # Update figure
            fig = px.imshow(df.T.iloc[::-1])
            fig.update_layout(
                title=f"Spectogram from {start_datetime} to {end_datetime}",
                xaxis_title="Datetime",
                yaxis_title="Frequency",
                font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"),
            )

            return fig
    else:
        return dash.no_update

if __name__ == '__main__':
    app.run_server(debug=True)
