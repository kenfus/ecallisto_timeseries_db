from datetime import timedelta
import pandas as pd
import plotly.express as px

def plot_spectogram(df, instrument_name, start_datetime, end_datetime, size=18):
    fig = px.imshow(df.T.iloc[::-1])
    fig.update_layout(
        title=f"Spectogram of {instrument_name} from {start_datetime} to {end_datetime}",
        xaxis_title="Datetime",
        yaxis_title="Frequency",
        font=dict(family="Courier New, monospace", size=size, color="#7f7f7f"),
    )
    return fig


def plot_background_image(df, instrument_name, end_time, length, timebucket, size=18):
    # Fill missing hours with NaN

    fig = px.imshow(df.T, aspect="auto")
    fig.update_layout(
        title=f"Background Image of {instrument_name}. Length: {length}. End time: {end_time}. Time bucket: {timebucket}",
        xaxis_title="Time",
        yaxis_title="Frequency",
        font=dict(family="Courier New, monospace", color="#7f7f7f"),
    )

    return fig


def add_burst_to_spectogram(fig, burst_start, burst_end, type, size=18):
    fig.add_vrect(
        x0=burst_start,
        x1=burst_end + timedelta(minutes=1),
        opacity=1,
        annotation_text=f"Type: {type}",
        annotation_position="top left",
        annotation_font_size=size,
    )

    return fig


def plot_spectogram_with_burst(
    df, instrument_name, start_datetime, end_datetime, burst_start, burst_end, size=18
):
    fig = plot_spectogram(df, instrument_name, start_datetime, end_datetime, size)
    fig = add_burst_to_spectogram(fig, burst_start, burst_end, size)

    return fig

def calculate_timedelta_from_strings(start, end):
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    return end - start

def timedelta_to_sql(timedelta):
    # Convert to seconds
    seconds = timedelta.total_seconds()

    # Convert to SQL-compatible value
    if seconds >= 86400:  # More than 1 day
        days = seconds / 86400
        sql_value = f"{int(days)} d" if days.is_integer() else f"{days:.1f} d"
    elif seconds >= 3600:  # More than 1 hour
        hours = seconds / 3600
        sql_value = f"{int(hours)} h" if hours.is_integer() else f"{hours:.1f} h"
    elif seconds >= 60:  # More than 1 minute
        minutes = seconds / 60
        sql_value = f"{int(minutes)} min" if minutes.is_integer() else f"{minutes:.1f} min"
    else:
        sql_value = f"{seconds:.1f} s"

    return sql_value
